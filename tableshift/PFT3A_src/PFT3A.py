from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import tableshift.models.torchutils
import torch
import torch.nn as nn
import torch.jit
import logging
from sklearn.metrics import pairwise_distances
import numpy as np
import math
import torch.nn.functional as F
import numpy as np

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class PFT3A(nn.Module):

    def __init__(self, model, optimizer_type, prior, lr_list = [1e-4,5e-4,1e-5], device=None, smooth_factor=0.1):
        super().__init__()
        self.base_model1 = deepcopy(model)
        self.model_list = [self.base_model1]
        self.optimizer_list = [optimizer_type(self.base_model1.parameters(), lr=lr_list[0])]
        self.prior = prior
        self.source_y = prior
        self.smooth_factor = smooth_factor
        self.device = device
        if not device:
            self.device = f"cuda:{torch.cuda.current_device()}" \
            if torch.cuda.is_available() else "cpu"
        logging.info(f"device is {self.device}")

        self._feat_cache = [None] * len(self.model_list)
        for idx, m in enumerate(self.model_list):
            head = self._resolve_feat_head(m)
            if head is None:
                raise RuntimeError(
                    f"PFT3A: cannot locate a feature head on {type(m).__name__}; "
                    "expected one of attributes "
                    "{mlp, blocks, head, transformer.head.linear}")
            head.register_forward_pre_hook(self._make_feat_hook(idx))
        self.last_remain_count = 0
        self.last_batch_size = 0

    def _make_feat_hook(self, idx):
        def hook(module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            self._feat_cache[idx] = x
        return hook

    @staticmethod
    def _resolve_feat_head(m):
        # Hook target is whichever submodule's forward-pre-hook input is the
        # per-sample feature vector (B, D). MLP/TabTransformer expose it
        # directly at the top level; FT-Transformer's classifier sits at
        # transformer.head.linear (input shape (B, d_token) = the [CLS] token).
        head = getattr(m, "mlp", None) or getattr(m, "blocks", None) \
               or getattr(m, "head", None)
        if head is not None:
            return head
        transformer = getattr(m, "transformer", None)
        if transformer is not None:
            t_head = getattr(transformer, "head", None)
            if t_head is not None:
                return getattr(t_head, "linear", t_head)
            return getattr(transformer, "blocks", None)
        return None

    def suit_neighbors(self, samples, logits, distance):
        samples = samples.detach().float()
        sample_distance = torch.cdist(samples, samples, p=2)
        if distance is None:
            distance = sample_distance.mean()
        pseudo_label = logits.argmax(dim=1).float()
        near = (sample_distance <= distance).float()
        is_remain = (pseudo_label.unsqueeze(0) * near).sum(dim=1) \
                    / near.sum(dim=1).clamp_min(1)
        keep = (is_remain - pseudo_label).abs() <= 0.3  # beta = 0.7
        return torch.nonzero(keep, as_tuple=False).flatten()


    @torch.enable_grad()
    def online_logits(self, out_list):
        factor = torch.zeros(size=(1,1),dtype=torch.float32)
        for i, out in enumerate(out_list):
            factor[0][i] = 1 - (softmax_entropy(out) * abs(out[:,1] - out[:,0])).mean()
        factor = F.normalize(factor, p=1, dim=1)
        return  out_list[0] * factor[0][0]
        

    def get_prior(self, x):
        out_logits = []
        for models in self.model_list:
            outputs = tableshift.models.torchutils.apply_model(models, x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            # change softmax outputs
            two_shape = 1 - torch.sigmoid(outputs)
            outputs = torch.cat([two_shape, torch.sigmoid(outputs)], dim=1)
            out_logits.append(outputs)
        final = self.online_logits(out_logits)
        logits_p = F.normalize(final * self.prior / self.source_y, p=1)
        condition_estimation = logits_p[softmax_entropy(logits_p) < softmax_entropy(torch.tensor([[0.7,1-0.7]]).to(self.device))]
        source_y_estimation = torch.mean(condition_estimation, dim=0)

        self.source_y = source_y_estimation

        condition_estimation = logits_p[softmax_entropy(logits_p) > softmax_entropy(torch.tensor([[0.7,1-0.7]]).to(self.device))]
        target_y_estimation = torch.mean(condition_estimation, dim=0)

        self.prior = target_y_estimation
  
    def diagonal_gaussian_kl_loss(self, m1, v1, m2, v2):
        loss = (v2.log() - v1.log() + (v1 + (m2 - m1).square()) / (v2 + 1.0e-8) - 1) / 2
        return loss.mean()
    
    def forward(self, x, gt):
        out_logits = []
        out_features = []
        for idx, models in enumerate(self.model_list):
            outputs = tableshift.models.torchutils.apply_model(models, x)
            if isinstance(outputs, tuple):
                outputs, f = outputs[0], outputs[1]
            else:
                f = self._feat_cache[idx]
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            two_shape = 1 - torch.sigmoid(outputs)
            outputs = torch.cat([two_shape, torch.sigmoid(outputs)], dim=1)
            out_logits.append(outputs)
            out_features.append(f)
        final = self.online_logits(out_logits)
        logits_p = F.normalize(final * self.prior / self.source_y, p=1)

        normalized_out_logits = [F.normalize(logit * self.prior.detach() / self.source_y, p=1, dim=1) for logit in out_logits]
        
        features = torch.cat(out_features)
        s_features = features[softmax_entropy(logits_p) < softmax_entropy(torch.tensor([[0.7,1-0.7]]).to(self.device))]
        t_features = features[softmax_entropy(logits_p) > softmax_entropy(torch.tensor([[0.7,1-0.7]]).to(self.device))]
        s_mean = s_features.mean(dim=0).cuda()
        s_features_c = s_features - s_mean
  
        s_cov = s_features_c.T @ s_features_c / (s_features.shape[0] - 1)
        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.detach())
        # eigh returns eigenvalues in ascending order; pick the top-5 largest.
        s_pca_basis = s_eigvecs[:, -5:].float()
        s_pc_vars = s_eigvals[-5:].float()


        t_f_pc = (t_features.cuda() - s_mean) @ s_pca_basis.cuda()
        t_f_pc_mean = t_f_pc.mean(dim=0).cuda()    
        t_f_pc_var = t_f_pc.var(dim=0).cuda()      
        zeros = torch.zeros_like(t_f_pc_mean).cuda()
        kl_loss = self.diagonal_gaussian_kl_loss(t_f_pc_mean, t_f_pc_var, zeros, s_pc_vars)

        remain_index = self.suit_neighbors(x, final, None)
        self.last_remain_count = int(len(remain_index))
        self.last_batch_size = int(x.shape[0])
        for logits_back, optimizer in zip(out_logits, self.optimizer_list):
            logits_back = logits_back
            if len(remain_index) > 0:
                loss = (softmax_entropy(logits_back)
                        * abs(logits_back[:,0] - logits_back[:,1])).mean(0)*1+ kl_loss*0.1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        y_hat = logits_p.argmax(axis=1)
        condition = logits_p[softmax_entropy(logits_p) < softmax_entropy(torch.tensor([[0.7,1-0.7]]).to(self.device))]
        A_pre = torch.stack([
            logits_p[y_hat == 0].mean(dim=0).detach(),
            logits_p[y_hat == 1].mean(dim=0).detach(),
        ])
        B_acc = torch.mean(condition, dim=0)


        prior_fac = torch.linalg.inv(A_pre) @ B_acc
        if not torch.any(torch.isnan(prior_fac)):
            self.prior = F.softmax((self.prior - self.smooth_factor * prior_fac), dim=0)

        adapt_samples = torch.where(abs(final[:, 0] - final[:, 1]) > 0, 1, 0)
        final[adapt_samples == 1] = F.normalize(final[adapt_samples == 1] * self.prior / self.source_y, p=1)
 
        return final[:, 1].detach()

