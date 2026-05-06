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

    def suit_neighbors(self, samples, logits, distance):
        sample_distance = pairwise_distances(samples.detach())
        if distance is None:
            distance = sample_distance.mean()
        pseudo_label = logits.argmax(axis=1)
        remain_index = [] # samples selected to join the model update
        for i, s in enumerate(sample_distance):
            near_neighbor = np.where(s > distance, 0, 1)
            is_remain = ((pseudo_label * near_neighbor).sum()) / near_neighbor.sum()
            if abs(is_remain - pseudo_label[i]) > 0.3:# here beta = 0.7
                # consistent with plabel
                continue
            else:
                remain_index.append(i)
        return torch.tensor(remain_index)


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
            outputs, f = tableshift.models.torchutils.apply_model(models, x)
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
        for models in self.model_list:
            outputs, f = tableshift.models.torchutils.apply_model(models, x)
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
        s_cov_np = s_cov.detach().cpu().numpy()
        s_eigvals, s_eigvecs = np.linalg.eigh(s_cov_np)
        s_eigvals = torch.from_numpy(s_eigvals)
        s_eigvecs = torch.from_numpy(s_eigvecs)
        s_eigval_topk_set = np.argsort(s_eigvals.numpy())[-5:]
        s_indices = torch.from_numpy(s_eigval_topk_set).long()
        s_pca_basis = s_eigvecs[:, s_indices].float()  
        s_pc_vars = s_eigvals[s_indices].float().cuda()


        t_f_pc = (t_features.cuda() - s_mean) @ s_pca_basis.cuda()
        t_f_pc_mean = t_f_pc.mean(dim=0).cuda()    
        t_f_pc_var = t_f_pc.var(dim=0).cuda()      
        zeros = torch.zeros_like(t_f_pc_mean).cuda()
        kl_loss = self.diagonal_gaussian_kl_loss(t_f_pc_mean, t_f_pc_var, zeros, s_pc_vars)

        remain_index = self.suit_neighbors(x.cpu(), final.cpu(), None).to(self.device)
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
        A_pre = torch.tensor([torch.mean(logits_p[y_hat == 0], dim=0).detach().cpu().numpy(),
                              torch.mean(logits_p[y_hat == 1], dim=0).detach().cpu().numpy()]).to(self.device)
        B_acc = torch.mean(condition, dim=0)


        prior_fac = torch.linalg.inv(A_pre) @ B_acc
        if not torch.any(torch.isnan(prior_fac)):
            self.prior = F.softmax((self.prior - self.smooth_factor * prior_fac), dim=0)

        adapt_samples = torch.where(abs(final[:, 0] - final[:, 1]) > 0, 1, 0)
        final[adapt_samples == 1] = F.normalize(final[adapt_samples == 1] * self.prior / self.source_y, p=1)
 
        return final[:, 1].detach()

