"""Tent-style PFT3A variant: TTA only updates Norm-layer affine params.

Subclasses :class:`tableshift.PFT3A_src.PFT3A.PFT3A`. Forward / loss / prior
update logic is unchanged; the only difference is that during ``__init__``
all parameters of the cloned model are frozen *except* LayerNorm / BatchNorm
/ GroupNorm ``weight`` and ``bias``, and the optimizer is rebuilt over just
those tensors. This mitigates mode collapse on highly class-imbalanced TTA
tasks where full-network entropy minimization tends to drive predictions to
a single class (cf. Tent / SAR / EATA).
"""

import torch.nn as nn

from tableshift.PFT3A_src.PFT3A import PFT3A as _BasePFT3A


_NORM_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm,
)


class PFT3A(_BasePFT3A):
    def __init__(self, model, optimizer_type, prior,
                 lr_list=[1e-4, 5e-4, 1e-5], device=None, smooth_factor=0.1):
        super().__init__(model=model, optimizer_type=optimizer_type,
                         prior=prior, lr_list=lr_list, device=device,
                         smooth_factor=smooth_factor)

        norm_params = []
        for sub in self.base_model1.modules():
            if isinstance(sub, _NORM_TYPES):
                if getattr(sub, "weight", None) is not None:
                    norm_params.append(sub.weight)
                if getattr(sub, "bias", None) is not None:
                    norm_params.append(sub.bias)

        norm_ids = {id(p) for p in norm_params}
        n_total = n_train = 0
        for p in self.base_model1.parameters():
            n_total += 1
            if id(p) in norm_ids:
                p.requires_grad_(True)
                n_train += 1
            else:
                p.requires_grad_(False)

        # Discard the full-parameter optimizer built by the parent and replace
        # it with one that only sees Norm affine params.
        self.optimizer_list = [optimizer_type(norm_params, lr=lr_list[0])]
        print(f"[PFT3A_ln] LN-only TTA: {n_train}/{n_total} "
              f"trainable param tensors (others frozen)")
