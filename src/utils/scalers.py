"""
Standardize tensors in PyTorch with feature- or channel-wise statistics.

You choose the reduction dimensions (e.g., dims=(0,) for per-feature over batch,
or dims=(0,2,3) for per-channel on image tensors shaped [N, C, H, W]).

Device behavior:
- The output of `transform(x)` (and `fit_transform`) stays on the same device as `x`
  (CPU ↔️ GPU). We move the fitted stats to `x.device` for the computation.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn

Dims = Union[int, Tuple[int, ...]]
ArrayLike = Union[torch.Tensor, np.ndarray]

class StandardScaler(nn.Module):
    """
    A simple standardizer for PyTorch tensors.

    The scaler computes mean and (population) standard deviation over the
    specified dimensions and applies (x - mean) / std with safe clamping.

    Args:
        dims (int | tuple[int, ...]): Dimensions to reduce over when computing
            statistics. For tabular data [N, F], use dims=(0,). For images
            [N, C, H, W], a common choice is dims=(0, 2, 3) to standardize
            per-channel.
        eps (float): Minimum standard deviation to avoid division by zero.
    """

    def __init__(self, dims: Dims = (0,), eps: float = 1e-6) -> None:
        super().__init__()
        if isinstance(dims, int):
            dims = (dims,)
        self.dims: Tuple[int, ...] = tuple(dims)
        self.eps: float = float(eps)

        # Buffers so they move with .to(device) and save in state_dict
        self.register_buffer("mean_", None)  # type: Optional[torch.Tensor]
        self.register_buffer("std_", None)   # type: Optional[torch.Tensor]

    def _as_float_tensor(self, x: ArrayLike) -> torch.Tensor:
        """
        Convert input to a floating-point torch.Tensor on its current device.

        If `x` is a NumPy array, it will be created on CPU.
        """
        t = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
        if not t.is_floating_point():
            t = t.float()
        return t

    def fit(self, x: ArrayLike) -> "StandardScaler":
        """
        Compute statistics over `self.dims` on the device of `x`.
        """
        xt = self._as_float_tensor(x)
        mean = xt.mean(dim=self.dims, keepdim=True)
        std = xt.std(dim=self.dims, unbiased=False, keepdim=True).clamp_min(self.eps)
        self.mean_ = mean
        self.std_ = std
        return self

    @torch.no_grad()
    def transform(self, x: ArrayLike) -> torch.Tensor:
        """
        Standardize using the fitted statistics.

        Output stays on the same device as `x`.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")

        xt = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
        compute_dtype = xt.dtype if xt.is_floating_point() else torch.float32
        xt = xt.to(dtype=compute_dtype)

        # Move buffers to the input's device (CPU/GPU) for computation
        mean = self.mean_.to(device=xt.device, dtype=compute_dtype)
        std = self.std_.to(device=xt.device, dtype=compute_dtype)

        return (xt - mean) / std

    def inverse_transform(self, x: ArrayLike) -> torch.Tensor:
        """
        Inverse the standardization using the fitted statistics.

        Output stays on the same device as `x`.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling inverse_transform().")

        xt = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
        compute_dtype = xt.dtype if xt.is_floating_point() else torch.float32
        xt = xt.to(dtype=compute_dtype)

        # Move buffers to the input's device (CPU/GPU) for computation
        mean = self.mean_.to(device=xt.device, dtype=compute_dtype)
        std = self.std_.to(device=xt.device, dtype=compute_dtype)

        return xt * std + mean

    def fit_transform(self, x: ArrayLike) -> torch.Tensor:
        """
        Fit to `x` and return the standardized result (on `x.device`).
        """
        self.fit(x)
        return self.transform(x)

    def extra_repr(self) -> str:
        return f"dims={self.dims}, eps={self.eps}"
