import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        # Phase 2 evaluated a Triton add_rms_norm path for large row counts;
        # committed process-level A/B runs showed no engine-level benefit, so
        # the default runtime keeps the compiled PyTorch baseline. The Triton
        # kernel remains available as a standalone experiment (add_rms_norm).
        return self.add_rms_forward(x, residual)


import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    H,
    eps,
    stride_m,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    scale = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    # Match baseline semantics: round (x*scale) back to the storage dtype,
    # then the weight multiply runs in fp32 (weight is an fp32 parameter).
    y = (x * scale).to(DTYPE).to(tl.float32) * w
    tl.store(y_ptr + row * stride_m + offs, y, mask=mask)


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    y_ptr,
    new_residual_ptr,
    M,
    H,
    eps,
    stride_m,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * stride_m + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    s = x + r
    # Baseline stores the fp32 sum cast back to the storage dtype as the new
    # residual (pre-normalization), not the normalized value.
    tl.store(new_residual_ptr + row * stride_m + offs, s.to(DTYPE), mask=mask)
    var = tl.sum(s * s, axis=0) / H
    scale = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = (s * scale).to(DTYPE).to(tl.float32) * w
    tl.store(y_ptr + row * stride_m + offs, y, mask=mask)


def _dtype_constexpr(dtype: torch.dtype):
    return tl.float16 if dtype == torch.float16 else tl.bfloat16


def _num_warps(block_size: int) -> int:
    # Small rows (e.g. the per-head QK norms, H=128) run best with a single
    # warp; larger hidden dims get more warps. Swept empirically on sm_120.
    if block_size <= 128:
        return 1
    if block_size <= 256:
        return 2
    return 4


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight.

    Forward-only, CUDA, fp16/bf16, contiguous last dimension. One program per
    row; rows are flattened for >2D inputs (used by the per-head QK norms).
    """
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    M, H = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(H)
    num_warps = _num_warps(block_size)
    _rms_norm_kernel[(M,)](
        x,
        weight,
        y,
        M,
        H,
        eps,
        x.stride(0),
        DTYPE=_dtype_constexpr(x.dtype),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y.view(shape)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Add+RMSNorm: (normalized(x + residual), new residual).

    The returned residual is the fp32 sum cast back to the storage dtype,
    matching RMSNorm.add_rms_forward. Forward-only, CUDA, fp16/bf16.
    """
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    residual = residual.reshape(-1, shape[-1])
    M, H = x.shape
    y = torch.empty_like(x)
    new_residual = torch.empty_like(x)
    block_size = triton.next_power_of_2(H)
    num_warps = _num_warps(block_size)
    _add_rms_norm_kernel[(M,)](
        x,
        residual,
        weight,
        y,
        new_residual,
        M,
        H,
        eps,
        x.stride(0),
        DTYPE=_dtype_constexpr(x.dtype),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y.view(shape), new_residual.view(shape)


class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.eps)
        return add_rms_norm(x, residual, self.weight, self.eps)
