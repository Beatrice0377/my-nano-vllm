import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    y_ptr,
    M,
    H,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program handles BLOCK_M rows x BLOCK_H columns of the first half;
    # the second half (up) sits H columns to the right.
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    ptrs = x_ptr + offs_m[:, None] * stride_m + offs_h[None, :]
    gate = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(ptrs + H, mask=mask, other=0.0).to(tl.float32)
    # silu(gate) = gate * sigmoid(gate), computed in fp32 like F.silu.
    tl.store(
        y_ptr + offs_m[:, None] * H + offs_h[None, :],
        gate * tl.sigmoid(gate) * up,
        mask=mask,
    )


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused silu(gate) * up for a contiguous [M, 2*H] input (first half = gate).

    Forward-only, CUDA, fp16/bf16. Simple static launch configs by batch size,
    no autotuning.
    """
    M, D = x.shape
    H = D // 2
    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    if M >= 1024:  # prefill-like batches
        block_m, block_h, num_warps = 4, 256, 4
    elif M >= 32:  # medium batches
        block_m, block_h, num_warps = 4, 128, 4
    else:  # decode batches (few rows)
        block_m, block_h, num_warps = 1, 128, 2
    grid = (triton.cdiv(M, block_m), triton.cdiv(H, block_h))
    _silu_and_mul_kernel[grid](
        x, y, M, H, D, BLOCK_M=block_m, BLOCK_H=block_h, num_warps=num_warps
    )
    return y


class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class TritonSiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
