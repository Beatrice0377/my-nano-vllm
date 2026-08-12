# Phase 1 environment

This is the reproducible validation target. GPU validation was performed in
the environment documented below (Linux/WSL + CUDA 12.8).

## Constraints

The imported `bb823b3e` `pyproject.toml` only specifies lower bounds:

```text
Python >=3.10,<3.13
torch >=2.4.0
triton >=3.0.0
transformers >=4.51.0
flash-attn
xxhash
```

Phase 1 also declares the baseline's direct `numpy`, `safetensors`, and
`tqdm` imports explicitly in `pyproject.toml`; the upstream metadata relied on
transitive installation for those packages.

The benchmark also imports `tqdm`. The current code path is Linux/CUDA-only:
`ModelRunner` initializes NCCL and CUDA, and FlashAttention's official install
requirements include a CUDA/ROCm toolkit, PyTorch 2.2+, and Linux for the CUDA
path. See the [FlashAttention installation notes](https://github.com/Dao-AILab/flash-attention#installation-and-features)
and [official PyTorch version commands](https://docs.pytorch.org/get-started/previous-versions/).

## Validated environment

This is the environment in which Phase 3/4 Triton kernels and the full
correctness suite were actually compiled and benchmarked:

| Component | Version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop (sm_120, 8 GB) |
| Python | 3.11.14 |
| PyTorch | 2.7.0+cu128 |
| CUDA Toolkit | 12.8 |
| Triton | **3.3.1** |
| FlashAttention | 2.7.4.post1 |
| Transformers | 4.51.3 |
| NumPy | 1.26.4 |
| safetensors | 0.5.3 |
| xxhash | 3.5.0 |
| tqdm | 4.67.1 |
| Build helpers | `packaging`, `psutil`, `ninja` |

### Triton 3.3.1

Triton 3.3.0 cannot compile the `tl.dot` kernels required by this project on
the RTX 5060's sm_120 architecture: its MMA-version selection asserts
`computeCapability not supported` for capability 120, so both the
PagedAttention and FlashAttention kernels fail at compile time. The
development and benchmark environment therefore upgrades to Triton 3.3.1,
whose `AccelerateMatmul` falls back to MMAv2 for `110 <= computeCapability <
130`. The working `.venv` applies this upgrade by replacing the Triton package
in-place (verified against PyTorch 2.7.0+cu128). No other packages were
rebuilt for this change. This is a known Triton issue
(`triton-lang/triton` #6087 / #6447); it is a sm_120-specific toolchain bug,
not a project code issue.

The first attempted target (`torch==2.6.0` with the CUDA 12.4 wheel) is not
valid for the RTX 5060 used for local validation: that wheel does not contain
an `sm_120` kernel image, and a one-line CUDA tensor smoke test fails with
`no kernel image is available`. PyTorch 2.7's CUDA 12.8 wheel is the first
target used here that includes Blackwell support. FlashAttention may build
from source, so the host must also expose a matching CUDA 12.8 toolkit
(`nvcc`) and compiler; the PyTorch wheel alone is not sufficient for that
build.

## Installation

```bash
cd /path/to/nano-vllm
uv venv --python 3.11 .venv
source .venv/bin/activate

# Run these in the WSL/GPU terminal before installing Python packages.
nvidia-smi
nvcc --version

uv pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install \
  "triton>=3.3.1,<3.4" \
  "transformers==4.51.3" \
  "numpy==1.26.4" \
  "safetensors==0.5.3" \
  "xxhash==3.5.0" \
  "tqdm==4.67.1" \
  packaging psutil ninja
MAX_JOBS=4 uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
uv pip install -e . --no-deps --no-build-isolation
```

Check the environment before loading a model:

```bash
python - <<'PY'
import torch
import triton
import transformers
import flash_attn
import xxhash
import tqdm

print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
print("xxhash", getattr(xxhash, "VERSION", "unknown"))
print("tqdm", tqdm.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name())
    torch.cuda.init()
else:
    raise SystemExit("CUDA is not available")
PY
```

## Model preparation

A Hugging Face cache root (e.g. `~/.cache/huggingface/hub/...` or a directory
containing only tokenizer links) is not a complete model directory. Prepare a
separate local snapshot:

```bash
MODEL_DIR=/path/to/models/Qwen3-0.6B
mkdir -p "$MODEL_DIR"
hf download Qwen/Qwen3-0.6B --local-dir "$MODEL_DIR"

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/tokenizer.json"
find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -print -quit
```

The benchmark accepts this `MODEL_DIR`, not the cache root.

## Phase 1 benchmark commands

Each successful command writes one JSON file and prints the same object:

```bash
python bench.py --workload balanced --model "$MODEL_DIR"
python bench.py --workload decode-heavy --model "$MODEL_DIR"
python bench.py --workload prefix-sharing --model "$MODEL_DIR"
```

Expected output paths are:

```text
benchmarks/results/baseline-balanced.json
benchmarks/results/baseline-decode-heavy.json
benchmarks/results/baseline-prefix-sharing.json
```

The benchmark writes no result file when the local WSL process cannot see CUDA
or the model snapshot is incomplete. That is a local execution diagnosis, not
a project-level Phase 2 decision.
