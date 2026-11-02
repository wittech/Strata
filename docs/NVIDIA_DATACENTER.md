# NVIDIA data center cards

A100/A800 use CUDA architecture `sm_80`; H100/H200 use `sm_90`. Strata reads the architecture from `nvidia-smi`, rather than a card-name allowlist:

```sh
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

NVIDIA lists H100/H200 at compute capability 9.0 and A100 at 8.0 in its [CUDA GPU table](https://developer.nvidia.com/cuda/gpus).
An [A800 deviceQuery report on NVIDIA's forum](https://forums.developer.nvidia.com/t/l2cache-size-of-a800-80gb/289934) reports 8.0.

## Why a launch can find no engine

The downloaded engine must contain code for the card. The older CUDA 13 main-engine package has `sm_75`, `sm_86`, `sm_89`,
`sm_120` and PTX 120. The older DeepSeek package has `sm_86`, `sm_89`, `sm_120` and PTX 120. Neither covers `sm_80` or `sm_90`.
`sm_86` machine code cannot run on `sm_80` or across the major-version boundary on `sm_90`. PTX 120 cannot run on either older architecture.
The CUDA 12 main-engine package already has `sm_80` and PTX 89 (which can be JIT-compiled on `sm_90`), but does not provide the DeepSeek engine.

This checkout adds `sm_80` and `sm_90` machine code to the x86_64 CUDA 13 Linux engine recipe, the Linux and Windows DeepSeek recipes,
the DeepSeek CMake default and the Windows picture reader. The source Dockerfile also adds `sm_90` (it already had `sm_80`).
Keep package metadata truthful: changing a manifest alone does not add executable GPU code. Previously published packages and feeds are unchanged.
The ARM64/DGX Spark package remains specific to `sm_121`; these recipe changes target x86_64 hosts.

## Build DeepSeek for strata launch

On the NVIDIA machine, install a CUDA toolkit, C++ compiler, CMake and Ninja and fetch the pinned llama.cpp checkout as described in
[source installation](SOURCE_INSTALL.md). From this Strata checkout, build for the detected GPU:

```sh
python3 -c 'import setup; setup.get_llama_cpp()'
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=third_party/llama.cpp \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-dsv41 --target strata-dsv41 -j 4
STRATA_DSV41_EXE="$PWD/build-dsv41/strata-dsv41" \
  strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

`--plan` does not serve or download the model. To install and serve, use the same executable setting and replace `--plan` with `--yes --no-tui`.
Keep your chosen model or existing `--model-path`; GPU compatibility does not mean the weights fit in RAM and VRAM.

For a build made on a different machine, replace `native` with `80` for A800/A100, `90` for H200/H100, or `"80-real;90"` for both.
On Windows, the executable is `build-dsv41\strata-dsv41.exe`; set `$env:STRATA_DSV41_EXE` to its absolute path in PowerShell before `strata launch`.

## Main engine and release packages

For a local Qwen source setup, use `./setup.sh --build --no-start` as described in [source installation](SOURCE_INSTALL.md).
Setup selects the local cards' architectures. It can download and prepare the chosen model.

For a Linux x86_64 CUDA 13 package, run `VARIANT=cuda13 packaging/build_cuda_linux.sh` on the matching CUDA toolchain.
It builds the main engine and reader for the same architecture list and records that list in the package. The default driver minimum remains 580.65.
The separate DeepSeek engine uses `packaging/build_dsv41_cuda.sh` or `packaging/build_dsv41_windows.bat`.
Windows main-engine packagers must also include `80-real` and `90-real` in their CMake architecture list and rebuild the matching reader.
Follow [engine packaging](../packaging/ENGINE-COMPONENT.md) to rebuild and sign release archives before publishing them in a feed.

## Validation status

The regression tests check recipe architecture coverage and the real command-line engine selector, including continued rejection of older incompatible
packages. These changes have not been compiled with CUDA or run on an A800 or H200 here. Before calling a build hardware-tested, run it on the target
card, check `--version`, load the intended model and test both prompt processing and generation; test pictures too if the package includes the reader.
No speed or output-parity claim is made for these cards.
