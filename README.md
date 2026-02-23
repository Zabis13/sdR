# sdR

**sdR** is an R package that provides a native, GPU-accelerated Stable Diffusion pipeline by wrapping the C++ implementation from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and using [ggmlR](https://github.com/Zabis13/ggmlR) as the tensor backend.

## Overview

sdR exposes a high-level R interface for text-to-image and image-to-image generation, while all heavy computation (tokenization, encoders, denoiser, sampler, VAE, model loading) is implemented in C++. The package targets local inference on Linux with Vulkan-enabled AMD GPUs (with automatic CPU fallback via ggml), without relying on external Python or web APIs.

## Architecture

- **C++ core** (`src/sd/`): tokenizers, text encoders (CLIP, Mistral, Qwen, UMT5), diffusion UNet/MMDiT denoiser, samplers, VAE encoder/decoder, and model loading for `.safetensors` and `.gguf` weights.
- **R layer**: user-facing pipeline functions, parameter validation, image helpers, testing, and documentation-friendly API.
- **Backend**: links against ggmlR (headers via `LinkingTo`) and `libggml.a`, reusing the same GGML/Vulkan stack that also powers llamaR and other ggmlR-based packages.

## Key Features

- **Text-to-image** generation via `sd_txt2img()`, supporting Stable Diffusion 1.x models (e.g. SD 1.5) with typical 512x512 generations taking a few seconds on Vulkan-enabled GPUs.
- **Image-to-image** workflows via `sd_img2img()`, including noise strength control and reuse of the same denoising pipeline as text-to-image.
- **Optional upscaling** using a dedicated upscaler context managed entirely in C++ and exposed to R through external pointers.
- **Image utilities** in R: saving generated images to PNG, converting between internal tensors and R raw vectors, and simple inspection of output tensors.
- **System introspection** via `sd_system_info()`, reporting GGML/Vulkan capabilities as detected by ggmlR at build time.

## Implementation Details

- **Rcpp bindings**: `src/sdR_interface.cpp` defines a thin bridge between R and the C API in `stable-diffusion.h`, returning `XPtr` objects with custom finalizers for correct lifetime management of `sd_ctx_t` and `upscaler_ctx_t`.
- **Build system**: `src/Makevars` compiles all `sd/*.cpp` sources, links them with `libggml.a`, and includes `r_ggml_compat.h` to stay compatible with the installed ggmlR headers.
- **Package metadata**: `DESCRIPTION` declares Rcpp and ggmlR in `LinkingTo`, and `NAMESPACE` is generated via roxygen2 with `useDynLib` and Rcpp imports.
- **On load**: `.onLoad()` initializes logging and registers constant values that mirror the underlying C++ enums using 0-based indices.

## Planned CRAN Readiness

To meet CRAN size limits and policy requirements, sdR plans to:

- Download large tokenizer vocabularies (CLIP, Mistral, Qwen, UMT5) at install time or on first use instead of bundling them in the source tarball.
- Ship complete Rd documentation and vignettes for common workflows (txt2img, img2img, GPU configuration).
- Include a `SystemRequirements` field describing the optional Vulkan GPU backend and supported platforms.

## System Requirements

- R ≥ 4.1.0, C++17 compiler
- **Optional GPU**: `libvulkan-dev` + `glslc` (Linux) or Vulkan SDK (Windows)
- Platforms: Linux, macOS, Windows (x86-64, ARM64)

## See Also

- [llamaR](https://github.com/Zabis13/llamaR) — LLM inference in R
- [sdR](https://github.com/Zabis13/sdR) — Stable Diffusion in R
- [ggml](https://github.com/ggml-org/ggml) — underlying C library

## License

MIT


