# sd2R 0.1.7

## Multi-GPU Model Parallelism
* New `device_layout` parameter in `sd_ctx()`: distribute sub-models across
  multiple Vulkan GPUs without separate processes.
  - `"mono"` — all on one GPU (default, backward-compatible).
  - `"split_encoders"` — CLIP/T5 on GPU 1, diffusion + VAE on GPU 0.
  - `"split_vae"` — CLIP/T5 + VAE on GPU 1, diffusion on GPU 0.
  - `"encoders_cpu"` — text encoders on CPU, diffusion + VAE on GPU.
* Low-level `diffusion_gpu`, `clip_gpu`, `vae_gpu` integer arguments for
  manual device assignment (override presets).

## Profiling
* New profiling API for per-stage timing of image generation:
  - `sd_profile_start()` / `sd_profile_stop()` — control event capture.
  - `sd_profile_get()` — raw event data frame.
  - `sd_profile_summary()` — formatted summary with durations and percentages.
* Stages tracked: `text_encode` (with `text_encode_clip` and `text_encode_t5`
  sub-stages), `sampling`, `vae_decode`, `vae_encode`, model loading.
* Pretty-printed output via `print.sd_profile()`.

---

# sd2R 0.1.6

## Pipeline Graph API
* New `sd_pipeline()` / `sd_node()` — sequential graph-based pipeline.
  Node types: `"txt2img"`, `"img2img"`, `"upscale"`, `"save"`.
* `sd_run_pipeline(pipeline, ctx)` — execute pipeline with a single context.
* `sd_save_pipeline()` / `sd_load_pipeline()` — JSON serialization.

---

# sd2R 0.1.5

## Flux Support
* Flux model family (flux1-dev, etc.) fully supported: text-to-image,
  image-to-image, highres fix, tiled sampling, multi-GPU.
* Separate model paths: `diffusion_model_path`, `vae_path`, `clip_l_path`,
  `t5xxl_path` in `sd_ctx()`.
* `cfg_scale` auto-defaults to 1.0 for Flux (guidance-distilled models).

## img2img Improvements
* `sd_generate()` now defaults `width`/`height` to init image dimensions
  when not specified explicitly.

---

# sd2R 0.1.4

## Build System
* `configure.win` rewritten to use template approach (`Makevars.win.in` →
  `Makevars.win`), matching `ggmlR` pattern.

---

# sd2R 0.1.3

## Unified `sd_generate()` Entry Point
* New `sd_generate()` — single function for all generation modes. Automatically
  selects the optimal strategy (direct, tiled sampling, or highres fix) based
  on output resolution and available VRAM.
* `vram_gb` parameter in `sd_ctx()`: set once, auto-routing handles the rest.

## Multi-GPU
* New `sd_generate_multi_gpu()` — parallel generation across multiple Vulkan
  GPUs via `callr`, one process per GPU, with progress reporting.

## Performance
* Batch compute optimization for tiled sampling: pre-allocated compute context
  buffer eliminates ~110 MB malloc/free per UNet call.

---

# sd2R 0.1.2

## Highres Fix
* New `sd_highres_fix()` — classic two-pass highres pipeline:
  txt2img at native resolution → upscale → tiled img2img refinement.
* `hr_strength` parameter (default 0.4) controls refinement intensity.

## Tiled img2img
* New `sd_img2img_tiled()` — img2img with MultiDiffusion tiled sampling for
  large images.

---

# sd2R 0.1.1

## VAE Tiling
* New `vae_mode` parameter: `"normal"`, `"tiled"`, `"auto"` (default).
  Auto-tiles when image area exceeds threshold.
* `vae_tile_rel_x` / `vae_tile_rel_y` for adaptive tile sizing.

## High-Resolution Pipeline
* New `sd_txt2img_highres()` — patch-based generation for 2K, 4K+ images.
* `model_type` parameter in `sd_ctx()`: `"sd1"`, `"sd2"`, `"sdxl"`, `"flux"`,
  `"sd3"`.

## Tiled Sampling (MultiDiffusion)
* New `sd_txt2img_tiled()` — tiled diffusion sampling at any resolution.
  VRAM bounded by tile size, not output resolution.

---

# sd2R 0.1.0

## Core
* Text-to-image generation via stable-diffusion.cpp (C++ backend).
* Support for SD 1.x, SD 2.x, SDXL model versions.
* SafeTensors and GGUF model format loading.
* Vulkan GPU backend via ggmlR.
* Samplers: Euler, Euler A, Heun, DPM2, DPM++ (2M), LCM, DDIM, TCD.
* Schedulers: Discrete, Karras, Exponential, Simple, SGM Uniform, AYS, LCM.

## R API
* `sd_ctx()` — create model context.
* `sd_generate()` — unified entry point.
* `sd_txt2img()`, `sd_img2img()` — low-level generation.
* `sd_save_image()`, `sd_system_info()`.
