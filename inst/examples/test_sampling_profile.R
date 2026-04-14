#!/usr/bin/env Rscript
# Sampling bottleneck profiler — Flux.1 768x768, 1 step, per-op Vulkan timings
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_sampling_profile.R

Sys.setenv(GGML_VK_PERF_LOGGER = "1")

library(sd2R)

models_dir <- "/mnt/Data2/DS_projects/sd_models"

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  tensor_type_rules    = "first_stage_model=f16",
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = TRUE,
  verbose              = FALSE
)

# --- 1 step: isolate single denoising step cost ---
cat("\n=== 1 step (isolate per-step cost) ===\n")
sd_profile_start()
t0 <- proc.time()

imgs <- sd_generate(
  ctx,
  prompt        = "a red apple on a white table",
  width         = 768L, height = 768L,
  sample_steps  = 1L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)

elapsed_1 <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs\n", elapsed_1))
print(sd_profile_summary(sd_profile_get()))

# --- 4 steps: check per-step overhead ---
cat("\n=== 4 steps ===\n")
sd_profile_start()
t0 <- proc.time()

imgs <- sd_generate(
  ctx,
  prompt        = "a red apple on a white table",
  width         = 768L, height = 768L,
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)

elapsed_4 <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs\n", elapsed_4))
cat(sprintf("per step:  %.2fs\n", elapsed_4 / 4))
print(sd_profile_summary(sd_profile_get()))

cat(sprintf("\n=== Summary ===\n"))
cat(sprintf("1 step wall:  %.2fs\n", elapsed_1))
cat(sprintf("4 step wall:  %.2fs\n", elapsed_4))
cat(sprintf("step overhead (graph build etc): ~%.2fs\n", elapsed_1 - elapsed_4 / 4))
