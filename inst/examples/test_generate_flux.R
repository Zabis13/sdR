library(sd2R)

cat("=== sd2R sd_generate() Flux Test ===\n\n")
#print(sd_system_info())

models_dir <- "/mnt/Data2/DS_projects/sd_models"

# Single context for all tests (vae_decode_only=FALSE needed for img2img/highres)
ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path = file.path(models_dir, "ae.safetensors"),
  clip_l_path = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads = 4L, model_type = "flux",
  vae_decode_only = FALSE, verbose = FALSE
)


# --- 1. Basic 768x768 (direct) ---
cat("\n--- 1. Basic 768x768 -> direct ---\n")
sd_profile_start()
t0 <- proc.time()
imgs <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting, foto, HDR",
  width = 768L, height = 768L,
  sample_steps = 10L, seed = 40L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)

elapsed <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs), imgs[[1]]$width, imgs[[1]]$height, elapsed))
sd_save_image(imgs[[1]], "/tmp/sd2R_flux_768.png")
cat("Saved: /tmp/sd2R_flux_768.png\n")
print(sd_profile_summary(sd_profile_get()))


cat("\n=== Done ===\n")
