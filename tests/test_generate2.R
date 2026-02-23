library(sdR)

cat("=== sdR Generation Test ===\n\n")
print(sd_system_info())

# --- SD 1.5 ---
cat("\n--- Loading SD 1.5 ---\n")
ctx <- sd_ctx(
  "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors",
  n_threads = 4L
)
cat("Context created\n")

cat("\n--- Generating 1024x1024, 35 steps, 20 images ---\n")

n_imgs <- 20L
seeds  <- 42L + seq_len(n_imgs) - 1L  # разные сиды

imgs <- vector("list", n_imgs)

for (i in seq_len(n_imgs)) {
  cat("  -> image", i, "seed", seeds[i], "\n")
  x <- sd_txt2img(
    ctx,
    prompt          = "a cat sitting on a chair, highly detailed, 4k, best quality",
    negative_prompt = "blurry, bad quality, lowres, artifacts",
    width           = 768L,
    height          = 768L,
    sample_steps    = 40L,
    cfg_scale       = 7.0,
    seed            = seeds[i],
    sample_method   = SAMPLE_METHOD$EULER,
    scheduler       = SCHEDULER$DISCRETE
  )
  imgs[[i]] <- x[[1]]
  fn <- sprintf("/tmp/sdR_sd15_cat_%02d.png", i)
  sd_save_image(imgs[[i]], fn)
  cat("     saved:", fn, "\n")
}

cat("Generated", length(imgs), "image(s)\n")
cat("Size of first:", imgs[[1]]$width, "x", imgs[[1]]$height, "x", imgs[[1]]$channel, "\n")

# Cleanup
rm(ctx)
gc()

cat("\n=== Done ===\n")

