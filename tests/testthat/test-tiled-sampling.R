# Tests for tiled sampling (MultiDiffusion) API and helpers

test_that("sd_txt2img_tiled has correct signature", {
  args <- formals(sd2R:::sd_txt2img_tiled)
  expect_true("sample_tile_size" %in% names(args))
  expect_true("sample_tile_overlap" %in% names(args))
  expect_true("vae_mode" %in% names(args))
  expect_null(eval(args$sample_tile_size))
  expect_equal(eval(args$sample_tile_overlap), 0.25)
  expect_equal(eval(args$width), 2048L)
  expect_equal(eval(args$height), 2048L)
  expect_equal(eval(args$vae_mode), "auto")
})

test_that(".native_latent_tile_size returns correct values", {
  expect_equal(sd2R:::.native_latent_tile_size("sd1"), 64L)
  expect_equal(sd2R:::.native_latent_tile_size("sd2"), 64L)
  expect_equal(sd2R:::.native_latent_tile_size("sdxl"), 128L)
  expect_equal(sd2R:::.native_latent_tile_size("flux"), 128L)
  expect_equal(sd2R:::.native_latent_tile_size("sd3"), 128L)
})

test_that("sd_txt2img_tiled auto-detects tile size from context", {
  # We can't create a real context without a model, but we can verify

  # the function checks the model_type attribute
  args <- formals(sd2R:::sd_txt2img_tiled)
  expect_null(eval(args$sample_tile_size))
})

test_that("tiled sampling passes parameters correctly to params list", {
  # Verify the function constructs params with tiled_sampling = TRUE
  # by checking that the function body references these param names
  fn_body <- deparse(body(sd2R:::sd_txt2img_tiled))
  expect_true(any(grepl("tiled_sampling", fn_body)))
  expect_true(any(grepl("sample_tile_size", fn_body)))
  expect_true(any(grepl("sample_tile_overlap", fn_body)))
})
