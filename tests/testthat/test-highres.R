# Tests for high-res patch-based pipeline helpers

test_that(".native_tile_size returns correct values", {
  expect_equal(sd2R:::.native_tile_size("sd1"), 512L)
  expect_equal(sd2R:::.native_tile_size("sd2"), 512L)
  expect_equal(sd2R:::.native_tile_size("sdxl"), 1024L)
  expect_equal(sd2R:::.native_tile_size("flux"), 1024L)
  expect_equal(sd2R:::.native_tile_size("sd3"), 1024L)
})

test_that(".compute_patch_grid covers the full canvas", {
  grid <- sd2R:::.compute_patch_grid(2048, 2048, 512, 64)

  # Every pixel must be covered by at least one patch
  covered_x <- logical(2048)
  covered_y <- logical(2048)
  for (i in seq_len(nrow(grid))) {
    xs <- (grid$x[i] + 1L):(grid$x[i] + 512L)
    ys <- (grid$y[i] + 1L):(grid$y[i] + 512L)
    covered_x[xs] <- TRUE
    covered_y[ys] <- TRUE
  }
  expect_true(all(covered_x))
  expect_true(all(covered_y))
})

test_that(".compute_patch_grid single tile when image fits", {
  grid <- sd2R:::.compute_patch_grid(512, 512, 512, 64)
  expect_equal(nrow(grid), 1L)
  expect_equal(grid$x[1], 0L)
  expect_equal(grid$y[1], 0L)
})

test_that(".compute_patch_grid non-square image", {
  grid <- sd2R:::.compute_patch_grid(2048, 512, 512, 64)

  # Should have multiple columns but only 1 row
  expect_true(all(grid$y == 0L))
  expect_true(length(unique(grid$x)) > 1)

  # Full x coverage
  covered <- logical(2048)
  for (i in seq_len(nrow(grid))) {
    xs <- (grid$x[i] + 1L):(grid$x[i] + 512L)
    covered[xs] <- TRUE
  }
  expect_true(all(covered))
})

test_that(".compute_patch_grid last patch aligns to edge", {
  grid <- sd2R:::.compute_patch_grid(1000, 1000, 512, 64)

  # Last patch should end exactly at width/height
  max_x <- max(grid$x) + 512L
  max_y <- max(grid$y) + 512L
  expect_equal(max_x, 1000L)
  expect_equal(max_y, 1000L)
})

test_that(".blend_mask is all-ones for a corner patch", {
  mask <- sd2R:::.blend_mask(512, 512, 64,
                            is_left = TRUE, is_top = TRUE,
                            is_right = FALSE, is_bottom = FALSE)
  # Top-left corner: left and top edges are at canvas boundary → no ramp
  # Right and bottom have ramp
  expect_equal(mask[1, 1], 1)
  expect_equal(mask[1, 512], 0, tolerance = 0.02)  # right edge ramp
  expect_equal(mask[512, 1], 0, tolerance = 0.02)  # bottom edge ramp
})

test_that(".blend_mask center patch has ramps on all sides", {
  mask <- sd2R:::.blend_mask(512, 512, 64,
                            is_left = FALSE, is_top = FALSE,
                            is_right = FALSE, is_bottom = FALSE)
  # Center should be 1
  expect_equal(mask[256, 256], 1)
  # Edges should be < 1
  expect_true(mask[1, 256] < 1)    # top
  expect_true(mask[512, 256] < 1)  # bottom
  expect_true(mask[256, 1] < 1)    # left
  expect_true(mask[256, 512] < 1)  # right
})

test_that(".blend_mask edge patch suppresses ramp at boundary", {
  mask <- sd2R:::.blend_mask(512, 512, 64,
                            is_left = TRUE, is_top = TRUE,
                            is_right = TRUE, is_bottom = TRUE)
  # All edges at boundary → no ramps, mask is all ones
  expect_true(all(mask == 1))
})

test_that(".array_to_sd_image roundtrips with sd_image_to_array", {
  arr <- array(runif(16 * 16 * 3), dim = c(16, 16, 3))
  img <- sd2R:::.array_to_sd_image(arr)
  arr2 <- sd_image_to_array(img)
  # uint8 quantization: max error < 1/255 + epsilon
  expect_true(max(abs(arr2 - arr)) < 1/255 + 0.01)
})

test_that("sd_txt2img_highres has correct signature", {
  args <- formals(sd2R:::sd_txt2img_highres)
  expect_true("tile_size" %in% names(args))
  expect_true("overlap" %in% names(args))
  expect_true("img2img_strength" %in% names(args))
  expect_null(eval(args$tile_size))
  expect_equal(eval(args$overlap), 0.125)
  expect_null(eval(args$img2img_strength))
  expect_equal(eval(args$width), 2048L)
  expect_equal(eval(args$height), 2048L)
})

test_that("sd_ctx accepts model_type parameter", {
  args <- formals(sd_ctx)
  expect_true("model_type" %in% names(args))
  expect_equal(eval(args$model_type), "sd1")
})
