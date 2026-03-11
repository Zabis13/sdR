# Tests for graph-based pipeline (R/graph.R)

# --- sd_node ---

test_that("sd_node creates valid node", {
  n <- sd_node("txt2img", prompt = "a cat", width = 512L)
  expect_s3_class(n, "sd_node")
  expect_equal(n$type, "txt2img")
  expect_equal(n$params$prompt, "a cat")
  expect_equal(n$params$width, 512L)
})

test_that("sd_node rejects unknown type", {
  expect_error(sd_node("unknown"), "Unknown node type")
})

test_that("sd_node accepts all valid types", {
  for (type in c("txt2img", "img2img", "upscale", "save")) {
    n <- sd_node(type)
    expect_equal(n$type, type)
  }
})

test_that("sd_node with no params has empty params list", {
  n <- sd_node("txt2img")
  expect_equal(length(n$params), 0L)
})

# --- sd_pipeline ---

test_that("sd_pipeline creates pipeline from nodes", {
  pipe <- sd_pipeline(
    sd_node("txt2img", prompt = "test"),
    sd_node("save", path = "out.png")
  )
  expect_s3_class(pipe, "sd_pipeline")
  expect_equal(length(pipe$nodes), 2L)
  expect_equal(pipe$nodes[[1]]$type, "txt2img")
  expect_equal(pipe$nodes[[2]]$type, "save")
})

test_that("sd_pipeline rejects non-node arguments", {
  expect_error(sd_pipeline("not a node"), "not an sd_node")
})

test_that("sd_pipeline rejects empty input", {
  expect_error(sd_pipeline(), "at least one node")
})

# --- print methods ---

test_that("print.sd_node works", {
  n <- sd_node("txt2img", prompt = "cat", width = 512L)
  out <- capture.output(print(n))
  expect_true(any(grepl("txt2img", out)))
  expect_true(any(grepl("prompt", out)))
})

test_that("print.sd_pipeline works", {
  pipe <- sd_pipeline(sd_node("txt2img"), sd_node("save", path = "x.png"))
  out <- capture.output(print(pipe))
  expect_true(any(grepl("2 node", out)))
})

# --- JSON serialization ---

test_that("sd_save_pipeline and sd_load_pipeline roundtrip", {
  pipe <- sd_pipeline(
    sd_node("txt2img", prompt = "a dog", width = 768L, height = 768L,
            cfg_scale = 7.5),
    sd_node("upscale", factor = 2L),
    sd_node("img2img", strength = 0.3),
    sd_node("save", path = "result.png")
  )

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))

  sd_save_pipeline(pipe, tmp)
  expect_true(file.exists(tmp))

  pipe2 <- sd_load_pipeline(tmp)
  expect_s3_class(pipe2, "sd_pipeline")
  expect_equal(length(pipe2$nodes), 4L)

  # Check types preserved
  expect_equal(pipe2$nodes[[1]]$type, "txt2img")
  expect_equal(pipe2$nodes[[2]]$type, "upscale")
  expect_equal(pipe2$nodes[[3]]$type, "img2img")
  expect_equal(pipe2$nodes[[4]]$type, "save")

  # Check params preserved
  expect_equal(pipe2$nodes[[1]]$params$prompt, "a dog")
  expect_equal(pipe2$nodes[[1]]$params$width, 768)
  expect_equal(pipe2$nodes[[1]]$params$height, 768)
  expect_equal(pipe2$nodes[[1]]$params$cfg_scale, 7.5)
  expect_equal(pipe2$nodes[[2]]$params$factor, 2)
  expect_equal(pipe2$nodes[[3]]$params$strength, 0.3)
  expect_equal(pipe2$nodes[[4]]$params$path, "result.png")
})

test_that("sd_save_pipeline rejects non-pipeline", {
  expect_error(sd_save_pipeline("not a pipeline", "x.json"),
               "Expected an sd_pipeline")
})

test_that("JSON file is human-readable", {
  pipe <- sd_pipeline(sd_node("txt2img", prompt = "test"))
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))

  sd_save_pipeline(pipe, tmp)
  json <- readLines(tmp)
  # Should contain type and prompt as readable strings
  expect_true(any(grepl("txt2img", json)))
  expect_true(any(grepl("test", json)))
})

# --- sd_run_pipeline validation ---

test_that("sd_run_pipeline rejects non-pipeline", {
  expect_error(sd_run_pipeline("not a pipeline", NULL),
               "Expected an sd_pipeline")
})

test_that("sd_run_pipeline img2img without input errors", {
  pipe <- sd_pipeline(sd_node("img2img", strength = 0.5))
  # Use a mock — we can't create a real ctx without a model,
  # but run_node_img2img checks for NULL image before calling sd_img2img
  expect_error(
    sd2R:::run_node_img2img(NULL, list(strength = 0.5), NULL, FALSE),
    "requires input image"
  )
})

test_that("sd_run_pipeline upscale without ctx errors", {
  expect_error(
    sd2R:::run_node_upscale(NULL, list(factor = 2), list()),
    "requires upscaler_ctx"
  )
})

test_that("sd_run_pipeline upscale without image errors", {
  expect_error(
    sd2R:::run_node_upscale("fake_ctx", list(factor = 2), NULL),
    "requires input image"
  )
})

test_that("sd_run_pipeline save without image errors", {
  expect_error(
    sd2R:::run_node_save(list(path = "x.png"), NULL),
    "requires input image"
  )
})

test_that("run_node_save requires path param", {
  img <- list(width = 4L, height = 4L, channel = 3L,
              data = as.raw(rep(0L, 48)))
  expect_error(
    sd2R:::run_node_save(list(), img),
    "requires 'path'"
  )
})
