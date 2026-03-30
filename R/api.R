# REST API for sd2R via plumber
#
# Usage:
#   sd_api_start("model.safetensors", model_type = "flux")
#   sd_api_start(port = 8080)  # start without pre-loaded model

# ---------------------------------------------------------------------------
# Internal model store (package-level environment)
# ---------------------------------------------------------------------------
.api_env <- new.env(parent = emptyenv())
.api_env$models <- list()        # named list of sd_ctx objects
.api_env$upscalers <- list()     # named list of upscaler XPtr
.api_env$default_model <- NULL   # name of the default model
.api_env$api_key <- NULL         # NULL = no auth
.api_env$pr <- NULL              # plumber router instance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Encode sd_image list to base64 PNG strings
#' @param images List of sd_image objects
#' @return Character vector of base64-encoded PNG strings
#' @keywords internal
.images_to_base64 <- function(images) {
  if (!requireNamespace("png", quietly = TRUE)) {
    stop("Package 'png' is required for API. Install with install.packages('png')",
         call. = FALSE)
  }
  vapply(images, function(img) {
    arr <- sd_image_to_array(img)
    arr <- pmin(pmax(arr, 0), 1)
    raw_png <- png::writePNG(arr)
    base64enc::base64encode(raw_png)
  }, character(1))
}

#' Decode base64 PNG to sd_image
#' @param b64 Base64-encoded PNG string
#' @return sd_image list
#' @keywords internal
.base64_to_image <- function(b64) {
  if (!requireNamespace("png", quietly = TRUE)) {
    stop("Package 'png' is required", call. = FALSE)
  }
  raw_data <- base64enc::base64decode(b64)
  img <- png::readPNG(raw_data)
  if (length(dim(img)) == 2) {
    img <- array(rep(img, 3), dim = c(dim(img), 3L))
  }
  if (dim(img)[3] == 4L) img <- img[, , 1:3]
  h <- dim(img)[1]; w <- dim(img)[2]; ch <- dim(img)[3]
  interleaved <- aperm(img, c(3, 2, 1))
  bytes <- as.raw(as.integer(pmin(pmax(as.numeric(interleaved) * 255, 0), 255)))
  list(width = as.integer(w), height = as.integer(h),
       channel = as.integer(ch), data = bytes)
}

#' Recursively unbox scalar values in nested lists for JSON serialization
#' @param x List or atomic value
#' @return Same structure with scalars wrapped in \code{jsonlite::unbox}
#' @keywords internal
.unbox_scalars <- function(x, keep_arrays = character(0)) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) {
    nms <- names(x)
    result <- vector("list", length(x))
    for (i in seq_along(x)) {
      nm <- if (!is.null(nms)) nms[i] else ""
      if (nm %in% keep_arrays) {
        result[[i]] <- x[[i]]
      } else {
        result[[i]] <- .unbox_scalars(x[[i]], keep_arrays)
      }
    }
    names(result) <- nms
    return(result)
  }
  if (length(x) == 1L && !is.raw(x)) return(jsonlite::unbox(x))
  x
}

#' Build JSON error response
#' @keywords internal
.api_error <- function(res, status, message) {
  res$status <- status
  list(error = jsonlite::unbox(message))
}

#' Get model context by name (or default)
#' @keywords internal
.get_ctx <- function(model_id = NULL) {
  if (is.null(model_id) || model_id == "") {
    model_id <- .api_env$default_model
  }
  if (is.null(model_id) || !model_id %in% names(.api_env$models)) {
    return(NULL)
  }
  .api_env$models[[model_id]]
}

# ---------------------------------------------------------------------------
# Plumber router definition
# ---------------------------------------------------------------------------

#' Build plumber router with sd2R endpoints
#'
#' Creates and configures a plumber router. Called internally by
#' \code{\link{sd_api_start}}.
#'
#' @return A plumber router object
#' @keywords internal
.build_router <- function() {
  if (!requireNamespace("plumber", quietly = TRUE)) {
    stop("Package 'plumber' is required. Install with install.packages('plumber')",
         call. = FALSE)
  }
  if (!requireNamespace("base64enc", quietly = TRUE)) {
    stop("Package 'base64enc' is required. Install with install.packages('base64enc')",
         call. = FALSE)
  }

  pr <- plumber::pr()

  # --- Auth filter ---
  pr <- plumber::pr_filter(pr, "auth", function(req, res) {
    key <- .api_env$api_key
    if (!is.null(key)) {
      # Skip auth for localhost when no key header present
      remote <- req$REMOTE_ADDR %||% ""
      is_local <- remote %in% c("127.0.0.1", "::1", "")
      provided <- req$HTTP_X_API_KEY %||% req$HTTP_AUTHORIZATION %||% ""
      # Strip "Bearer " prefix if present
      provided <- sub("^Bearer\\s+", "", provided)
      if (!is_local || nchar(provided) > 0) {
        if (provided != key) {
          res$status <- 401L
          return(list(error = "Invalid or missing API key"))
        }
      }
    }
    plumber::forward()
  })

  # --- GET /health ---
  pr <- plumber::pr_get(pr, "/health", function(req, res) {
    models_info <- lapply(names(.api_env$models), function(nm) {
      ctx <- .api_env$models[[nm]]
      list(
        id = nm,
        model_type = attr(ctx, "model_type") %||% "unknown",
        vae_decode_only = attr(ctx, "vae_decode_only") %||% TRUE
      )
    })
    vram <- tryCatch({
      mem <- ggmlR::ggml_vulkan_device_memory(0L)
      list(free_mb = round(mem$free / 1024^2),
           total_mb = round(mem$total / 1024^2))
    }, error = function(e) list(free_mb = NA, total_mb = NA))

    .unbox_scalars(list(
      status = "ok",
      models_loaded = length(.api_env$models),
      default_model = .api_env$default_model,
      models = models_info,
      vram = vram
    ))
  })

  # --- GET /system ---
  pr <- plumber::pr_get(pr, "/system", function(req, res) {
    info <- sd_system_info()
    .unbox_scalars(list(
      sd2R_version = info$sd2R_version,
      sd_cpp_version = info$sd_cpp_version,
      system_info = info$system_info,
      num_cores = info$num_cores,
      vulkan_available = info$vulkan_available,
      vulkan_devices = sd_vulkan_device_count()
    ))
  })

  # --- POST /models/load ---
  pr <- plumber::pr_post(pr, "/models/load", function(req, res) {
    body <- req$body
    if (is.null(body$model_path) && is.null(body$diffusion_model_path)) {
      return(.api_error(res, 400L, "model_path or diffusion_model_path required"))
    }

    model_id <- body$model_id %||% basename(body$model_path %||% body$diffusion_model_path)

    # Build sd_ctx arguments from body
    ctx_args <- list()
    str_fields <- c("model_path", "vae_path", "clip_l_path", "clip_g_path",
                     "t5xxl_path", "diffusion_model_path", "control_net_path",
                     "taesd_path")
    for (f in str_fields) {
      if (!is.null(body[[f]])) ctx_args[[f]] <- body[[f]]
    }
    ctx_args$model_type <- body$model_type %||% "sd1"
    ctx_args$vae_decode_only <- if (!is.null(body$vae_decode_only)) {
      as.logical(body$vae_decode_only)
    } else TRUE
    if (!is.null(body$n_threads)) ctx_args$n_threads <- as.integer(body$n_threads)
    if (!is.null(body$device_layout)) ctx_args$device_layout <- body$device_layout
    if (!is.null(body$diffusion_gpu)) ctx_args$diffusion_gpu <- as.integer(body$diffusion_gpu)
    if (!is.null(body$clip_gpu)) ctx_args$clip_gpu <- as.integer(body$clip_gpu)
    if (!is.null(body$vae_gpu)) ctx_args$vae_gpu <- as.integer(body$vae_gpu)
    if (!is.null(body$keep_clip_on_cpu)) ctx_args$keep_clip_on_cpu <- as.logical(body$keep_clip_on_cpu)
    if (!is.null(body$keep_vae_on_cpu)) ctx_args$keep_vae_on_cpu <- as.logical(body$keep_vae_on_cpu)
    if (!is.null(body$vram_gb)) ctx_args$vram_gb <- as.numeric(body$vram_gb)
    ctx_args$verbose <- as.logical(body$verbose %||% FALSE)

    ctx <- tryCatch(do.call(sd_ctx, ctx_args), error = function(e) {
      return(.api_error(res, 500L, paste("Failed to load model:", conditionMessage(e))))
    })
    if (is.list(ctx) && !is.null(ctx$error)) return(ctx)

    .api_env$models[[model_id]] <- ctx
    if (is.null(.api_env$default_model)) .api_env$default_model <- model_id

    .unbox_scalars(list(status = "ok", model_id = model_id,
         model_type = attr(ctx, "model_type")))
  })

  # --- POST /models/unload ---
  pr <- plumber::pr_post(pr, "/models/unload", function(req, res) {
    model_id <- req$body$model_id
    if (is.null(model_id) || !model_id %in% names(.api_env$models)) {
      return(.api_error(res, 404L, "Model not found"))
    }
    .api_env$models[[model_id]] <- NULL
    if (identical(.api_env$default_model, model_id)) {
      remaining <- names(.api_env$models)
      .api_env$default_model <- if (length(remaining) > 0) remaining[1] else NULL
    }
    gc()
    .unbox_scalars(list(status = "ok", unloaded = model_id))
  })

  # --- GET /models ---
  pr <- plumber::pr_get(pr, "/models", function(req, res) {
    models_info <- lapply(names(.api_env$models), function(nm) {
      ctx <- .api_env$models[[nm]]
      list(
        id = nm,
        model_type = attr(ctx, "model_type") %||% "unknown",
        vae_decode_only = attr(ctx, "vae_decode_only") %||% TRUE,
        is_default = identical(nm, .api_env$default_model)
      )
    })
    .unbox_scalars(list(models = models_info, default = .api_env$default_model))
  })

  # --- POST /txt2img ---
  pr <- plumber::pr_post(pr, "/txt2img", function(req, res) {
    body <- req$body
    if (is.null(body$prompt)) {
      return(.api_error(res, 400L, "prompt is required"))
    }
    ctx <- .get_ctx(body$model_id)
    if (is.null(ctx)) {
      return(.api_error(res, 400L, "No model loaded. POST /models/load first."))
    }

    gen_args <- list(ctx = ctx, prompt = body$prompt)
    gen_args$negative_prompt <- body$negative_prompt %||% ""
    gen_args$width <- as.integer(body$width %||% 512L)
    gen_args$height <- as.integer(body$height %||% 512L)
    gen_args$sample_steps <- as.integer(body$sample_steps %||% 20L)
    gen_args$cfg_scale <- as.numeric(body$cfg_scale %||% 7.0)
    gen_args$seed <- as.integer(body$seed %||% 42L)
    gen_args$batch_count <- as.integer(body$batch_count %||% 1L)
    if (!is.null(body$sample_method)) gen_args$sample_method <- as.integer(body$sample_method)
    if (!is.null(body$scheduler)) gen_args$scheduler <- as.integer(body$scheduler)
    if (!is.null(body$clip_skip)) gen_args$clip_skip <- as.integer(body$clip_skip)
    if (!is.null(body$vae_mode)) gen_args$vae_mode <- body$vae_mode

    t0 <- proc.time()[3]
    images <- tryCatch(do.call(sd_txt2img, gen_args), error = function(e) {
      return(.api_error(res, 500L, paste("Generation failed:", conditionMessage(e))))
    })
    if (is.list(images) && !is.null(images$error)) return(images)
    elapsed <- round(proc.time()[3] - t0, 2)

    .unbox_scalars(list(
      images = as.list(.images_to_base64(images)),
      info = list(
        prompt = body$prompt,
        negative_prompt = gen_args$negative_prompt,
        width = gen_args$width,
        height = gen_args$height,
        steps = gen_args$sample_steps,
        cfg_scale = gen_args$cfg_scale,
        seed = gen_args$seed,
        model_id = body$model_id %||% .api_env$default_model,
        elapsed_s = elapsed
      )
    ))
  })

  # --- POST /img2img ---
  pr <- plumber::pr_post(pr, "/img2img", function(req, res) {
    body <- req$body
    if (is.null(body$prompt)) {
      return(.api_error(res, 400L, "prompt is required"))
    }
    if (is.null(body$init_image)) {
      return(.api_error(res, 400L, "init_image (base64 PNG) is required"))
    }
    ctx <- .get_ctx(body$model_id)
    if (is.null(ctx)) {
      return(.api_error(res, 400L, "No model loaded"))
    }

    init_img <- tryCatch(.base64_to_image(body$init_image), error = function(e) {
      return(.api_error(res, 400L, paste("Invalid init_image:", conditionMessage(e))))
    })
    if (is.list(init_img) && !is.null(init_img$error)) return(init_img)

    gen_args <- list(ctx = ctx, prompt = body$prompt, init_image = init_img)
    gen_args$negative_prompt <- body$negative_prompt %||% ""
    gen_args$strength <- as.numeric(body$strength %||% 0.75)
    gen_args$width <- as.integer(body$width %||% init_img$width)
    gen_args$height <- as.integer(body$height %||% init_img$height)
    gen_args$sample_steps <- as.integer(body$sample_steps %||% 20L)
    gen_args$cfg_scale <- as.numeric(body$cfg_scale %||% 7.0)
    gen_args$seed <- as.integer(body$seed %||% 42L)
    gen_args$batch_count <- as.integer(body$batch_count %||% 1L)
    if (!is.null(body$sample_method)) gen_args$sample_method <- as.integer(body$sample_method)
    if (!is.null(body$scheduler)) gen_args$scheduler <- as.integer(body$scheduler)
    if (!is.null(body$vae_mode)) gen_args$vae_mode <- body$vae_mode

    t0 <- proc.time()[3]
    images <- tryCatch(do.call(sd_img2img, gen_args), error = function(e) {
      return(.api_error(res, 500L, paste("Generation failed:", conditionMessage(e))))
    })
    if (is.list(images) && !is.null(images$error)) return(images)
    elapsed <- round(proc.time()[3] - t0, 2)

    .unbox_scalars(list(
      images = as.list(.images_to_base64(images)),
      info = list(
        prompt = body$prompt,
        width = gen_args$width,
        height = gen_args$height,
        strength = gen_args$strength,
        steps = gen_args$sample_steps,
        seed = gen_args$seed,
        model_id = body$model_id %||% .api_env$default_model,
        elapsed_s = elapsed
      )
    ))
  })

  # --- POST /generate ---
  pr <- plumber::pr_post(pr, "/generate", function(req, res) {
    body <- req$body
    if (is.null(body$prompt)) {
      return(.api_error(res, 400L, "prompt is required"))
    }
    ctx <- .get_ctx(body$model_id)
    if (is.null(ctx)) {
      return(.api_error(res, 400L, "No model loaded"))
    }

    gen_args <- list(ctx = ctx, prompt = body$prompt)
    gen_args$negative_prompt <- body$negative_prompt %||% ""
    gen_args$width <- as.integer(body$width %||% 512L)
    gen_args$height <- as.integer(body$height %||% 512L)
    gen_args$sample_steps <- as.integer(body$sample_steps %||% 20L)
    gen_args$cfg_scale <- as.numeric(body$cfg_scale %||% 7.0)
    gen_args$seed <- as.integer(body$seed %||% 42L)
    gen_args$batch_count <- as.integer(body$batch_count %||% 1L)
    if (!is.null(body$sample_method)) gen_args$sample_method <- as.integer(body$sample_method)
    if (!is.null(body$scheduler)) gen_args$scheduler <- as.integer(body$scheduler)
    if (!is.null(body$vae_mode)) gen_args$vae_mode <- body$vae_mode
    if (!is.null(body$init_image)) {
      gen_args$init_image <- tryCatch(.base64_to_image(body$init_image), error = function(e) NULL)
      if (!is.null(body$strength)) gen_args$strength <- as.numeric(body$strength)
    }

    t0 <- proc.time()[3]
    images <- tryCatch(do.call(sd_generate, gen_args), error = function(e) {
      return(.api_error(res, 500L, paste("Generation failed:", conditionMessage(e))))
    })
    if (is.list(images) && !is.null(images$error)) return(images)
    elapsed <- round(proc.time()[3] - t0, 2)

    .unbox_scalars(list(
      images = as.list(.images_to_base64(images)),
      info = list(
        prompt = body$prompt,
        width = gen_args$width,
        height = gen_args$height,
        steps = gen_args$sample_steps,
        seed = gen_args$seed,
        model_id = body$model_id %||% .api_env$default_model,
        elapsed_s = elapsed
      )
    ))
  })

  # --- POST /upscale ---
  pr <- plumber::pr_post(pr, "/upscale", function(req, res) {
    body <- req$body
    if (is.null(body$image)) {
      return(.api_error(res, 400L, "image (base64 PNG) is required"))
    }
    if (is.null(body$esrgan_path)) {
      return(.api_error(res, 400L, "esrgan_path is required"))
    }

    img <- tryCatch(.base64_to_image(body$image), error = function(e) {
      return(.api_error(res, 400L, paste("Invalid image:", conditionMessage(e))))
    })
    if (is.list(img) && !is.null(img$error)) return(img)

    factor <- as.integer(body$upscale_factor %||% 4L)

    t0 <- proc.time()[3]
    result <- tryCatch(
      sd_upscale_image(body$esrgan_path, img, upscale_factor = factor),
      error = function(e) {
        return(.api_error(res, 500L, paste("Upscale failed:", conditionMessage(e))))
      }
    )
    if (is.list(result) && !is.null(result$error)) return(result)
    elapsed <- round(proc.time()[3] - t0, 2)

    .unbox_scalars(list(
      images = as.list(.images_to_base64(list(result))),
      info = list(
        upscale_factor = factor,
        width = result$width,
        height = result$height,
        elapsed_s = elapsed
      )
    ))
  })

  # Set JSON serializer
  pr <- plumber::pr_set_serializer(pr, plumber::serializer_json(auto_unbox = TRUE))

  pr
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#' Start sd2R REST API server
#'
#' Launches a plumber-based REST API for image generation. Optionally pre-loads
#' a model at startup.
#'
#' @param model_path Optional path to model file to load at startup
#' @param model_type Model type for the pre-loaded model (default "sd1")
#' @param model_id Identifier for the pre-loaded model (default: basename of
#'   model_path)
#' @param vae_decode_only VAE decode only for the pre-loaded model (default TRUE)
#' @param host Host to bind to (default "0.0.0.0")
#' @param port Port to listen on (default 8080)
#' @param api_key Optional API key string. When set, non-localhost requests
#'   must include \code{X-API-Key} or \code{Authorization: Bearer <key>} header.
#'   Default \code{NULL} (no auth).
#' @param ... Additional arguments passed to \code{\link{sd_ctx}} for the
#'   pre-loaded model
#' @return Invisibly returns the plumber router object
#' @export
#' @examples
#' \dontrun{
#' # Start with a pre-loaded model
#' sd_api_start("model.safetensors", model_type = "flux", port = 8080)
#'
#' # Start empty, load models via API
#' sd_api_start(port = 8080)
#'
#' # With API key
#' sd_api_start("model.safetensors", api_key = "my-secret-key")
#' }
sd_api_start <- function(model_path = NULL,
                         model_type = "sd1",
                         model_id = NULL,
                         vae_decode_only = TRUE,
                         host = "0.0.0.0",
                         port = 8080L,
                         api_key = NULL,
                         ...) {
  .api_env$api_key <- api_key

  # Pre-load model if provided
  dots <- list(...)
  has_model <- !is.null(model_path) || !is.null(dots$diffusion_model_path)
  if (has_model) {
    if (is.null(model_id)) {
      model_id <- basename(model_path %||% dots$diffusion_model_path)
    }
    ctx <- sd_ctx(model_path = model_path, model_type = model_type,
                  vae_decode_only = vae_decode_only, ...)
    .api_env$models[[model_id]] <- ctx
    .api_env$default_model <- model_id
    message("Model loaded: ", model_id, " (", model_type, ")")
  }

  pr <- .build_router()
  .api_env$pr <- pr

  message("sd2R API starting on ", host, ":", port)
  if (!is.null(api_key)) message("API key authentication enabled")

  pr$run(host = host, port = as.integer(port), docs = TRUE)

  invisible(pr)
}

#' Stop sd2R REST API server
#'
#' Stops the running plumber server and unloads all models.
#'
#' @return No return value, called for side effects.
#' @export
sd_api_stop <- function() {
  if (!is.null(.api_env$pr)) {
    # plumber doesn't have a clean stop method for foreground servers,
    # but we clean up state
    .api_env$pr <- NULL
  }
  .api_env$models <- list()
  .api_env$upscalers <- list()
  .api_env$default_model <- NULL
  .api_env$api_key <- NULL
  gc()
  message("sd2R API stopped, all models unloaded")
  invisible(NULL)
}
