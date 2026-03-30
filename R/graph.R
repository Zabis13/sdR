# Graph-based pipeline for sd2R
# Sequential node execution: output of each node feeds into the next.

#' Create a pipeline node
#'
#' @param type Node type: \code{"txt2img"}, \code{"img2img"}, \code{"upscale"},
#'   or \code{"save"}.
#' @param ... Parameters for the node (passed to the corresponding function).
#' @return A list with class \code{"sd_node"}.
#' @export
sd_node <- function(type, ...) {
  valid_types <- c("txt2img", "img2img", "upscale", "save")
  if (!type %in% valid_types) {
    stop("Unknown node type '", type, "'. Must be one of: ",
         paste(valid_types, collapse = ", "), call. = FALSE)
  }
  node <- list(type = type, params = list(...))
  class(node) <- "sd_node"
  node
}

#' Create a pipeline from nodes
#'
#' Nodes are executed sequentially. The image output of each node is passed
#' as input to the next node.
#'
#' @param ... \code{sd_node} objects in execution order.
#' @return A list with class \code{"sd_pipeline"}.
#' @export
sd_pipeline <- function(...) {
  nodes <- list(...)
  for (i in seq_along(nodes)) {
    if (!inherits(nodes[[i]], "sd_node")) {
      stop("Argument ", i, " is not an sd_node", call. = FALSE)
    }
  }
  if (length(nodes) == 0L) {
    stop("Pipeline must contain at least one node", call. = FALSE)
  }
  pipe <- list(nodes = nodes)
  class(pipe) <- "sd_pipeline"
  pipe
}

#' Run a pipeline
#'
#' Executes nodes sequentially. The first node must be \code{"txt2img"}
#' (produces an image from nothing). Subsequent nodes receive the previous
#' node's image output.
#'
#' @param pipeline An \code{sd_pipeline} object.
#' @param ctx A Stable Diffusion context created by \code{\link{sd_ctx}}.
#' @param upscaler_ctx Optional upscaler context created by
#'   \code{\link{sd_upscale_image}} setup. Required if the pipeline contains
#'   an \code{"upscale"} node. Pass the result of
#'   \code{sd_create_upscaler(path)}.
#' @param verbose Logical. Print progress messages. Default \code{FALSE}.
#' @return The final image (sd_image list), or the path string if the last
#'   node is \code{"save"}.
#' @export
sd_run_pipeline <- function(pipeline, ctx, upscaler_ctx = NULL,
                            verbose = FALSE) {
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }

  nodes <- pipeline$nodes
  image <- NULL

  for (i in seq_along(nodes)) {
    node <- nodes[[i]]
    type <- node$type
    params <- node$params

    if (verbose) message("[", i, "/", length(nodes), "] ", type)

    image <- switch(type,
      txt2img = run_node_txt2img(ctx, params, verbose),
      img2img = run_node_img2img(ctx, params, image, verbose),
      upscale = run_node_upscale(upscaler_ctx, params, image),
      save    = run_node_save(params, image),
      stop("Unknown node type: ", type, call. = FALSE)
    )
  }

  image
}

#' Save pipeline to JSON
#'
#' @param pipeline An \code{sd_pipeline} object.
#' @param path File path (should end in \code{.json}).
#' @return The file path, invisibly.
#' @export
sd_save_pipeline <- function(pipeline, path) {
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }
  nodes_list <- lapply(pipeline$nodes, function(n) {
    list(type = n$type, params = n$params)
  })
  json <- pipeline_to_json(nodes_list)
  writeLines(json, path)
  invisible(path)
}

#' Load pipeline from JSON
#'
#' @param path Path to a JSON file saved by \code{\link{sd_save_pipeline}}.
#' @return An \code{sd_pipeline} object.
#' @export
sd_load_pipeline <- function(path) {
  json <- paste(readLines(path, warn = FALSE), collapse = "\n")
  nodes_list <- pipeline_from_json(json)
  nodes <- lapply(nodes_list, function(n) {
    do.call(sd_node, c(list(type = n$type), n$params))
  })
  do.call(sd_pipeline, nodes)
}

# --- print methods ---

#' @export
print.sd_node <- function(x, ...) {
  cat("sd_node:", x$type, "\n")
  if (length(x$params) > 0L) {
    for (nm in names(x$params)) {
      val <- x$params[[nm]]
      if (is.character(val) && nchar(val) > 60L) {
        val <- paste0(substr(val, 1, 57), "...")
      }
      cat("  ", nm, "=", format(val), "\n")
    }
  }
  invisible(x)
}

#' @export
print.sd_pipeline <- function(x, ...) {
  cat("sd_pipeline:", length(x$nodes), "node(s)\n")
  for (i in seq_along(x$nodes)) {
    cat("  [", i, "] ", x$nodes[[i]]$type, "\n")
  }
  invisible(x)
}

# =========================================================================
# Internal: node runners
# =========================================================================

run_node_txt2img <- function(ctx, params, verbose) {
  args <- c(list(ctx = ctx, verbose = verbose), params)
  result <- do.call(sd_txt2img, args)
  # sd_txt2img returns a list of images (batch); take first
  if (is.list(result) && !is.null(result$width)) {
    result
  } else {
    result[[1L]]
  }
}

run_node_img2img <- function(ctx, params, image, verbose) {
  if (is.null(image)) {
    stop("img2img node requires input image from a previous node", call. = FALSE)
  }
  args <- c(list(ctx = ctx, image = image, verbose = verbose), params)
  result <- do.call(sd_img2img, args)
  if (is.list(result) && !is.null(result$width)) {
    result
  } else {
    result[[1L]]
  }
}

run_node_upscale <- function(upscaler_ctx, params, image) {
  if (is.null(upscaler_ctx)) {
    stop("upscale node requires upscaler_ctx argument in sd_run_pipeline()",
         call. = FALSE)
  }
  if (is.null(image)) {
    stop("upscale node requires input image from a previous node",
         call. = FALSE)
  }
  factor <- params$factor %||% 4L
  sd_upscale(upscaler_ctx, image, as.integer(factor))
}

run_node_save <- function(params, image) {
  if (is.null(image)) {
    stop("save node requires input image from a previous node", call. = FALSE)
  }
  path <- params$path
  if (is.null(path)) {
    stop("save node requires 'path' parameter", call. = FALSE)
  }
  sd_save_image(image, path)
  image
}

# =========================================================================
# Internal: minimal JSON serialization (no external dependencies)
# =========================================================================

pipeline_to_json <- function(nodes_list) {
  node_strings <- vapply(nodes_list, function(n) {
    params_str <- if (length(n$params) == 0L) {
      "{}"
    } else {
      pairs <- vapply(names(n$params), function(nm) {
        paste0("      ", json_quote(nm), ": ", json_value(n$params[[nm]]))
      }, character(1L))
      paste0("{\n", paste(pairs, collapse = ",\n"), "\n    }")
    }
    paste0("  {\n    \"type\": ", json_quote(n$type),
           ",\n    \"params\": ", params_str, "\n  }")
  }, character(1L))
  paste0("[\n", paste(node_strings, collapse = ",\n"), "\n]\n")
}

pipeline_from_json <- function(json) {
  # Minimal JSON array-of-objects parser for pipeline format.
  # Each element: {"type": "...", "params": {...}}
  # Supports string, numeric, logical, null values.
  parsed <- eval(parse(text = gsub("null", "NULL",
    gsub("false", "FALSE", gsub("true", "TRUE",
    gsub(":", "=", gsub("\\{", "list(", gsub("\\}", ")",
    gsub("\\[", "list(", gsub("\\]", ")", json))))))))))
  lapply(parsed, function(n) {
    list(type = n$type, params = as.list(n$params))
  })
}

json_quote <- function(x) {
  paste0("\"", gsub("\"", "\\\\\"", as.character(x)), "\"")
}

json_value <- function(x) {
  if (is.null(x)) return("null")
  if (is.logical(x)) return(tolower(as.character(x)))
  if (is.numeric(x)) return(format(x, scientific = FALSE))
  json_quote(x)
}

# base R %||% for older R versions
`%||%` <- function(a, b) if (is.null(a)) b else a
