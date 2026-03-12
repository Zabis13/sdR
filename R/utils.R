# Utility functions for sd2R

#' Get system information
#'
#' Returns information about the stable-diffusion.cpp backend.
#'
#' @return List with system info, version, and core count
#' @export
sd_system_info <- function() {
  info <- list(
    sd2R_version = as.character(utils::packageVersion("sd2R")),
    sd_cpp_version = sd_version_cpp(),
    system_info = sd_system_info_cpp(),
    num_cores = sd_num_physical_cores_cpp(),
    vulkan_available = ggmlR::ggml_vulkan_available()
  )
  class(info) <- "sd_system_info"
  info
}

#' Get number of Vulkan GPU devices
#'
#' Returns the number of Vulkan-capable GPU devices available on the system.
#' Useful for deciding whether to use \code{\link{sd_generate_multi_gpu}}.
#'
#' @return Integer, number of Vulkan devices (0 if Vulkan is not available)
#' @export
sd_vulkan_device_count <- function() {
  tryCatch(ggmlR::ggml_vulkan_device_count(), error = function(e) 0L)
}

#' @export
print.sd_system_info <- function(x, ...) {
  cat("sd2R System Information\n")
  cat("  sd2R version:   ", x$sd2R_version, "\n")
  cat("  sd.cpp version: ", x$sd_cpp_version, "\n")
  cat("  Physical cores: ", x$num_cores, "\n")
  cat("  Backend info:   ", x$system_info, "\n")
  cat("  Vulkan GPU:     ", if (x$vulkan_available) "available" else "not available", "\n")
  invisible(x)
}

#' Start profiling
#'
#' Clears the event buffer and begins capturing stage timings from sd.cpp.
#'
#' @export
#' @name sd_profile_start
NULL

#' Stop profiling
#'
#' Stops capturing stage events. Call \code{\link{sd_profile_get}} to retrieve.
#'
#' @export
#' @name sd_profile_stop
NULL

#' Get raw profile events
#'
#' Returns a data frame of captured events with columns \code{stage},
#' \code{kind} (\code{"start"}/\code{"end"}), and \code{timestamp_ms}.
#'
#' @return Data frame of profile events.
#' @export
#' @name sd_profile_get
NULL

#' Build a profile summary from raw events
#'
#' Matches start/end events by stage and computes durations.
#'
#' @param events Data frame from \code{sd_profile_get()} with columns
#'   \code{stage}, \code{kind}, \code{timestamp_ms}.
#' @return Data frame with columns \code{stage}, \code{start_ms},
#'   \code{end_ms}, \code{duration_ms}, \code{duration_s}.
#'   Has class \code{"sd_profile"} for pretty printing.
#' @export
sd_profile_summary <- function(events) {
  if (nrow(events) == 0L) return(events)
  starts <- events[events$kind == "start", , drop = FALSE]
  ends   <- events[events$kind == "end",   , drop = FALSE]

  # Infer end times for load_* stages: each load ends when the next load starts,

  # or at load_all end
  load_starts <- events[grepl("^load_", events$stage) & events$kind == "start", ,
                         drop = FALSE]
  load_all_end <- events[events$stage == "load_all" & events$kind == "end", ,
                          drop = FALSE]
  if (nrow(load_starts) > 1L || (nrow(load_starts) == 1L && nrow(load_all_end) > 0L)) {
    load_starts <- load_starts[order(load_starts$timestamp_ms), , drop = FALSE]
    for (i in seq_len(nrow(load_starts))) {
      end_ts <- if (i < nrow(load_starts)) {
        load_starts$timestamp_ms[i + 1L]
      } else if (nrow(load_all_end) > 0L) {
        load_all_end$timestamp_ms[1L]
      } else NA_real_
      if (!is.na(end_ts)) {
        events <- rbind(events, data.frame(
          stage = load_starts$stage[i], kind = "end",
          timestamp_ms = end_ts, stringsAsFactors = FALSE
        ))
      }
    }
  }

  starts <- events[events$kind == "start", , drop = FALSE]
  ends   <- events[events$kind == "end",   , drop = FALSE]

  stages <- unique(c(starts$stage, ends$stage))
  rows <- list()
  for (s in stages) {
    if (s == "load_all") next
    s_starts <- starts$timestamp_ms[starts$stage == s]
    s_ends   <- ends$timestamp_ms[ends$stage == s]
    n <- min(length(s_starts), length(s_ends))
    if (n > 0L) {
      for (i in seq_len(n)) {
        dur <- s_ends[i] - s_starts[i]
        rows[[length(rows) + 1L]] <- data.frame(
          stage = s,
          start_ms = s_starts[i],
          end_ms = s_ends[i],
          duration_ms = dur,
          duration_s = round(dur / 1000, 2),
          stringsAsFactors = FALSE
        )
      }
    } else if (length(s_ends) > 0L) {
      for (i in seq_along(s_ends)) {
        rows[[length(rows) + 1L]] <- data.frame(
          stage = s,
          start_ms = NA_real_,
          end_ms = s_ends[i],
          duration_ms = NA_real_,
          duration_s = NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  result <- do.call(rbind, rows)
  result <- result[order(result$end_ms), , drop = FALSE]
  rownames(result) <- NULL
  class(result) <- c("sd_profile", "data.frame")
  result
}

#' @export
print.sd_profile <- function(x, ...) {
  if (nrow(x) == 0L) {
    cat("(no profile events)\n")
    return(invisible(x))
  }
  total_row <- x[x$stage == "generate_total", , drop = FALSE]
  total_s <- if (nrow(total_row) > 0L) total_row$duration_s[1L] else NA_real_

  .pct <- function(dur) {
    if (!is.na(total_s) && !is.na(dur) && total_s > 0) {
      sprintf(" (%4.1f%%)", dur / total_s * 100)
    } else ""
  }

  .line <- function(label, dur, indent = 2L) {
    pad <- paste(rep(" ", indent), collapse = "")
    cat(sprintf("%s%-20s %7.2fs%s\n", pad, label, dur, .pct(dur)))
  }

  cat("sd2R Profile\n")

  # Load stages (if present)
  load_stages <- x[grepl("^load_", x$stage) & !is.na(x$duration_s), , drop = FALSE]
  if (nrow(load_stages) > 0L) {
    for (i in seq_len(nrow(load_stages))) {
      .line(load_stages$stage[i], load_stages$duration_s[i])
    }
  }

  # Text encoding total + sub-stages
  te_total <- x[x$stage == "text_encode" & !is.na(x$duration_s), , drop = FALSE]
  te_clip  <- x[x$stage == "text_encode_clip" & !is.na(x$duration_s), , drop = FALSE]
  te_t5    <- x[x$stage == "text_encode_t5" & !is.na(x$duration_s), , drop = FALSE]
  if (nrow(te_total) > 0L) {
    .line("text_encode", te_total$duration_s[1L])
    if (nrow(te_clip) > 0L) .line("clip", te_clip$duration_s[1L], 4L)
    if (nrow(te_t5) > 0L)   .line("t5", te_t5$duration_s[1L], 4L)
  }

  # VAE encode (img2img)
  ve <- x[x$stage == "vae_encode" & !is.na(x$duration_s), , drop = FALSE]
  if (nrow(ve) > 0L) .line("vae_encode", ve$duration_s[1L])

  # Sampling
  samp <- x[x$stage == "sampling" & !is.na(x$duration_s), , drop = FALSE]
  if (nrow(samp) > 0L) .line("sampling", samp$duration_s[1L])

  # VAE decode
  vd <- x[x$stage == "vae_decode" & !is.na(x$duration_s), , drop = FALSE]
  if (nrow(vd) > 0L) .line("vae_decode", vd$duration_s[1L])

  # Total
  if (!is.na(total_s)) {
    cat(sprintf("  %-20s %7.2fs\n", "TOTAL", total_s))
  }
  invisible(x)
}
