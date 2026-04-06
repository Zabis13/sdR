#' Launch sd2R Shiny GUI
#'
#' Opens an interactive Shiny application for text-to-image generation.
#' Requires the \pkg{shiny} and \pkg{base64enc} packages.
#'
#' @param model_dir Path to folder with model files. If provided, the app
#'   scans the folder on startup and auto-assigns model roles.
#' @param launch.browser Open in browser (default TRUE)
#' @param port Port number (default NULL = random)
#' @param ... Additional arguments passed to \code{\link[shiny]{runApp}}
#' @return This function does not return; it runs the Shiny app until stopped.
#' @export
#' @examples
#' \dontrun{
#' sd_app()
#' sd_app(model_dir = "/path/to/models")
#' }
sd_app <- function(model_dir = NULL, launch.browser = TRUE, port = NULL, ...) {
  if (!requireNamespace("shiny", quietly = TRUE))
    stop("Package 'shiny' is required. Install with: install.packages('shiny')",
         call. = FALSE)
  if (!requireNamespace("base64enc", quietly = TRUE))
    stop("Package 'base64enc' is required. Install with: install.packages('base64enc')",
         call. = FALSE)
  if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("Package 'jsonlite' is required. Install with: install.packages('jsonlite')",
         call. = FALSE)
  app_dir <- system.file("shiny", "sd2R_app", package = "sd2R")
  if (!nzchar(app_dir))
    stop("Shiny app not found in sd2R installation", call. = FALSE)

  # Pass model_dir to the app via option
  old_opt <- getOption("sd2R.model_dir")
  options(sd2R.model_dir = if (!is.null(model_dir)) normalizePath(model_dir) else "")
  on.exit(options(sd2R.model_dir = old_opt), add = TRUE)

  shiny::runApp(app_dir, launch.browser = launch.browser, port = port, ...)
}
