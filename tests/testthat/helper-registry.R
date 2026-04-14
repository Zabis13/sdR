# Redirect the sd2R model registry into a per-session temporary directory
# so that running the test suite (including R CMD check) never writes to
# the user's home directory, as required by the CRAN Policy.
local({
  reg_dir <- file.path(tempdir(), "sd2R-registry")
  dir.create(reg_dir, recursive = TRUE, showWarnings = FALSE)
  Sys.setenv(SD2R_REGISTRY_DIR = reg_dir)
})
