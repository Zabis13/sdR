# sd2R REST API example
#
# Start server:
#   Rscript inst/examples/test_api.R
#
# Test with curl:
#   curl http://localhost:8080/health
#   curl -X POST http://localhost:8080/txt2img \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "a cat in space", "width": 512, "height": 512, "sample_steps": 10}'
#
# With API key:
#   export SD2R_API_KEY=my-secret
#   Rscript inst/examples/test_api.R
#   curl -H "X-API-Key: my-secret" http://localhost:8080/health

library(sd2R)

# --- Configuration ---
model_path <- Sys.getenv("SD2R_MODEL", "model.safetensors")
model_type <- Sys.getenv("SD2R_MODEL_TYPE", "sd1")
port <- as.integer(Sys.getenv("SD2R_PORT", "8080"))
api_key <- Sys.getenv("SD2R_API_KEY", unset = NA)
if (is.na(api_key)) api_key <- NULL

cat("=== sd2R API Server ===\n")
cat("Model:", model_path, "\n")
cat("Type:", model_type, "\n")
cat("Port:", port, "\n")
cat("Auth:", if (is.null(api_key)) "disabled" else "enabled", "\n\n")

# --- Start ---
sd_api_start(
  model_path = model_path,
  model_type = model_type,
  port = port,
  api_key = api_key,
  vae_decode_only = FALSE
)
