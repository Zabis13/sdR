#!/bin/sh
# Patch vendored sd/ sources for R package compatibility.
#
# C++ <cstdio> does '#undef printf', so the r_ggml_compat.h macro approach
# does not work for .cpp/.hpp files.  Instead we do direct text replacements.
#
# Usage:  ./tools/patch_sd_sources.sh [src_dir]
#   src_dir defaults to 'src/sd' (relative to package root).
#
# After updating upstream (git subtree pull), re-run this script.
#
# Uses perl instead of sed for cross-platform compatibility (macOS/Linux).

SD_DIR="${1:-src/sd}"

if [ ! -d "$SD_DIR" ]; then
  echo "ERROR: directory '$SD_DIR' not found" >&2
  exit 1
fi

echo "* Patching sd/ sources for R compatibility..."

# --- 1. Replace printf/puts/fflush/putchar with R-safe wrappers ---
#
# We only patch active (non-commented) calls.
#   - Skip lines starting with optional whitespace + '//'
#   - Replace whole-word printf( -> r_ggml_printf(
#   - Replace whole-word puts( -> r_ggml_puts(
#   - Replace whole-word putchar( -> r_ggml_putchar(
#   - Replace fflush(stdout) -> r_ggml_fflush(NULL)
#   - Replace fflush(stderr) -> r_ggml_fflush(NULL)
#   - Do NOT touch snprintf, sprintf, fprintf, vprintf, log_printf

# Process all .cpp, .hpp, .h, .c files (excluding thirdparty/)
find "$SD_DIR" -maxdepth 1 \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \) | while read -r f; do
  # Skip already-patched files (idempotent)
  if grep -q 'r_ggml_printf' "$f" 2>/dev/null; then
    continue
  fi

  perl -pi -e '
    next if /^\s*\/\//;
    s/\bprintf\s*\(/r_ggml_printf(/g;
    s/\bputs\s*\(/r_ggml_puts(/g;
    s/\bputchar\s*\(/r_ggml_putchar(/g;
    s/\bfflush\s*\(stdout\)/r_ggml_fflush(NULL)/g;
    s/\bfflush\s*\(stderr\)/r_ggml_fflush(NULL)/g;
  ' "$f"

  echo "  patched: $(basename "$f")"
done

# --- 2. Patch thirdparty/ separately (only specific files) ---
if [ -f "$SD_DIR/thirdparty/stb_image_resize.h" ]; then
  if ! grep -q 'r_ggml_printf' "$SD_DIR/thirdparty/stb_image_resize.h" 2>/dev/null; then
    perl -pi -e '
      next if /^\s*\/\//;
      s/\bprintf\s*\(/r_ggml_printf(/g;
    ' "$SD_DIR/thirdparty/stb_image_resize.h"
    echo "  patched: thirdparty/stb_image_resize.h"
  fi
fi

# --- 3. Fix compiler warnings (CRAN -Wall) ---

# 3a. util.cpp: unused variable 'size2'
perl -pi -e 's/int size2 = vsnprintf\(buf\.data\(\)/vsnprintf(buf.data()/' \
  "$SD_DIR/util.cpp" && echo "  patched: util.cpp (unused variable size2)"

# 3b. util.cpp: sign-compare warnings (int vs size_t)
perl -pi -e 's/for \(int p = start_position; p < res\.size\(\)/for (size_t p = (size_t)start_position; p < res.size()/' \
  "$SD_DIR/util.cpp"
perl -pi -e 's/while \(i \+ 1 < res\.size\(\)\)/while ((size_t)(i + 1) < res.size())/' \
  "$SD_DIR/util.cpp" && echo "  patched: util.cpp (sign-compare)"

# 3c. ggml_extend.hpp: sign-compare (int vs size_t for N)
perl -pi -e 's/for \(int i = 0; i < N; \+\+i\) \{/for (size_t i = 0; i < N; ++i) {/' \
  "$SD_DIR/ggml_extend.hpp" && echo "  patched: ggml_extend.hpp (sign-compare N)"

# 3d. ggml_extend.hpp: unused variable 'param' in get_param_tensors
perl -pi -e 's/^.*struct ggml_tensor\* param    = pair\.second;\n?//' \
  "$SD_DIR/ggml_extend.hpp" && echo "  patched: ggml_extend.hpp (unused variable param)"

# 3e. model.h: unused function sd_version_is_inpaint_or_unet_edit
perl -pi -e 's/^static bool sd_version_is_inpaint_or_unet_edit/[[maybe_unused]] static bool sd_version_is_inpaint_or_unet_edit/' \
  "$SD_DIR/model.h" && echo "  patched: model.h (unused function)"

# 3f. json.hpp.inc: deprecated literal operator with space (C++17)
#     operator "" _json  ->  operator ""_json
perl -pi -e 's/operator ""\s+_json\b/operator ""_json/g' \
  "$SD_DIR/thirdparty/json.hpp.inc" && echo "  patched: thirdparty/json.hpp.inc (literal operator spacing)"

# 3g. util.h: GNU extension ##__VA_ARGS__ -> C99/C++20 __VA_OPT__
perl -pi -e 's/, ##__VA_ARGS__/ __VA_OPT__(,) __VA_ARGS__/g' \
  "$SD_DIR/util.h" && echo "  patched: util.h (VA_OPT)"

# --- 4. Defensive mask creation in generate_image (img2img) ---
# stable-diffusion.cpp creates mask tensor at aligned width x height, but
# caller may provide mask_image with different (or zero) dimensions.
# Patch: create all-white mask if missing or size mismatch.
SDCPP="$SD_DIR/stable-diffusion.cpp"
if ! grep -q 'PATCH(sd2R): create all-white mask' "$SDCPP" 2>/dev/null; then
  perl -pi -e '
    if (/sd_image_to_ggml_tensor\(sd_img_gen_params->mask_image, mask_img\);/) {
      $_ = "        // PATCH(sd2R): create all-white mask if missing or size mismatch\n"
         . "        // The mask tensor uses the final aligned width x height, which may\n"
         . "        // differ from the caller-provided mask_image dimensions.\n"
         . "        // 255 = white = keep everything (correct default for img2img without inpainting).\n"
         . "        sd_image_t mask_image_used = sd_img_gen_params->mask_image;\n"
         . "        std::vector<uint8_t> default_mask;\n"
         . "        if (mask_image_used.data == nullptr ||\n"
         . "            (int)mask_image_used.width  != width ||\n"
         . "            (int)mask_image_used.height != height) {\n"
         . "            default_mask.assign((size_t)width * height, 255);\n"
         . "            mask_image_used.width   = width;\n"
         . "            mask_image_used.height  = height;\n"
         . "            mask_image_used.channel = 1;\n"
         . "            mask_image_used.data    = default_mask.data();\n"
         . "        }\n"
         . "        sd_image_to_ggml_tensor(mask_image_used, mask_img);\n"
         . "        // (original line replaced by PATCH above)\n";
    }
  ' "$SDCPP" && echo "  patched: stable-diffusion.cpp (defensive mask creation)"
fi

echo "* Done."
