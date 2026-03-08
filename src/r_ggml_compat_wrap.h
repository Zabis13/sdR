/*
 * Wrapper around ggmlR's r_ggml_compat.h for Windows builds.
 *
 * Problem: r_ggml_compat.h defines a macro `#define abort() ...` to intercept
 * abort() calls and route them through R's error handling. On Windows, the
 * system header msxml.h (pulled in transitively) uses abort() at global scope,
 * which breaks compilation because the macro expansion produces invalid syntax
 * at file scope.
 *
 * Solution: include r_ggml_compat.h normally (getting all R-safe redirections
 * for printf, exit, etc.), then immediately #undef abort. This is safe because
 * sd/ sources do not call abort() directly — they have been patched by
 * tools/patch_sd_sources.sh to use R-compatible alternatives.
 *
 * This is a temporary workaround until ggmlR adds a Windows-specific guard
 * around the abort() macro.
 */
#include "r_ggml_compat.h"
#undef abort
