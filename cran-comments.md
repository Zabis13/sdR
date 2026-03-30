## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local: Ubuntu 24.04, R 4.4.x

## Installed size

The package includes C++ source code for Stable Diffusion inference
using the ggml backend, which results in a larger compiled library size.
This is unavoidable due to the nature of the computational routines involved.

## Reviewer feedback addressed

* Added `\value` tag to `sd_save_image.Rd`.
* Exported `sd_txt2img_highres()` and `sd_txt2img_tiled()` (previously
  had examples but were not exported).
* Examples use `\dontrun{}` because they require large model files
  (multi-GB Stable Diffusion weights) that cannot be shipped with the
  package or downloaded in a test environment.
* Added all third-party copyright holders to Authors@R with 'cph' roles:
  Martin Raiber, Rich Geldreich, RAD Game Tools, Valve Software (miniz.h);
  Sean Barrett (stb_image.h);
  Jorge L Rodriguez (stb_image_resize.h);
  Alex Evans (PNG writing code in miniz.h);
  Niels Lohmann (json.hpp);
  Susumu Yata (darts.h/darts-clone); Kuba Podgorski (zip.h/zip.c);
  Meta Platforms Inc. (PyTorch-derived RNG code);
  Google Inc. (SentencePiece tokenizer code).
* rng_philox.hpp implements the standard Philox 4x32 algorithm
  (Salmon et al., 2011). Constants are from the original publication,
  not specific to any derived work. Attribution updated to reference
  the primary source.
* Removed single quotes around function names in DESCRIPTION.

## Notes

Resubmission.
