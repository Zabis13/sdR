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

## Notes

Resubmission.
