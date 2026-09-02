# Dataset provenance

I do not redistribute raw archives. `scripts/download_data.sh` downloads each public release and verifies its publisher checksum before extraction. The committed audit figures and corruption example contain small selections or transformations from the CC BY 4.0 datasets cited below. Those image portions retain CC BY 4.0.

## SMIDS

- **Source.** [Takidin, Ceylan, and Kusetogullari](https://doi.org/10.17632/6xvdhc9fyb.1), Mendeley Data version 1
- **License.** CC BY 4.0 according to the Mendeley record
- **Composition.** 3,000 images with 1,021 normal, 1,005 abnormal, and 974 non-sperm images
- **Release issue.** 480 files use a `.bmp` suffix but contain PNG data. The images contain 1,914 distinct width-by-height pairs.

I decode files by content rather than extension. I use an image-stratified split because the release provides no patient or source-field identifier.

## HuSHeM

- **Source.** [Shaker and Monadjemi](https://doi.org/10.17632/tt3yj2pf38.3), Mendeley Data version 3
- **License.** CC BY 4.0 according to the Mendeley record
- **Composition.** 216 RGB images with 54 normal, 53 tapered, 57 pyriform, and 52 amorphous images
- **Release issue.** Six images differ from the nominal 131 by 131 dimensions. The release reports 15 patients but provides no image-to-patient map.

I use image-level five-fold cross-validation because patient grouping is unavailable. Correlation between images from the same patient could make this estimate optimistic.

## Kromp blastocysts

- **Source.** [Kromp et al.](https://doi.org/10.6084/m9.figshare.20123153.v3), Figshare version 3
- **License.** CC BY 4.0 according to the Figshare metadata. The archive contains no license file.
- **Composition.** 2,344 RGB images at 512 by 384 pixels
- **Release issues.** The audit found 15 exact-byte duplicate groups. The public files provide no patient identifiers. Filename prefixes produce 851 groups rather than the 837 patients reported in the paper. The silver annotation file contains conflicting labels for `838_02.png`. The image `846_01.png` has no released label.

I exclude Kromp from modeling until the duplicate and label issues are resolved and an author-verified image-to-patient map is available. I do not substitute filename prefixes for patient identifiers.

These public datasets are not representative clinical deployment cohorts. I do not treat their use as clinical validation.
