# Public data audit

I completed this audit on 2026-08-31 in the America/Los_Angeles time zone. I verified archive checksums before decoding images with Pillow. Machine-readable audit records are under `results/audits/` and sample figures are under `results/figures/`.

## SMIDS

- **Archive SHA-256.** `f46868e3a957414da55793973f75394ad2469fed48d3368e7f7f8a3aa59780a3`, matching the Mendeley version 1 manifest
- **Decoded images.** 3,000 RGB images with no decode failures and no exact-byte duplicates
- **Classes.** 1,021 normal, 1,005 abnormal, and 974 non-sperm images
- **Dimensions.** 1,914 distinct width and height pairs with widths from 122 to 259 pixels and heights from 80 to 264 pixels
- **Content issue.** 480 files use a `.bmp` suffix but have a PNG signature

| Split | Normal | Abnormal | Non-sperm | Total |
|---|---|---|---|---|
| Train | 715 | 703 | 682 | 2,100 |
| Validation | 102 | 100 | 98 | 300 |
| Calibration | 102 | 101 | 97 | 300 |
| Test | 102 | 101 | 97 | 300 |

I used an image-stratified split because the release provides no patient or source-field identifier. Possible correlation by source field remains a limitation.

![SMIDS sample](figures/smids_samples.png)

## HuSHeM

- **Archive SHA-256.** `aec7c19643a298386cae2399fb225b6b382a149b78d3a6d9239e842cce95de00`, matching the Mendeley version 3 manifest
- **Decoded images.** 216 RGB BMP images with no decode failures and no exact-byte duplicates
- **Classes.** 54 normal, 53 tapered, 57 pyriform, and 52 amorphous images
- **Dimensions.** 210 images are 131 by 131 pixels. Six images use one dimension from 118 to 127 pixels.
- **Folds.** Each outer fold contains 43 or 44 test images, 32 or 33 validation images, 22 calibration images, and 118 training images.

I used image-level cross-validation because the release reports 15 patients but provides no image-to-patient map. Correlation between images from the same patient could make the estimate optimistic.

![HuSHeM sample](figures/hushem_samples.png)

## Kromp blastocysts

- **Archive MD5.** `d19532b4b6bc4792b44738b8930d9ad2`, matching Figshare version 3
- **Decoded images.** 2,344 RGB PNG images at 512 by 384 pixels with no decode failures
- **Duplicates.** 15 exact-byte duplicate groups, including groups that cross filename prefixes
- **Annotations.** The silver file has 2,044 rows for 2,043 unique filenames. The gold file has 300 rows for 300 filenames with no overlap.

I did not create a Kromp split for the following reasons.

- The paper reports 837 patients but the release provides no patient identifier or map.
- Filename prefixes produce 851 groups and do not match the reported patient count.
- `838_02.png` appears twice in the silver annotations with conflicting expansion labels.
- `846_01.png` is present without a released label.
- The public gold set is image-level and crosses inferred filename-prefix groups.

`src.data.prepare` requires an author-verified patient map and raises an explicit error for conflicting labels. I do not report a Kromp model result.

![Kromp sample](figures/kromp_samples.png)
