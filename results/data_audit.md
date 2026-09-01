# Public-data audit

Audit date: 2026-08-31 (America/Los_Angeles). The machine-readable full results are in `results/audits/`; sample grids are in `results/figures/`. Images were decoded with Pillow after archive checksum verification.

## SMIDS — complete

- Archive SHA-256: `f46868e3a957414da55793973f75394ad2469fed48d3368e7f7f8a3aa59780a3` (matches the Mendeley v1 manifest).
- Decodable images: 3,000 RGB; corrupt: 0; exact byte duplicates: 0.
- Classes: normal 1,021; abnormal 1,005; non-sperm 974.
- Dimensions: 1,914 distinct width×height pairs; width 122–259 px; height 80–264 px.
- Content mismatch: 480 files have a `.bmp` suffix but a PNG signature. The loader decodes by content.

| Split | Normal | Abnormal | Non-sperm | Total |
|---|---:|---:|---:|---:|
| Train | 715 | 703 | 682 | 2,100 |
| Validation | 102 | 100 | 98 | 300 |
| Calibration | 102 | 101 | 97 | 300 |
| Test | 102 | 101 | 97 | 300 |

The v1 release supplies no patient or originating-field IDs. The split is therefore image-stratified, and possible source-field correlation is a limitation.

![SMIDS stratified sample grid](figures/smids_samples.png)

## HuSHeM — complete

- Archive SHA-256: `aec7c19643a298386cae2399fb225b6b382a149b78d3a6d9239e842cce95de00` (matches the Mendeley v3 manifest).
- Decodable images: 216 RGB BMP; corrupt: 0; exact byte duplicates: 0.
- Classes: normal 54; tapered 53; pyriform 57; amorphous 52. These exactly match the published counts.
- Dimensions: 210 images are 131×131. Six are 118×131, 131×118, 131×120, 131×123, 131×124, or 131×127.
- Five stratified outer folds contain 43–44 test, 32–33 validation, 22 calibration, and 118 train images each.

The record says the images originate from 15 patients but releases no image-to-patient map. Cross-validation is necessarily image-level; correlated donor images could make it optimistic.

![HuSHeM stratified sample grid](figures/hushem_samples.png)

## Kromp blastocysts — release audited, blocked before splitting

- Archive MD5: `d19532b4b6bc4792b44738b8930d9ad2` (matches Figshare v3).
- Decodable images: 2,344/2,344 RGB PNG, all 512×384; unreadable: 0.
- Fifteen exact-byte duplicate groups were found, including duplicates that cross filename prefixes; the full paths are in `results/audits/kromp.json`.
- Silver annotations: 2,044 rows / 2,043 unique filenames. Gold annotations: 300 rows / 300 filenames, with no silver/gold overlap.

The local archive and annotation schema were fully audited, but a defensible split cannot yet be made:

- The paper reports 2,344 images from 837 patients, but the release contains no patient identifier or mapping. Filename prefixes produce 851 groups and are not treated as patients.
- `Gardner_train_silver.csv` has 2,044 rows but 2,043 unique filenames. `838_02.png` appears twice with conflicting expansion labels (3 and 4).
- `846_01.png` is present but unlabeled, so released train and test annotations cover 2,343 of 2,344 images.
- The public gold set is image-level and crosses inferred filename-prefix groups.

`src.data.prepare` therefore requires an author-verified patient map and fails loudly on conflicting labels instead of producing a misleading “patient-level” manifest. No Kromp model result should be reported until those issues are resolved.

![Kromp release sample grid](figures/kromp_samples.png)
