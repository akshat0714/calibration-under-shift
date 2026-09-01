# Dataset provenance and release caveats

Raw archives are not redistributed by this repository. `scripts/download_data.sh` downloads the original public archives and verifies the publisher-recorded checksum before extraction. The committed SMIDS, HuSHeM, and Kromp audit grids, plus the SMIDS corruption grid, contain small selections or transformations of images from the CC BY 4.0 releases cited below; those image portions retain CC BY 4.0 and the citations below provide source attribution.

## SMIDS

- Source: Hidayet Takidin, Halil Ibrahim Ceylan, and Hakan Kusetogullari, *SMIDS: Sperm Morphology Image Data Set*, Mendeley Data v1, DOI [10.17632/6xvdhc9fyb.1](https://doi.org/10.17632/6xvdhc9fyb.1).
- License: CC BY 4.0, according to the Mendeley record.
- Expected composition: 3,000 images: normal 1,021; abnormal 1,005; non-sperm 974.
- Release caveat: 480 files use a `.bmp` suffix but contain PNG payloads, and the images span 1,914 dimensions. Preparation uses Pillow content sniffing rather than trusting extensions.

## HuSHeM

- Source: M. Shaker and S. A. Monadjemi, *Human Sperm Head Morphology dataset (HuSHeM)*, Mendeley Data v3, DOI [10.17632/tt3yj2pf38.3](https://doi.org/10.17632/tt3yj2pf38.3).
- License: CC BY 4.0, according to the Mendeley record.
- Expected composition: 216 RGB images: normal 54; tapered 53; pyriform 57; amorphous 52.
- Release caveat: six images differ from the nominal 131×131 dimensions. The data come from 15 patients, but the public release has no image-to-patient map. Reported five-fold CV is therefore image-level and may be optimistic if multiple images from one donor are correlated.

## Kromp blastocysts

- Source: Florian Kromp et al., *An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization*, Figshare v3, DOI [10.6084/m9.figshare.20123153.v3](https://doi.org/10.6084/m9.figshare.20123153.v3).
- License: CC BY 4.0 in the Figshare metadata. The archive itself does not contain a license file.
- Local audit: all 2,344 images decode as 512×384 RGB; 15 exact-byte duplicate groups are present, including groups that cross filename prefixes.
- Release caveat: the public archive does not provide patient IDs. Filename prefixes produce 851 groups, not the paper's 837 patients, so they are not silently substituted for patient identifiers here. The released silver-standard CSV also contains a conflicting duplicate annotation for `838_02.png`, while `846_01.png` is unlabeled. Kromp experiments are blocked until duplicates/labels are resolved and an author-verified image-to-patient map is supplied.

These are public research datasets, not clinically representative deployment cohorts. Their inclusion does not constitute clinical validation.
