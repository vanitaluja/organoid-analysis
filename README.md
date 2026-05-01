# organoid-analysis
## Organoid Segmentation Pipeline
# created by vani, 2025

*work in progress*

Script to measure organoids from brightfield images. Best run in a bash shell for batches of images.

## contents

(1) runSAM2seg.sh - shell script to loop through images in a list of parent folders.
(2) v2_sam2_analysis.py - python script that does the segmentation. Should work without modification,
                            other than 3 lines which specify directories.
