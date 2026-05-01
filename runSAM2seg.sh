#!bin/bash

# script to run organoid segmentation and measurement with SAM 2 for wi26
# vt 14 apr 2026

source activate sam2_env

# loop through folders
# ====== CONFIG ======
data_path="/home/vtaluja/hCOs/wi26"
output_path="/home/vtaluja/hCOs/wi26/analysis"
code_path="/home/vtaluja/hCOs/scripts/sam2_testing"
parent_folders="div10_0323 div11_0324"
# ====================

cd ${data_path}
for set in ${parent_folders}; do
    if [ -d "${set}" ]; then
        echo "Processing directory ${set} ... "
        
        # make output folder if missing
        outdir="${output_path}/masks_${set}"
        mkdir -p "${outdir}"
        
        python ${code_path}/v2_sam2_analysis.py \
          --input_dir "${data_path}/${set}" \
          --output_dir "${outdir}"
  fi
done
