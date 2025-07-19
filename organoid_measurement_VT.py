
# ********************
# script to measure organoids ~ work in progress
# uses trained pixel classification & object segmentation from ilastik
# additional adaptations from OrgM FIJI macro for use in pure python
# created by vani taluja
# last modified 19 july 2025

# imagej==0.3.2
# numpy==2.0.2
# pandas==2.3.0
# pyimagej==1.7.0
# scikit-image==0.24.0
# scipy==1.13.1
# scyjava==1.12.0
# ********************

import os, sys, math, datetime
import numpy as np
import argparse
import subprocess
from skimage.io import imread
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pathlib import Path
import h5py
import tifffile
import shutil
import scyjava
import csv
import gc

# initialize ImageJ
scyjava.config.jvm_path = "/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home/bin/java" 

import imagej
ij = imagej.init('/Applications/Fiji.app', mode='headless', add_legacy=False)

# Java class imports for ImageJ
IJ = scyjava.jimport('ij.IJ')
Measurements = scyjava.jimport('ij.measure.Measurements')
ParticleAnalyzer = scyjava.jimport('ij.plugin.filter.ParticleAnalyzer')
ResultsTable = scyjava.jimport('ij.measure.ResultsTable')
ImageProcessor = scyjava.jimport('ij.process.ImageProcessor')

# argument parsing
parser = argparse.ArgumentParser(description='Organoid Size Measurement with Ilastik segmentation')
parser.add_argument('--input_dir', required=True, help='Directory of input brightfield images')
parser.add_argument('--output_dir', required=True, help='Directory to save output results')
parser.add_argument('--ilastik_project', required=True, help='Path to Ilastik .ilp project file')
parser.add_argument('--object_project', required=True, help='Path to Ilastik object classification .ilp file')
parser.add_argument('--alt_object_project', help='Path to low threshold Ilastik object classification .ilp file')
parser.add_argument('--ilastik_exec', default='/Applications/ilastik-1.4.1.post1-arm64-OSX.app/Contents/MacOS/ilastik', help='Command or path to Ilastik executable')
parser.add_argument('--pix_width', type=float, default=1.887, help='Pixel width in microns')
parser.add_argument('--pix_height', type=float, default=1.894, help='Pixel height in microns')
parser.add_argument('--area_threshold', type=float, default=50000, help='Minimum area threshold to consider an organoid')
parser.add_argument('--feret_threshold', type=float, default=150, help='Minimum feret threshold to consider an organoid')
parser.add_argument('--round_threshold', type=float, default=0.62, help='Minimum roundness threshold to consider an organoid')
parser.add_argument('--min_size', type=float, default=1000, help='Minimum size to be considered an ROI')
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
mask_dir = output_dir / "ilastik_masks"
prob_output_dir = output_dir / "prob_maps"

output_dir.mkdir(parents=True, exist_ok=True)
prob_output_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

output_csv_path = output_dir / f"organoid_measurements_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"

# define the column names based on output_entry + ROI data structure
column_names = [
    "Subfolder", "File Name", "RoiRank", "Feret", "MinFeret", "AvgFeret", "Area",
    "EquivDiameter", "Ellipse Major", "Ellipse Minor", "Circularity", "Roundness",
    "Solidity", "MeetsCriteria"
]

with open(output_csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=column_names)
    writer.writeheader()


# find all image files recursively in subfolders
image_files = sorted(input_dir.rglob("*.tif"))

def run_ilastik_object_classification(h5_path, prob_path, mask_path, object_project):
    subprocess.run([
        args.ilastik_exec,
        "--headless",
        "--project=" + str(object_project),
        "--raw_data=" + str(h5_path) + "/data",
        "--prediction_maps=" + str(prob_path) + "/exported_data",
        "--export_source=object identities",
        "--output_format=tiff",
        "--output_filename_format=" + str(mask_path)
    ], check=True)

def analyze_image(tif_path, object_project):
    relative_subfolder = tif_path.parent.relative_to(input_dir)
    prob_subfolder = prob_output_dir / relative_subfolder
    mask_subfolder = mask_dir / relative_subfolder
    prob_subfolder.mkdir(parents=True, exist_ok=True)
    mask_subfolder.mkdir(parents=True, exist_ok=True)

    h5_path = tif_path.with_suffix(".h5")
    with tifffile.TiffFile(tif_path) as tif:
        data = tif.series[0].asarray()
        if data.ndim == 3:
            if data.shape[0] in (3, 4): 
                data = data.mean(axis=0)
            elif data.shape[-1] in (3, 4): 
                data = data.mean(axis=-1)
            else:
                raise ValueError(f"Unexpected 3D image shape {data.shape} in {tif_path.name}")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=data.astype(np.float32), compression="gzip")

    prob_path = prob_subfolder / f"{tif_path.stem}_probabilities.h5"
    subprocess.run([
        args.ilastik_exec,
        "--headless",
        "--project=" + str(args.ilastik_project),
        "--export_source=Probabilities",
        "--output_format=hdf5",
        "--output_filename_format=" + str(prob_path),
        str(h5_path) + "/data"
    ], check=True)

    mask_path = mask_subfolder / f"{tif_path.stem}_objects.tiff"
    run_ilastik_object_classification(h5_path, prob_path, mask_path, object_project)

    h5_path.unlink()
    return mask_path, relative_subfolder

def process_mask(mask_path):
    if not mask_path.exists():
        print(f"\n+++++ Warning: Ilastik mask not found for {tif_path.name}, skipping analysis.")
        return []
    mask = imread(str(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    label_img = mask.astype(np.uint16)
    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels != 0]

    
    roi_data = []
    for label_val in unique_labels:
        roi_mask = (label_img == label_val).astype(np.uint8) * 255
        roi_ij = ij.py.to_imageplus(roi_mask)
        IJ.run(roi_ij, "Properties...", f"unit=um pixel_width={args.pix_width} pixel_height={args.pix_height}")
        IJ.run(roi_ij, "8-bit", "")
        roi_ij.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)
        rt_single = ResultsTable()
        pa = ParticleAnalyzer(
            ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES,
            Measurements.AREA | Measurements.FERET | Measurements.CIRCULARITY | Measurements.SHAPE_DESCRIPTORS | Measurements.CENTROID | Measurements.ELLIPSE,
            rt_single,
            args.min_size, 9999999999999999,
            0.2, 1.0
        )
        pa.setHideOutputImage(True)
        pa.analyze(roi_ij)
        if rt_single.size() == 0:
            print(f"\n+++++ No ROIs found for {tif_path.name}, detected organoid {label_val}.")
            continue
        for i in range(rt_single.size()):
            print(f"\n+++++ Processing organoid {label_val} for {tif_path.name}.")
            area = rt_single.getValue("Area", i)
            feret = rt_single.getValue("Feret", i)
            min_feret = rt_single.getValue("MinFeret", i)
            roundness = rt_single.getValue("Round", i) if rt_single.getColumnIndex("Round") != -1 else 1.0
            diameter = 2 * math.sqrt(area / (2 * math.pi))
            if feret >= args.feret_threshold:
                meets_all = (area > args.area_threshold) and (roundness > args.round_threshold)
                roi_data.append({
                    "RoiRank": label_val,
                    "Area": area,
                    "Feret": feret,
                    "MinFeret": min_feret,
                    "AvgFeret": (feret + min_feret) / 2,
                    "EquivDiameter": diameter,
                    "Ellipse Major": rt_single.getValue("Major", i),
                    "Ellipse Minor": rt_single.getValue("Minor", i),
                    "Circularity": rt_single.getValue("Circ.", i),
                    "Roundness": rt_single.getValue("Round", i),
                    "Solidity": rt_single.getValue("Solidity", i),
                    "MeetsCriteria": meets_all
                })
    return roi_data, label_img

color_palette = [scyjava.jimport("java.awt.Color").pink, 
                     scyjava.jimport("java.awt.Color").orange, 
                     scyjava.jimport("java.awt.Color").red, 
                     scyjava.jimport("java.awt.Color").yellow, 
                     scyjava.jimport("java.awt.Color").blue, 
                     scyjava.jimport("java.awt.Color").magenta, 
                     scyjava.jimport("java.awt.Color").green, 
                     scyjava.jimport("java.awt.Color").cyan]

# loop through all TIFFs
for i, tif_path in enumerate(image_files, start=1):
    print(f"\n+++++ [{i}/{len(image_files)}] Processing {tif_path.name} ...")
    output_entry = {
        "Subfolder": "NA",
        "File Name": tif_path.name,
        "RoiRank": "NA",
        "Feret": "NA",
        "MinFeret": "NA",
        "AvgFeret": "NA",
        "Area": "NA",
        "EquivDiameter": "NA",
        "Ellipse Major": "NA",
        "Ellipse Minor": "NA",
        "Circularity": "NA",
        "Roundness": "NA",
        "Solidity": "NA",
        "MeetsCriteria": "NA"
    }

    # attempt with strict object segmentation
    mask_path, relative_subfolder = analyze_image(tif_path, args.object_project)

    roi_data, label_img = process_mask(mask_path)

    # retry with alternate object segmentation if none passed
    if not roi_data and args.alt_object_project:
        print(f"\n+++++ No ROIs passed threshold. Retrying with alternate object classification model for {tif_path.name}")
        mask_path, _ = analyze_image(tif_path, args.alt_object_project)
        roi_data, label_img = process_mask(mask_path)

    if not roi_data:
        print(f"\n+++++ No ROIs passed threshold for {tif_path.name} even after fallback.")
        output_entry["Subfolder"] = relative_subfolder.as_posix()
        with open(output_csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writerow(output_entry)
        continue

    # successful ROI extraction
    roi_data.sort(key=lambda x: x["Area"], reverse=True)
    overlay = scyjava.jimport('ij.gui.Overlay')()
    for rank, roi in enumerate(roi_data, start=1):
        row = {**output_entry, **roi, "RoiRank": rank, "Subfolder": relative_subfolder.as_posix()}
        with open(output_csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writerow(row)

        roi_mask = (label_img == roi["RoiRank"]).astype(np.uint8) * 255
        roi_ij = ij.py.to_imageplus(roi_mask)
        IJ.run(roi_ij, "Create Selection", "")
        shape_roi = roi_ij.getRoi()
        if shape_roi is not None:
            shape_roi.setFillColor(color_palette[(rank - 1) % len(color_palette)])
            overlay.add(shape_roi)

    blank_img = ij.py.to_imageplus(np.zeros_like(label_img, dtype=np.uint8))
    IJ.run(blank_img, "Properties...", f"unit=um pixel_width={args.pix_width} pixel_height={args.pix_height}")
    blank_img.setOverlay(overlay)
    output_overlay_path = mask_dir / relative_subfolder / f"{tif_path.stem}_ROIoverlay.tif"
    IJ.save(blank_img, str(output_overlay_path))

    # memory cleanup
    try:
        roi_ij.close()
    except:
        pass
    try:
        blank_img.close()
    except:
        pass
    try:
        overlay.clear()
    except:
        pass

    del mask_path, label_img, roi_data, roi_ij, roi_mask, blank_img, overlay
    gc.collect()

print("\n+++++ Processing complete. Results saved to:", output_csv_path)

