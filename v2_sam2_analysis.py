
# SAM2 segmentation script v2
# vt 20 nov 2025
# updated to remove the masks that touch the edge of the image

import sys
sys.path.append('/home/vtaluja/hCOs/scripts/sam2_testing/sam2_repo') 
sys.path.append('/home/vtaluja/.local/lib/python3.11/site-packages')

import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import os
import argparse
import gc
import hydra

def show_anns(anns, borders=True):
    """
    annotation fxn, copied from SAM2 github
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            #import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def measure_mask_geometry(masks, img_name, results_csv):
    """
    Measures Feret diameters, roundness, and circularity for each SAM mask.

    Args:
        masks (list): SAM mask list (each has 'segmentation').
        img_name (str): Image filename.
        results_csv (str): Path to CSV file to append results.
    """
    records = []

    for i, ann in enumerate(masks):
        mask = ann["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 10 or perimeter == 0:
            continue  # skip noise

        # Convex hull for more stable Feret measurement
        hull = cv2.convexHull(contour)
        hull_points = hull[:, 0, :]

        # Compute max Feret: maximum distance between hull points
        dists = np.sqrt(((hull_points[:, None, :] - hull_points[None, :, :]) ** 2).sum(axis=2))
        feret_max = np.max(dists)

        # Approximate min Feret using rotated bounding boxes for multiple angles
        min_feret = np.inf
        for angle in range(0, 180, 5):  # coarse rotation sampling (faster)
            R = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
            rotated = np.dot(hull_points, R[:, :2].T)
            min_feret = min(min_feret, rotated[:, 0].max() - rotated[:, 0].min())
        feret_mean = 0.5 * (feret_max + min_feret)

        # Derived shape descriptors
        roundness = min_feret / feret_max if feret_max > 0 else np.nan
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan

        records.append({
            "image": img_name,
            "mask_index": i,
            "feret_max": feret_max,
            "feret_min": min_feret,
            "feret_mean": feret_mean,
            "roundness": roundness,
            "circularity": circularity,
            "area_px": area,
            "perimeter_px": perimeter
        })

    # Append to CSV
    df = pd.DataFrame(records)
    if not df.empty:
        header = not os.path.exists(results_csv)
        df.to_csv(results_csv, mode='a', index=False, header=header)


def generate_masks(folder_dir, results_dir='results', model_cfg="sam2.1_hiera_l.yaml", sam2_checkpoint="/home/vtaluja/hCOs/scripts/sam2_testing/sam2_repo/checkpoints/sam2.1_hiera_large.pt", min_mask_area = 100):
    """
    Generate and save segmentation masks for all .tif images in a directory.
    Trying w out this first , min_mask_region_area=100
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2.to(device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "sam2_measurements.csv")

    tif_files = glob.iglob(os.path.join(folder_dir, '*.tif'))
    for img_path in tif_files:
        print(f"Processing: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: unable to read {img_path}, skipping.")
            continue
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)

        # remove edge masks & small noise
        filtered_masks = []
        h, w = image_rgb.shape[:2]
        EDGE_TOL = 3
        
        for m in masks:
            mask = m["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            contour = max(contours, key=cv2.contourArea)
            # ---- edge skip ----
            on_edge = False
            for x, y in contour[:, 0, :]:
                if x <= EDGE_TOL or x >= w - 1 - EDGE_TOL or \
                   y <= EDGE_TOL or y >= h - 1 - EDGE_TOL:
                    on_edge = True
                    break
            if on_edge:
                continue
            # ---- area filter ----
            if m['area'] < min_mask_area:
                continue
        
            filtered_masks.append(m)
    
        filename = os.path.basename(img_path)
        measure_mask_geometry(filtered_masks, filename, results_csv)
        out_path = os.path.join(results_dir, filename)

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        show_anns(filtered_masks)
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close('all')  # prevents figure buildup in memory

        del masks, filtered_masks, img, image_rgb
        torch.cuda.empty_cache()  # helps prevent GPU OOM in large batches
        gc.collect() # force garbage collection

    print("All images processed.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organoid SAM pipeline")
    parser.add_argument("--input_dir", type=str, help="Folder with images for prediction", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Set the random seed
    np.random.seed(42)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir("/home/vtaluja/hCOs/scripts/sam2_testing/sam2_repo/sam2/configs/sam2.1")

    generate_masks(args.input_dir, results_dir = args.output_dir, min_mask_area = 500)













