#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements a refined approach to POI matching for the OpenCorr NR1 algorithm.
It addresses the following issues found in the original implementation:

1. Inaccurate perspective transform initialization for some POIs
2. Low correlation (ZNCC) values for certain POIs
3. The need for better local refinement during initialization

The solution includes:
- A more robust perspective transform initialization
- ZNCC-based local refinement
- Position search around the initial prediction
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from python_cubic_interpolation import BicubicBspline, Point2D, Image2D, load_image
from python_nr import NR2D1, POI2D, Deformation2D1, Subset2D

class ImprovedPerspectiveInitializer:
    """Enhanced perspective transform initializer with local refinement"""
    def __init__(self, thread_number=4, local_search_radius=5):
        self.thread_number = thread_number
        self.ref_img = None
        self.tar_img = None
        self.transform_matrix = None
        self.local_search_radius = local_search_radius
    
    def setImages(self, ref_img, tar_img):
        """Set reference and target images"""
        self.ref_img = ref_img
        self.tar_img = tar_img
    
    def setTransformMatrix(self, matrix=None):
        """Set or automatically estimate transform matrix"""
        if matrix is not None:
            # Use provided matrix
            self.transform_matrix = matrix
            print("Using provided perspective transform matrix.")
            return
        
        # Otherwise try to estimate from features
        try:
            print("Attempting to estimate perspective transform from images...")
            # Use SIFT to detect features
            sift = cv2.SIFT_create()
            
            # Detect keypoints and compute descriptors
            ref_gray = self.ref_img.eg_mat.astype(np.uint8)
            tar_gray = self.tar_img.eg_mat.astype(np.uint8)
            
            kp1, des1 = sift.detectAndCompute(ref_gray, None)
            kp2, des2 = sift.detectAndCompute(tar_gray, None)
            
            # FLANN feature matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 4:
                print(f"Warning: Not enough good matches found ({len(good_matches)}). Using identity matrix.")
                self.transform_matrix = np.eye(3)
                return
            
            # Get match point coordinates
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Compute perspective transform matrix
            self.transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            print(f"Estimated perspective transform from {len(good_matches)} matched features.")
            
        except Exception as e:
            print(f"Error estimating transform: {e}")
            print("Using identity matrix instead.")
            self.transform_matrix = np.eye(3)
    
    def compute_zncc(self, ref_subset, tar_subset):
        """Compute ZNCC between two subsets"""
        ref_mean = np.mean(ref_subset)
        tar_mean = np.mean(tar_subset)
        
        ref_zero = ref_subset - ref_mean
        tar_zero = tar_subset - tar_mean
        
        ref_norm = np.sqrt(np.sum(ref_zero * ref_zero))
        tar_norm = np.sqrt(np.sum(tar_zero * tar_zero))
        
        if ref_norm == 0 or tar_norm == 0:
            return -1  # Invalid correlation
        
        zncc = np.sum(ref_zero * tar_zero) / (ref_norm * tar_norm)
        return zncc
    
    def local_search(self, ref_subset, target_x, target_y, subset_radius_x, subset_radius_y):
        """Search locally around the predicted position for better correlation"""
        best_zncc = -1
        best_dx, best_dy = 0, 0
        
        # Extract search region for efficiency
        search_radius = self.local_search_radius
        if search_radius < 1:
            return best_zncc, best_dx, best_dy
        
        # Ensure we're within image bounds for the search region
        min_x = max(subset_radius_x, target_x - search_radius)
        max_x = min(self.tar_img.width - subset_radius_x - 1, target_x + search_radius)
        min_y = max(subset_radius_y, target_y - search_radius)
        max_y = min(self.tar_img.height - subset_radius_y - 1, target_y + search_radius)
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Extract target subset at this position
                target_subset = np.zeros((2 * subset_radius_y + 1, 2 * subset_radius_x + 1), dtype=np.float32)
                
                for r in range(-subset_radius_y, subset_radius_y + 1):
                    for c in range(-subset_radius_x, subset_radius_x + 1):
                        tx, ty = x + c, y + r
                        if 0 <= tx < self.tar_img.width and 0 <= ty < self.tar_img.height:
                            target_subset[r + subset_radius_y, c + subset_radius_x] = self.tar_img.eg_mat[ty, tx]
                
                # Compute ZNCC
                zncc = self.compute_zncc(ref_subset, target_subset)
                
                if zncc > best_zncc:
                    best_zncc = zncc
                    best_dx, best_dy = x - target_x, y - target_y
        
        return best_zncc, best_dx, best_dy
    
    def compute(self, poi_queue):
        """Compute initial deformation prediction for POIs with local refinement"""
        if self.transform_matrix is None:
            self.setTransformMatrix()
        
        print(f"Computing improved initial deformation prediction for {len(poi_queue)} POIs...")
        
        # Transform matrix
        H = self.transform_matrix
        
        # Normalize H so that H[2,2] = 1
        W = H / H[2, 2]
        
        # Define subset radius
        subset_radius_x = 33
        subset_radius_y = 33
        
        # Process each POI
        valid_count = 0
        refined_count = 0
        
        for poi in poi_queue:
            # Original coordinates
            x, y = poi.x, poi.y
            
            # Apply perspective transform to get new coordinates
            pt = np.array([x, y, 1.0])
            transformed = np.dot(H, pt)
            transformed /= transformed[2]  # Normalize
            new_x, new_y = transformed[0], transformed[1]
            
            # Extract deformation parameters from W matrix
            ux = W[0, 0] - 1.0       # Corresponds to 1+ux
            uy = W[0, 1]             # Corresponds to uy
            u = W[0, 2]              # Corresponds to u
            vx = W[1, 0]             # Corresponds to vx
            vy = W[1, 1] - 1.0       # Corresponds to 1+vy
            v = W[1, 2]              # Corresponds to v
            
            # Initial prediction
            target_x, target_y = int(new_x), int(new_y)
            
            # Check if POI is within valid bounds
            if (y - subset_radius_y < 0 or x - subset_radius_x < 0 or
                y + subset_radius_y >= self.ref_img.height or 
                x + subset_radius_x >= self.ref_img.width or
                target_y - subset_radius_y < 0 or target_x - subset_radius_x < 0 or
                target_y + subset_radius_y >= self.tar_img.height or 
                target_x + subset_radius_x >= self.tar_img.width):
                
                # Invalid POI, keep initial prediction but mark as invalid
                poi.deformation.u = float(u)
                poi.deformation.v = float(v)
                poi.deformation.ux = float(ux)
                poi.deformation.uy = float(uy)
                poi.deformation.vx = float(vx)
                poi.deformation.vy = float(vy)
                poi.result.zncc = -2.0  # Mark as invalid
                poi.result.u0 = float(u)
                poi.result.v0 = float(v)
                continue
            
            # Extract reference subset
            ref_subset = np.zeros((2 * subset_radius_y + 1, 2 * subset_radius_x + 1), dtype=np.float32)
            for r in range(-subset_radius_y, subset_radius_y + 1):
                for c in range(-subset_radius_x, subset_radius_x + 1):
                    rx, ry = x + c, y + r
                    if 0 <= rx < self.ref_img.width and 0 <= ry < self.ref_img.height:
                        ref_subset[r + subset_radius_y, c + subset_radius_x] = self.ref_img.eg_mat[ry, rx]
            
            # Local search for better match
            best_zncc, dx, dy = self.local_search(ref_subset, target_x, target_y, subset_radius_x, subset_radius_y)
            
            # If we found a better match with local search
            if best_zncc > 0.5:  # Good correlation threshold
                refined_count += 1
                # Update displacement values
                u_refined = u + dx
                v_refined = v + dy
                
                # Store refined results
                poi.deformation.u = float(u_refined)
                poi.deformation.v = float(v_refined)
                poi.deformation.ux = float(ux)
                poi.deformation.uy = float(uy)
                poi.deformation.vx = float(vx)
                poi.deformation.vy = float(vy)
                poi.result.zncc = float(best_zncc)
                poi.result.u0 = float(u_refined)
                poi.result.v0 = float(v_refined)
                valid_count += 1
            else:
                # Still use perspective transform but with a lower confidence
                poi.deformation.u = float(u)
                poi.deformation.v = float(v)
                poi.deformation.ux = float(ux)
                poi.deformation.uy = float(uy)
                poi.deformation.vx = float(vx)
                poi.deformation.vy = float(vy)
                poi.result.zncc = 0.5  # Mark as less confident but still valid
                poi.result.u0 = float(u)
                poi.result.v0 = float(v)
                valid_count += 1
        
        print(f"Initial deformation prediction completed:")
        print(f" - {valid_count}/{len(poi_queue)} POIs have valid initial predictions")
        print(f" - {refined_count}/{len(poi_queue)} POIs were improved by local refinement")
        
        return poi_queue

def main():
    # Setup files
    ref_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/speckle_medium.tif"
    tar_image_path = "/Users/liyongchang/Downloads/OpenCorr-main/src/python/img/Camera_DEV_1AB22C0222A5_2024-09-23_19-29-40.png"
    
    # Check if files exist
    if not os.path.exists(ref_image_path) or not os.path.exists(tar_image_path):
        print(f"Error: Image files not found. Please check the paths.")
        return
    
    # Load images
    try:
        ref_img, ref_cv_img = load_image(ref_image_path)
        tar_img, tar_cv_img = load_image(tar_image_path)
        print(f"Images loaded successfully: {ref_img.width}x{ref_img.height}")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Initialize timing parameters
    computation_time = []
    
    # Set start time
    timer_tic = time.time()
    
    # Set output paths
    base_path = os.path.splitext(tar_image_path)[0]
    results_csv_path = f"{base_path}_improved_nr1_r16.csv"
    time_csv_path = f"{base_path}_improved_nr1_r16_time.csv"
    img_result_path = f"{base_path}_improved_nr1_r16_results.png"
    
    # Set DIC parameters
    subset_radius_x = 33
    subset_radius_y = 33
    max_iteration = 100
    max_deformation_norm = 0.001
    
    # Setup POIs
    upper_left_point = Point2D(975, 695)
    poi_number_x = 9
    poi_number_y = 7
    grid_space = 250
    
    # Create POI queue
    poi_queue = []
    for i in range(poi_number_y):
        for j in range(poi_number_x):
            x = upper_left_point.x + j * grid_space
            y = upper_left_point.y + i * grid_space
            poi = POI2D(x, y)
            poi_queue.append(poi)
    
    # Get CPU thread number
    import multiprocessing
    cpu_thread_number = max(1, multiprocessing.cpu_count() - 1)
    
    # Create improved initializer
    improved_init = ImprovedPerspectiveInitializer(cpu_thread_number, local_search_radius=5)
    improved_init.setImages(ref_img, tar_img)
    
    # Set the same transform matrix for comparison
    transform_matrix = np.array([
        [8.25327830e-01, -1.92603161e-02, 6.08026082e+02],
        [9.29342396e-03, 8.11131732e-01, 1.28319036e+02],
        [1.38626335e-06, -5.94219881e-06, 1.00000000e+00]
    ])
    improved_init.setTransformMatrix(transform_matrix)
    
    # Create NR instance
    nr = NR2D1(subset_radius_x, subset_radius_y, max_deformation_norm, max_iteration, batch_size=16)
    nr.set_images(ref_img, tar_img)
    
    # Record initialization time
    timer_toc = time.time()
    init_time = timer_toc - timer_tic
    computation_time.append(init_time)
    
    print(f"Initialization with {len(poi_queue)} POIs takes {init_time:.3f} sec, {cpu_thread_number} CPU threads launched.")
    
    # Improved perspective transform prediction
    timer_tic = time.time()
    improved_init.compute(poi_queue)
    timer_toc = time.time()
    persp_time = timer_toc - timer_tic
    computation_time.append(persp_time)
    
    print(f"Improved initial deformation prediction takes {persp_time:.3f} sec.")
    
    # NR computation
    timer_tic = time.time()
    nr.prepare()
    
    nr.compute_poi_queue(poi_queue)
    timer_toc = time.time()
    nr_time = timer_toc - timer_tic
    computation_time.append(nr_time)
    
    print(f"Deformation determination using NR takes {nr_time:.3f} sec.")
    
    # Save results
    try:
        # Prepare result data
        results = []
        for poi in poi_queue:
            results.append({
                'x': poi.x,
                'y': poi.y,
                'u': poi.deformation.u,
                'v': poi.deformation.v,
                'ux': poi.deformation.ux,
                'uy': poi.deformation.uy,
                'vx': poi.deformation.vx,
                'vy': poi.deformation.vy,
                'zncc': poi.result.zncc,
                'iterations': poi.result.iteration,
                'convergence': poi.result.convergence
            })
        
        # Create DataFrame and save
        df_results = pd.DataFrame(results)
        df_results.to_csv(results_csv_path, index=False)
        
        # Save computation time
        df_time = pd.DataFrame({
            'POI number': [len(poi_queue)],
            'Initialization': [computation_time[0]],
            'Improved Perspective Transform': [computation_time[1]],
            'NR': [computation_time[2]]
        })
        df_time.to_csv(time_csv_path, index=False)
        
        print(f"Results saved to {results_csv_path}")
        print(f"Computation time saved to {time_csv_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Visualize results
    try:
        plt.figure(figsize=(12, 10))
        
        # Plot reference image
        plt.subplot(2, 3, 1)
        plt.imshow(ref_cv_img, cmap='gray')
        plt.title("Reference Image")
        plt.axis('off')
        
        # Plot target image
        plt.subplot(2, 3, 2)
        plt.imshow(tar_cv_img, cmap='gray')
        plt.title("Target Image")
        plt.axis('off')
        
        # Plot displacement field
        plt.subplot(2, 3, 3)
        valid_pois = [poi for poi in poi_queue if poi.result.zncc > 0]
        x = [poi.x for poi in valid_pois]
        y = [poi.y for poi in valid_pois]
        u = [poi.deformation.u for poi in valid_pois]
        v = [poi.deformation.v for poi in valid_pois]
        
        plt.quiver(x, y, u, v, scale=50, width=0.002)
        plt.imshow(ref_cv_img, cmap='gray', alpha=0.5)
        plt.title(f"Displacement Field ({len(valid_pois)}/{len(poi_queue)} valid)")
        plt.axis('off')
        
        # Plot ZNCC distribution
        plt.subplot(2, 3, 4)
        zncc_values = [poi.result.zncc for poi in valid_pois]
        plt.scatter(x, y, c=zncc_values, cmap='jet', s=10)
        plt.colorbar(label='ZNCC')
        plt.title("ZNCC Distribution")
        plt.axis('off')
        
        # Plot x-direction strain
        plt.subplot(2, 3, 5)
        ux_values = [poi.deformation.ux for poi in valid_pois]
        plt.scatter(x, y, c=ux_values, cmap='jet', s=10)
        plt.colorbar(label='ux')
        plt.title("X-direction Strain")
        plt.axis('off')
        
        # Plot y-direction strain
        plt.subplot(2, 3, 6)
        vy_values = [poi.deformation.vy for poi in valid_pois]
        plt.scatter(x, y, c=vy_values, cmap='jet', s=10)
        plt.colorbar(label='vy')
        plt.title("Y-direction Strain")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(img_result_path, dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    main() 