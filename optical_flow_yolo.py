"""
Neuromorphic Optical Flow with YOLO Integration

This script combines memristor-based region-of-interest extraction with optical flow computation
and YOLO object detection. The workflow is:

1. Extract regions of interest using memristor data that exceed a threshold
2. Compute optical flow only in these regions (memory efficient)
3. Run YOLO object detection on the extracted regions
4. Perform motion segmentation
5. Save results including YOLO detections

YOLO Configuration:
- Model: Set YOLO_MODEL_PATH (default: yolov8n.pt)
- Confidence: Set YOLO_CONFIDENCE (default: 0.5)
- IOU Threshold: Set YOLO_IOU_THRESHOLD (default: 0.45)
- Save Path: Set YOLO_SAVE_PATH for detection results

Requirements:
- ultralytics (for YOLO): pip install ultralytics
- opencv-python, numpy, scipy, numba, PIL
"""

import time
import cv2
import numpy as np
import flow_viz
from PIL import Image
import os
import scipy.io
from numba import njit
import csv
# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available. YOLO functionality will be disabled.")
    YOLO_AVAILABLE = False

def viz(flo, imgname):  
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    #save by name
    img_rgb = flo[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    imgpil = Image.fromarray(img_rgb_uint8)
    imgpil.save(imgname)

# data path
# Default data paths (can be overridden by command-line arguments or a config file)
# Note: Replace these with your actual data paths
RGB_PATH = 'data/autodriving/RGB'
MASK_PATH = 'data/autodriving/gtmask'
IMGLIST = 'data/autodriving/imgs.txt'
MEMMATPATH = 'data/autodriving/constructed_3D_matrix.mat'

# Output paths
OUTPUT_DIR = 'output/autodriving_yolo'
SEG_SAVE_PATH = os.path.join(OUTPUT_DIR, 'segimg')
SEG_SAVE_PATH2 = os.path.join(OUTPUT_DIR, 'originalsegimg')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'farneback_seg.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_seg.csv')
YOLO_SAVE_PATH = os.path.join(OUTPUT_DIR, 'yolo_detections')


MEMSIZE= 200
OFFSET = 15

EXTEND_HEIGHT_UPPER = 60
EXTEND_HEIGHT_LOWER = 60
EXTEND_WIDTH_LEFT = 60
EXTEND_WIDTH_RIGHT = 60
THRES = 114
CONNECT  = 4
FLAG = 2

PADDING = 60    
SEG_TH = 1

# YOLO configuration
YOLO_MODEL_PATH = r'yolov8n.pt'  # You can change this to yolov8s.pt, yolov8m.pt, etc.
YOLO_CONFIDENCE = 0.25
YOLO_IOU_THRESHOLD = 0.45
# YOLO_SAVE_PATH = fr'F:\shengbowang_v3\shengbowang_v3\neuromorphic optical flow\Results\0611\results\autodriving\yolo_detections'

mem_opticalflow_times = []
mem_cal_times = []
mem_velocity_times = []

mem_task_times = []
mem_combination_times = []

original_opticalflow_times = []
original_task_times = []

# YOLO timing lists
yolo_region_times = []
yolo_full_times = []

# # tabletennis farneback parameters
# farneback_params = {
#         'pyr_scale': 0.6,
#         'levels': 3,
#         'winsize': 4,
#         'iterations': 2,
#         'poly_n': 1,
#         'poly_sigma': 1.05,
#         'flags': 0
#     }

# uavnew2 grasp farneback parameters
farneback_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

# # autodriving farneback parameters
# farneback_params = {
#         'pyr_scale': 0.6,
#         'levels': 3,
#         'winsize': 3,
#         'iterations': 3,
#         'poly_n': 10,
#         'poly_sigma': 1.05,
#         'flags': 0
#     }

# farneback_params = {
#         'pyr_scale': 0.5,
#         'levels': 3,
#         'winsize': 15,
#         'iterations': 3,
#         'poly_n': 5,
#         'poly_sigma': 1.2,
#         'flags': 0
#     }

# farneback_params = {
#         'pyr_scale': 0.6,
#         'levels': 3,
#         'winsize': 4,
#         'iterations': 2,
#         'poly_n': 1,
#         'poly_sigma': 1.05,
#         'flags': 0
#     }

# Add a new function to compare with the threshold to create a mask for speed estimation
@njit
def update_transition_pic(prev_memristor, transition_pic, thres):
    for i in range(prev_memristor.shape[0]):
        for j in range(prev_memristor.shape[1]):
            if prev_memristor[i, j] >= thres:
                transition_pic[i, j] = 255
    return transition_pic

def process_separate_regions(stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    """Separate processing FLAG=1"""
    region_list = []
    regions_info = []
    h, w = rgbimg1.shape[:2]

    for i in range(1, len(stats)):
        x, y, a, b = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate region boundaries
        region_start_time = time.time()
        x_start = max(x * pixel_width - EXTEND_WIDTH_LEFT, 0)
        y_start = max(y * pixel_height - EXTEND_HEIGHT_UPPER, 0)
        x_end = min((x + a) * pixel_width + EXTEND_WIDTH_RIGHT, w)
        y_end = min((y + b) * pixel_height + EXTEND_HEIGHT_LOWER, h)
        
        regions_info.append((x_start, y_start, x_end, y_end))
        prev_region = rgbimg1[y_start:y_end, x_start:x_end]
        next_region = rgbimg2[y_start:y_end, x_start:x_end]

        region_end_time = time.time()
        mem_cal_times.append(region_end_time - region_start_time)   

        # Calculate region percentage
        region_pixels = prev_region.shape[0] * prev_region.shape[1]
        total_pixels = rgbimg1.shape[0] * rgbimg1.shape[1]
        region_percentage = (region_pixels / total_pixels) * 100
        region_list.append(region_percentage)
        print(f"Region covers {region_percentage:.2f}% of the full image\n"
                f"Memristor process time is {region_end_time - region_start_time}")
        
        # Calculate optical flow
        if prev_region.size > 0 and next_region.size > 0:
            start_time = time.time()
            current_flow = cv2.calcOpticalFlowFarneback(prev_region, next_region, None, **farneback_params)
            flow[y_start:y_end, x_start:x_end] = current_flow
            end_time = time.time()
            mem_velocity_times.append(end_time - start_time)
            print(f"Separate opticalflow process time is {end_time - start_time}")
        else:
            mem_velocity_times.append(0)
            
    return flow, mem_cal_times, mem_velocity_times, region_list, len(stats), regions_info

def process_merged_region(stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    """Merge processing FLAG=2"""
    region_list = []
    h, w = rgbimg1.shape[:2]
    start_time = time.time()

    x_min = min(stats[i, cv2.CC_STAT_LEFT] for i in range(1, len(stats)))
    y_min = min(stats[i, cv2.CC_STAT_TOP] for i in range(1, len(stats)))
    x_max = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in range(1, len(stats)))
    y_max = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, len(stats)))
    
    # Calculate extended boundaries
    x_start = max(x_min * pixel_width - EXTEND_WIDTH_LEFT, 0)
    y_start = max(y_min * pixel_height - EXTEND_HEIGHT_UPPER, 0)
    x_end = min(x_max * pixel_width + EXTEND_WIDTH_RIGHT, w)
    y_end = min(y_max * pixel_height + EXTEND_HEIGHT_LOWER, h)
    
    # Extract region
    prev_region = rgbimg1[y_start:y_end, x_start:x_end]
    next_region = rgbimg2[y_start:y_end, x_start:x_end]

    end_time = time.time()
    mem_cal_times.append(end_time - start_time) 

    # Calculate region percentage
    region_pixels = prev_region.shape[0] * prev_region.shape[1]
    total_pixels = rgbimg1.shape[0] * rgbimg1.shape[1]
    region_percentage = (region_pixels / total_pixels) * 100
    region_list.append(region_percentage)
    print(f"Merged region covers {region_percentage:.2f}% of the full image\n"
          f"Memristor process time is {end_time - start_time}")
    
    # Calculate optical flow
    if prev_region.size > 0 and next_region.size > 0:
        start_time = time.time()
        current_flow = cv2.calcOpticalFlowFarneback(prev_region, next_region, None, **farneback_params)
        flow[y_start:y_end, x_start:x_end] = current_flow
        end_time = time.time()
        mem_velocity_times.append(end_time - start_time)
        print(f"Merged opticalflow process time is {end_time - start_time}")
 
    return flow, mem_cal_times, mem_velocity_times, region_list, (x_start, y_start, x_end, y_end)

def opticalFlow3D(memimg1, memimg2, rgbimg1, rgbimg2, pixel_width, pixel_height):
    """Main optical flow calculation function"""
    start_time = time.time()
    h, w = rgbimg1.shape[:2]
    flow = np.zeros((h, w, 2))
    
    # Create transition image
    transition_pic = np.zeros((int(h / pixel_height), int(w / pixel_width)))
    transition_pic = update_transition_pic(memimg2, transition_pic, THRES)
    transition_pic = transition_pic.astype(np.uint8)
    
    # Find connected areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        transition_pic, connectivity=CONNECT)
    
    # If no area is found, return directly
    if num_labels == 1:
        end_time = time.time()
        mem_cal_times.append(end_time - start_time)
        mem_opticalflow_times.append(end_time - start_time)
        if FLAG == 1:
            return flow, mem_cal_times, mem_opticalflow_times, [], num_labels, []
        else:
            return flow, mem_cal_times, mem_opticalflow_times, [], (0, 0, 0, 0)
    
    # There is a region that needs to be processed. Select the processing method according to the FLAG.
    if FLAG == 1:
        flow, cal_times, vel_times, region_list, num_labels, regions_info = process_separate_regions(
            stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height)
    else:
        flow, cal_times, vel_times, region_list, regions_info = process_merged_region(
            stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height)
    
    # Add total processing time measurement
    end_time = time.time()
    mem_opticalflow_times.append(end_time - start_time)
    
    # return result
    if FLAG == 1:
        return flow, cal_times, vel_times, region_list, num_labels, regions_info
    else:
        return flow, cal_times, vel_times, region_list, regions_info

def task_results(prev_frame, next_frame, flow, num_labels, regions_info, EST_FLAG, MERGE_FLAG):
    """
    Process optical flow results
    FLAG: 1 or 2 represents the region estimation method
    MERGE_FLAG: Effective when FLAG=1. True means merge processing, False means separate processing.
    """
    start_time = time.time()
    start_time_combined = time.time()
    h, w = prev_frame.shape[:2]
    motion_binary = np.zeros((h, w), dtype=np.uint8)

    # Create HSV image for optical flow visualization
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    if num_labels > 1:
        if EST_FLAG == 1:
            if MERGE_FLAG:
                # Merge processing mode
                padding = PADDING
                x_min = max(0, min(region[0] for region in regions_info) - padding)
                y_min = max(0, min(region[1] for region in regions_info) - padding)
                x_max = min(w, max(region[2] for region in regions_info) + padding)
                y_max = min(h, max(region[3] for region in regions_info) + padding)
                
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                
                # Process the optical flow of the selected region
                flow_region = flow[y_min:y_max, x_min:x_max]
                mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                
                # Process selected regions
                motion_binary[y_min:y_max, x_min:x_max] = process_flow_region(mag, ang)
            else:
                # Separate processing
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                
                for x_min, y_min, x_max, y_max in regions_info:
                    flow_region = flow[y_min:y_max, x_min:x_max]
                    mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                    
                    # Process current area
                    motion_binary[y_min:y_max, x_min:x_max] = process_flow_region(mag, ang)

        else:  # EST_FLAG == 2
            # Process directly using the merged area
            x_min, y_min, x_max, y_max = regions_info
            
            end_time_combined = time.time()
            mem_combination_times.append(end_time_combined - start_time_combined)
            
            flow_region = flow[y_min:y_max, x_min:x_max]
            mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
            
            # Process selected regions
            motion_binary[y_min:y_max, x_min:x_max] = process_flow_region(mag, ang)
 
    else:
        end_time_combined = time.time()
        mem_combination_times.append(end_time_combined - start_time_combined)
    
    end_time = time.time()
    mem_task_times.append(end_time - start_time)
    
    return motion_binary

def process_flow_region(mag, ang):
    """
    Process optical flow data in a specific region and generate a motion segmentation mask
    """
    # Create an HSV image of the size of the region
    region_hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    region_hsv[..., 1] = 255
    
    # Map angle to hue channel
    region_hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Normalize optical flow amplitude to brightness channel
    region_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to BGR then to grayscale
    bgr = cv2.cvtColor(region_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Thresholding processing
    threshold = SEG_TH
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Create motion mask
    motion_mask = np.zeros_like(gray)
    motion_mask[mag > threshold] = 255
    
    # Morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    for _ in range(5):  # 5次迭代 -> 5 iterations
        motion_mask = cv2.dilate(motion_mask, kernel)
        motion_mask = cv2.erode(motion_mask, kernel)
    
    # Final binarization
    _, motion_binary = cv2.threshold(motion_mask, 1, 255, cv2.THRESH_BINARY)
    
    return motion_binary

def prepare_folder():
    os.makedirs(SEG_SAVE_PATH, exist_ok=True)
    os.makedirs(SEG_SAVE_PATH2, exist_ok=True)
    os.makedirs(YOLO_SAVE_PATH, exist_ok=True)  # Create YOLO save directory
    with open(SAVETXTPATH, 'w', encoding='utf-8') as file:
        pass 

    # Create and initialize CSV file
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Frame_Pair',
            'Original_Flow_Time',
            'Mem_Flow_Time',
            'Flow_Time_Improvement',
            'Flow_Time_Improvement_Percent',
            'Original_Seg_Time',
            'Mem_Seg_Time',
            'Combination_Time',
            'Original_PA',
            'Mem_PA',
            'Region_Percent',
            'Cal_Times',           # Added: all calculation times
            'Velocity_Times',      # Added: all speed times
            'YOLO_Region_Time',    # Added: Area YOLO processing time
            'YOLO_Full_Time',      # Added: Full image YOLO processing time
            'YOLO_Time_Improvement',  # Added: YOLO time improvement
            'YOLO_Time_Improvement_Percent',  # Added: YOLO time improvement percentage
            'YOLO_Region_Detections_Count',  # Added: Number of YOLO detections in the area
            'YOLO_Full_Detections_Count',    # Added: Number of YOLO detections in the full image
            'YOLO_Region_Classes',        # Added: Categories detected in the area
            'YOLO_Region_Confidences',    # Added: Regional Confidence
            'YOLO_Full_Classes',          # Added: Categories detected in the full image
            'YOLO_Full_Confidences'       # Added: full image confidence
        ])

def calculate_pixel_accuracy(image1, image2):
    total_pixels = image1.size
    matching_pixels = np.sum(image1 == image2)
    accuracy = (matching_pixels / total_pixels) * 100
    return accuracy

def run_yolo_on_regions(frame, regions_info, model, frame_name, save_detections=True):
    """
    Run YOLO detection on extracted regions of interest
    
    Args:
        frame: Input image (BGR format)
        regions_info: List of region coordinates [(x_start, y_start, x_end, y_end), ...] or single tuple
        model: Loaded YOLO model
        frame_name: Name for saving detection results
        save_detections: Whether to save detection images
    
    Returns:
        all_detections: List of detection results with full image coordinates
        detection_image: Image with all detections drawn
        processing_time: Total time spent on YOLO processing
    """
    start_time = time.time()
    
    if not YOLO_AVAILABLE:
        print("YOLO not available. Skipping detection.")
        return [], frame.copy(), 0.0
    
    all_detections = []
    detection_image = frame.copy()
    h, w = frame.shape[:2]
    yolo_inference_time = 0.0
    
    # Handle both single region (FLAG=2) and multiple regions (FLAG=1)
    if isinstance(regions_info, tuple):
        # Single region case (FLAG=2)
        regions_list = [regions_info]
    else:
        # Multiple regions case (FLAG=1)
        regions_list = regions_info if regions_info else []
    
    region_count = 0
    for region_coords in regions_list:
        if len(region_coords) != 4:
            continue
            
        x_start, y_start, x_end, y_end = region_coords
        
        # Ensure coordinates are within image bounds
        x_start = max(0, int(x_start))
        y_start = max(0, int(y_start))
        x_end = min(w, int(x_end))
        y_end = min(h, int(y_end))
        
        # Skip invalid regions
        if x_end <= x_start or y_end <= y_start:
            continue
            
        # Extract region
        region = frame[y_start:y_end, x_start:x_end]
        
        if region.size == 0:
            continue
            
        try:
            # Run YOLO on the region with timing
            yolo_start = time.time()
            results = model(region, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD, verbose=False)
            yolo_end = time.time()
            yolo_inference_time += (yolo_end - yolo_start)
            
            # Process detections
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        # Map region coordinates back to full image coordinates
                        x1_full = x_start + box[0]
                        y1_full = y_start + box[1]
                        x2_full = x_start + box[2]
                        y2_full = y_start + box[3]
                        
                        # Store detection info
                        detection_info = {
                            'bbox': [x1_full, y1_full, x2_full, y2_full],
                            'confidence': conf,
                            'class': int(cls),
                            'class_name': model.names[int(cls)],
                            'region': region_count,
                            'region_coords': region_coords
                        }
                        all_detections.append(detection_info)
                        
                        # Draw detection on full image
                        cv2.rectangle(detection_image, 
                                    (int(x1_full), int(y1_full)), 
                                    (int(x2_full), int(y2_full)), 
                                    (0, 255, 0), 2)
                        
                        # Add label
                        label = f"{model.names[int(cls)]}: {conf:.2f}"
                        cv2.putText(detection_image, label, 
                                  (int(x1_full), int(y1_full) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw region boundary
            cv2.rectangle(detection_image, 
                        (x_start, y_start), (x_end, y_end), 
                        (255, 0, 0), 2)
            cv2.putText(detection_image, f"Region {region_count}", 
                      (x_start, y_start - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                      
        except Exception as e:
            print(f"Error running YOLO on region {region_count}: {e}")
            
        region_count += 1
    
    # Add text overlay if no detections found
    if not all_detections and region_count > 0:
        cv2.putText(detection_image, "No objects detected in memristor regions", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Save detection results if requested (always save, even with no detections)
    if save_detections:
        os.makedirs(YOLO_SAVE_PATH, exist_ok=True)
        detection_path = os.path.join(YOLO_SAVE_PATH, f"det_region_{frame_name}")
        cv2.imwrite(detection_path, detection_image)
        
        # Save detection info to text file
        txt_path = os.path.join(YOLO_SAVE_PATH, f"det_region_{frame_name.replace('.jpg', '.txt')}")
        with open(txt_path, 'w') as f:
            f.write(f"Region-based Detections for {frame_name}:\n")
            f.write(f"Total regions processed: {region_count}\n")
            f.write(f"Total detections: {len(all_detections)}\n")
            f.write(f"YOLO inference time: {yolo_inference_time:.4f}s\n\n")
            if all_detections:
                for i, det in enumerate(all_detections):
                    f.write(f"Detection {i+1}:\n")
                    f.write(f"  Class: {det['class_name']} (ID: {det['class']})\n")
                    f.write(f"  Confidence: {det['confidence']:.3f}\n")
                    f.write(f"  Bbox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]\n")
                    f.write(f"  Region: {det['region']} {det['region_coords']}\n\n")
            else:
                f.write("No objects detected in the memristor-based regions.\n")
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    return all_detections, detection_image, total_processing_time

def run_yolo_on_full_image(frame, model, frame_name, save_detections=True):
    """
    Run YOLO detection on the full image
    
    Args:
        frame: Input image (BGR format)
        model: Loaded YOLO model
        frame_name: Name for saving detection results
        save_detections: Whether to save detection images
    
    Returns:
        all_detections: List of detection results
        detection_image: Image with all detections drawn
        processing_time: Total time spent on YOLO processing
    """
    start_time = time.time()
    
    if not YOLO_AVAILABLE:
        print("YOLO not available. Skipping detection.")
        return [], frame.copy(), 0.0
    
    all_detections = []
    detection_image = frame.copy()
    
    try:
        # Run YOLO on the full image with timing
        yolo_start = time.time()
        results = model(frame, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD, verbose=False)
        yolo_end = time.time()
        yolo_inference_time = yolo_end - yolo_start
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    # Store detection info
                    detection_info = {
                        'bbox': [box[0], box[1], box[2], box[3]],
                        'confidence': conf,
                        'class': int(cls),
                        'class_name': model.names[int(cls)]
                    }
                    all_detections.append(detection_info)
                    
                    # Draw detection on image
                    cv2.rectangle(detection_image, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 0, 255), 2)
                    
                    # Add label
                    label = f"{model.names[int(cls)]}: {conf:.2f}"
                    cv2.putText(detection_image, label, 
                              (int(box[0]), int(box[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add text overlay if no detections found
        if not all_detections:
            cv2.putText(detection_image, "No objects detected in full image", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Save detection results if requested (always save, even with no detections)
        if save_detections:
            os.makedirs(YOLO_SAVE_PATH, exist_ok=True)
            detection_path = os.path.join(YOLO_SAVE_PATH, f"det_full_{frame_name}")
            cv2.imwrite(detection_path, detection_image)
            
            # Save detection info to text file
            txt_path = os.path.join(YOLO_SAVE_PATH, f"det_full_{frame_name.replace('.jpg', '.txt')}")
            with open(txt_path, 'w') as f:
                f.write(f"Full-image Detections for {frame_name}:\n")
                f.write(f"Total detections: {len(all_detections)}\n")
                f.write(f"YOLO inference time: {yolo_inference_time:.4f}s\n\n")
                if all_detections:
                    for i, det in enumerate(all_detections):
                        f.write(f"Detection {i+1}:\n")
                        f.write(f"  Class: {det['class_name']} (ID: {det['class']})\n")
                        f.write(f"  Confidence: {det['confidence']:.3f}\n")
                        f.write(f"  Bbox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]\n\n")
                else:
                    f.write("No objects detected in the full image.\n")
                    
    except Exception as e:
        print(f"Error running YOLO on full image: {e}")
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    return all_detections, detection_image, total_processing_time

def load_yolo_model():
    """Load YOLO model"""
    if not YOLO_AVAILABLE:
        return None
        
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded successfully: {YOLO_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def example_yolo_usage():
    """
    Example of how to use YOLO with custom settings
    This function shows how to modify YOLO parameters for different use cases
    """
    global YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD
    
    # Example 1: High precision detection
    print("Example 1: High precision detection")
    YOLO_MODEL_PATH = 'yolov8m.pt'  # Medium model for better accuracy
    YOLO_CONFIDENCE = 0.7           # Higher confidence threshold
    YOLO_IOU_THRESHOLD = 0.3        # Lower IOU for stricter NMS
    
    # Example 2: Fast detection for real-time applications
    print("Example 2: Fast detection")
    YOLO_MODEL_PATH = 'yolov8n.pt'  # Nano model for speed
    YOLO_CONFIDENCE = 0.3           # Lower confidence for more detections
    YOLO_IOU_THRESHOLD = 0.5        # Standard IOU threshold
    
    # Example 3: Specific object classes (modify in run_yolo_on_regions if needed)
    print("Example 3: For detecting specific classes, modify the model results processing")
    print("You can filter by class IDs in the run_yolo_on_regions function")
    
    # Reset to defaults
    YOLO_MODEL_PATH = r'yolov8n.pt'
    YOLO_CONFIDENCE = 0.5
    YOLO_IOU_THRESHOLD = 0.45

if __name__ == '__main__':
    prepare_folder()
    
    # Load YOLO model
    yolo_model = load_yolo_model()
    if yolo_model is None:
        print("Continuing without YOLO detection...")
    
    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()

    # No longer read the picture, but read a .mat file organized by matlab. In this case, there will be a displacement in the serial number when reading, because the first 15 in this .mat file are a piece of empty data (equivalent to the initial setting of the memristor) #
    mem_data = scipy.io.loadmat(MEMMATPATH)
    mem_state = mem_data['constructed3DMatrix']

    rgbimages = [os.path.join(RGB_PATH,i) for i in imgs]
    rgbimages = sorted(rgbimages, key=lambda x: int(x.split('\\')[-1].split('.')[0]))

    # Create an HSV image of the same size as the first frame
    hsv = np.zeros_like(
        cv2.imread(f"{RGB_PATH}/41.jpg"))
    hsv[..., 1] = 255
    total_accuracy1 = 0
    total_accuracy2 = 0
    ourtime = 0
    originaltime = 0
    cnt = 0
    for i in range(len(imgs) - 2):
        mem_state1 = mem_state[:, :, OFFSET + i].astype(np.double)
        mem_state2 = mem_state[:, :, OFFSET + i + 1].astype(np.double) # The value actually passed
        rgbimfile1 = rgbimages[i]
        rgbimfile2 = rgbimages[i + 1]
        filename1 = os.path.basename(rgbimfile1)
        filename2 = os.path.basename(rgbimfile2)
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Calculation between {filename1} and {filename2}\n')

        with np.errstate(divide='ignore', invalid='ignore'):  # To handle log10 of non-positive values safely
            mem_state1 = - 3366 / np.log10(mem_state1) - 306
        mem_state1 = np.clip(mem_state1, 0, 255)
        # Convert to uint8
        mem_state1 = mem_state1.astype(np.uint8)
        with np.errstate(divide='ignore', invalid='ignore'):  # To handle log10 of non-positive values safely
            mem_state2 = - 3366 / np.log10(mem_state2) - 306
        mem_state2 = np.clip(mem_state2, 0, 255)
        # Convert to uint8
        mem_state2 = mem_state1.astype(np.uint8)
        memimg1 = mem_state1  # memimg1 is a 2D matrix representing memristor state at time step 15 + idx
        memimg2 = mem_state2  # memimg2 is a 2D matrix representing memristor state at time step 16 + idx
        # breakpoint()
        prev_frame = cv2.imread(rgbimages[i])
        next_frame = cv2.imread(rgbimages[i + 1])

        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        # Ground Truth Reading
        image = cv2.imread(f"{MASK_PATH}/{imgs[i + 1]}")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 256, cv2.THRESH_BINARY)

        pixel_width = MEMSIZE
        pixel_height = MEMSIZE
        h, w = next_frame.shape[:2]

        # Calculate optical flow (memristor-accelerated)
        if FLAG == 1:
            flow, cal_times, vel_times, region_list, num_labels, regions_info = opticalFlow3D(memimg1, memimg2, prev_frame_gray, next_frame_gray, pixel_width, pixel_height)
        else:
            flow, cal_times, vel_times, region_list, (x_start, y_start, x_end, y_end) = opticalFlow3D(memimg1, memimg2, prev_frame_gray, next_frame_gray, pixel_width, pixel_height)

        # Invert optical flow (only for Farneback)
        flow = -flow

        # Run YOLO detection on extracted regions
        if yolo_model is not None:
            if FLAG == 1:
                # Multiple regions case
                if regions_info:  # Check if regions_info is not empty
                    yolo_detections, yolo_image, yolo_time = run_yolo_on_regions(
                        next_frame, regions_info, yolo_model, imgs[i + 1]
                    )
                else:
                    # No memristor regions found - save image with notification
                    yolo_detections, yolo_image, yolo_time = [], next_frame.copy(), 0.0
                    cv2.putText(yolo_image, "No memristor regions activated", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # Save the image
                    os.makedirs(YOLO_SAVE_PATH, exist_ok=True)
                    detection_path = os.path.join(YOLO_SAVE_PATH, f"det_region_{imgs[i + 1]}")
                    cv2.imwrite(detection_path, yolo_image)
                    # Save text file
                    txt_path = os.path.join(YOLO_SAVE_PATH, f"det_region_{imgs[i + 1].replace('.jpg', '.txt')}")
                    with open(txt_path, 'w') as f:
                        f.write(f"Region-based Detections for {imgs[i + 1]}:\n")
                        f.write("No memristor regions were activated.\n")
            else:
                # Single merged region case
                if x_start != 0 or y_start != 0 or x_end != 0 or y_end != 0:
                    yolo_detections, yolo_image, yolo_time = run_yolo_on_regions(
                        next_frame, (x_start, y_start, x_end, y_end), yolo_model, imgs[i + 1]
                    )
                else:
                    # No memristor regions activated - save image with notification
                    yolo_detections, yolo_image, yolo_time = [], next_frame.copy(), 0.0
                    cv2.putText(yolo_image, "No memristor regions activated", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # Save the image
                    os.makedirs(YOLO_SAVE_PATH, exist_ok=True)
                    detection_path = os.path.join(YOLO_SAVE_PATH, f"det_region_{imgs[i + 1]}")
                    cv2.imwrite(detection_path, yolo_image)
                    # Save text file
                    txt_path = os.path.join(YOLO_SAVE_PATH, f"det_region_{imgs[i + 1].replace('.jpg', '.txt')}")
                    with open(txt_path, 'w') as f:
                        f.write(f"Region-based Detections for {imgs[i + 1]}:\n")
                        f.write("No memristor regions were activated.\n")
            
            print(f"YOLO detected {len(yolo_detections)} objects in frame {imgs[i + 1]}")
            for det in yolo_detections:
                print(f"  - {det['class_name']}: {det['confidence']:.3f}")
            
            yolo_region_times.append(yolo_time)
        else:
            yolo_detections = []
            yolo_time = 0.0
            yolo_region_times.append(0.0)

        # Run YOLO detection on full image for comparison
        if yolo_model is not None:
            yolo_full_detections, yolo_full_image, yolo_full_time = run_yolo_on_full_image(
                next_frame, yolo_model, imgs[i + 1]
            )
            print(f"Full-image YOLO detected {len(yolo_full_detections)} objects in frame {imgs[i + 1]}")
            for det in yolo_full_detections:
                print(f"  - {det['class_name']}: {det['confidence']:.3f}")
            
            yolo_full_times.append(yolo_full_time)
        else:
            yolo_full_detections = []
            yolo_full_time = 0.0
            yolo_full_times.append(0.0)

        if FLAG == 1:
            motion_binary = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                regions_info, 
                EST_FLAG=FLAG,
                MERGE_FLAG=True  # When EST_FLAG=1, True means combined processing, False means separate processing
            )
        else:
            if x_start == 0 and y_start == 0 and x_end == 0 and y_end == 0:
                num_labels = 1  # No valid area
            else:
                num_labels = 2  # There is a valid area
            # In the case of FLAG == 2, a single area information tuple is passed
            motion_binary = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                (x_start, y_start, x_end, y_end),  # Directly pass the area boundary coordinates
                EST_FLAG=FLAG,
                MERGE_FLAG=True  # This parameter is invalid when FLAG=2
            )
        
        cv2.imwrite(f"{SEG_SAVE_PATH}/{imgs[i + 1]}", motion_binary)
        ####################################################################
        # Farneback
        start_time = time.time()
        # Use cv2.calcOpticalFlowFarneback to calculate dense optical flow
        flow1 = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, **farneback_params)
        end_time = time.time()
        original_opticalflow_times.append(end_time - start_time)

        # Invert optical flow (only for Farneback)
        flow1 = -flow1

        start_time = time.time()
        mag, ang = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
        
        # Map angle to Hue channel
        hsv[..., 0] = ang * 180 / np.pi / 2

        # Normalize the magnitude of the optical flow to the range [0, 255] and store it in the brightness (Value) channel
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image to BGR image
        bgr1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Convert color image to grayscale image
        gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)

        # Use thresholding to convert a grayscale image to a binary image
        threshold = SEG_TH  # threshold, adjust according to the actual situation
        _, binary1 = cv2.threshold(gray1, threshold, 255, cv2.THRESH_BINARY)

        # Create a blank image of the same size as the input image
        motion_mask1 = np.zeros_like(bgr1)

        # Judge the optical flow of each pixel. If the magnitude of the optical flow is greater than the threshold, mark the pixel as a motion area.
        motion_mask1[mag > threshold] = 255

        # Perform morphological operations on the motion area to fill holes or remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        num_iterations = 5  # Adjust the number of iterations according to the actual situation
        for _ in range(num_iterations):
            motion_mask1 = cv2.dilate(motion_mask1, kernel)
            motion_mask1 = cv2.erode(motion_mask1, kernel)

        # Convert the binary image of the motion area to a grayscale image
        motion_gray1 = cv2.cvtColor(motion_mask1, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to obtain the final binary image
        _, motion_binary1 = cv2.threshold(motion_gray1, 1, 255, cv2.THRESH_BINARY)
        end_time = time.time()
        original_task_times.append(end_time - start_time)

        cv2.imwrite(f"{SEG_SAVE_PATH2}/{imgs[i + 1]}", motion_binary1)
        accuracy1 = calculate_pixel_accuracy(motion_binary, binary_image)
        accuracy2 = calculate_pixel_accuracy(motion_binary1, binary_image)
        total_accuracy1 += accuracy1
        total_accuracy2 += accuracy2
        
        # Calculate YOLO timing improvement before printing
        yolo_time_improvement = yolo_full_time - yolo_time
        yolo_time_improvement_percent = (yolo_time_improvement / yolo_full_time * 100) if yolo_full_time > 0 else 0
        
        print("The first, i+1, to, i+2, picture", mem_task_times[0], original_task_times[0], accuracy1, accuracy2)
        print(f"YOLO timing - Region: {yolo_time:.4f}s, Full: {yolo_full_time:.4f}s, "
              f"Improvement: {yolo_time_improvement:.4f}s ({yolo_time_improvement_percent:.2f}%)")
        
        # Compare flow computation time (first value)
        flow_orig_time = original_opticalflow_times[0]
        flow_mem_time = mem_opticalflow_times[0]
        flow_improvement = flow_orig_time - flow_mem_time
        flow_improvement_percent = (flow_improvement / flow_orig_time) * 100
        
        # Still keep writing to the text file
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Flow computation time: Original={flow_orig_time:.4f}s, Mem={flow_mem_time:.4f}s, \n'
                      f' Improvement={flow_improvement:.4f}s ({flow_improvement_percent:.2f}%)\n')

        # CSV data collection
        csv_row = [
            f'{filename2}-{filename1}',  # Frame pair
            f'{flow_orig_time:.4f}',
            f'{flow_mem_time:.4f}',
            f'{flow_improvement:.4f}',
            f'{flow_improvement_percent:.2f}'
        ]        
        
        pred_orig_time = original_task_times[0]
        pred_mem_time = mem_task_times[0]
        com_mem_time = mem_combination_times[0]
        current_raft_ssim = accuracy2
        current_memraft_ssim = accuracy1
        
        # Convert the time list to a semicolon-separated string
        cal_times_str = ';'.join([f'{t:.4f}' for t in region_list])
        cal_times_str = ';'.join([f'{t:.4f}' for t in mem_cal_times])
        vel_times_str = ';'.join([f'{t:.4f}' for t in mem_velocity_times])

        # Continue adding to CSV line
        csv_row.extend([
            f'{pred_orig_time:.4f}',
            f'{pred_mem_time:.4f}',
            f'{com_mem_time:.4f}',
            f'{current_raft_ssim:.4f}',
            f'{current_memraft_ssim:.4f}',
            region_list,           # Add percentage
            cal_times_str,         # Add all calculation time
            vel_times_str,         # Add all speed times
            f'{yolo_time:.4f}',    # Region YOLO processing time
            f'{yolo_full_time:.4f}',  # Full image YOLO processing time
            f'{yolo_time_improvement:.4f}',  # YOLO time improvement
            f'{yolo_time_improvement_percent:.2f}',  # YOLO time improvement percentage
            len(yolo_detections),  # Number of YOLO detections in the region
            len(yolo_full_detections),  # Number of YOLO detections in the full image
            ';'.join([det['class_name'] for det in yolo_detections]) if yolo_detections else '',  # Class name detected in the region
            ';'.join([f"{det['confidence']:.3f}" for det in yolo_detections]) if yolo_detections else '',  # Regional Confidence
            ';'.join([det['class_name'] for det in yolo_full_detections]) if yolo_full_detections else '',  # Class name detected in the full image
            ';'.join([f"{det['confidence']:.3f}" for det in yolo_full_detections]) if yolo_full_detections else ''  # Full image confidence
        ])
        
        # Write to CSV file
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

        # Still keep writing to the text file
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Segmentation time: Original={pred_orig_time:.4f}s, Mem={pred_mem_time:.4f}s, Combination={com_mem_time:.4f}s\n'
                        f'Accuracy: Original={current_raft_ssim:.4f}, Mem={current_memraft_ssim:.4f}\n')
            
            # Add YOLO detection information
            file.write(f'YOLO Processing Times: Region={yolo_time:.4f}s, Full={yolo_full_time:.4f}s, '
                      f'Improvement={yolo_time_improvement:.4f}s ({yolo_time_improvement_percent:.2f}%)\n')
            
            if yolo_detections:
                file.write(f'Region-based YOLO Detections: {len(yolo_detections)} objects found\n')
                for j, det in enumerate(yolo_detections):
                    file.write(f'  Detection {j+1}: {det["class_name"]} (conf: {det["confidence"]:.3f}) '
                              f'at [{det["bbox"][0]:.1f}, {det["bbox"][1]:.1f}, {det["bbox"][2]:.1f}, {det["bbox"][3]:.1f}]\n')
            else:
                file.write('Region-based YOLO Detections: No objects detected\n')
                
            if yolo_full_detections:
                file.write(f'Full-image YOLO Detections: {len(yolo_full_detections)} objects found\n')
                for j, det in enumerate(yolo_full_detections):
                    file.write(f'  Detection {j+1}: {det["class_name"]} (conf: {det["confidence"]:.3f}) '
                              f'at [{det["bbox"][0]:.1f}, {det["bbox"][1]:.1f}, {det["bbox"][2]:.1f}, {det["bbox"][3]:.1f}]\n')
            else:
                file.write('Full-image YOLO Detections: No objects detected\n')
            file.write('\n')

        # f.write(f"{i} {time1} {time2} {accuracy1} {accuracy2}\n")
        # Convert image to color image
        image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        motion_binary_color = cv2.cvtColor(motion_binary, cv2.COLOR_GRAY2BGR)
        motion_binary1_color = cv2.cvtColor(motion_binary1, cv2.COLOR_GRAY2BGR)

        cnt += 1
        # Clear list
        mem_opticalflow_times = []
        mem_cal_times = []
        mem_velocity_times = []

        mem_task_times = []
        mem_combination_times = []

        original_opticalflow_times = []
        original_task_times = []
        
        # Clear YOLO timing lists
        yolo_region_times = []
        yolo_full_times = []

    print(f"our method average accuracy: {total_accuracy1 / cnt}")
    print(f"original method average accuracy: {total_accuracy2 / cnt}")
    with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
        file.write(f'Total average accuracy of our method : {total_accuracy1 / cnt}, Total average accuracy of original farneback : {total_accuracy2 / cnt}\n')