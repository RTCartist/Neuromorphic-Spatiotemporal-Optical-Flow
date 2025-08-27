import time
import cv2
import numpy as np
import flow_viz
from PIL import Image
import os
import scipy.io
from numba import njit
import csv

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
RGB_PATH = 'data/grasp/RGB'
MASK_PATH = 'data/grasp/gtmask'
IMGLIST = 'data/grasp/imgs.txt'
MEMMATPATH = 'data/grasp/constructed_3D_matrix.mat'

# Output paths
OUTPUT_DIR = 'output/grasp_seg'
SEG_SAVE_PATH = os.path.join(OUTPUT_DIR, 'segimg')
SEG_SAVE_PATH2 = os.path.join(OUTPUT_DIR, 'originalsegimg')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'farneback_seg.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_seg.csv')

MEMSIZE= 80
OFFSET = 0

EXTEND_HEIGHT_UPPER = 20
EXTEND_HEIGHT_LOWER = 20
EXTEND_WIDTH_LEFT = 20
EXTEND_WIDTH_RIGHT = 20
THRES = 250
CONNECT  = 4
# Add a new variable to specify which calculation mode to use when calculating optical flow
FLAG = 2

PADDING = 20
SEG_TH = 1

mem_opticalflow_times = []
mem_cal_times = []
mem_velocity_times = []

mem_task_times = []
mem_combination_times = []

original_opticalflow_times = []
original_task_times = []

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
    for _ in range(5):  # 5 iterations
        motion_mask = cv2.dilate(motion_mask, kernel)
        motion_mask = cv2.erode(motion_mask, kernel)
    
    # Final binarization
    _, motion_binary = cv2.threshold(motion_mask, 1, 255, cv2.THRESH_BINARY)
    
    return motion_binary

def prepare_folder():
    os.makedirs(SEG_SAVE_PATH, exist_ok=True)
    os.makedirs(SEG_SAVE_PATH2, exist_ok=True)
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
            'Velocity_Times'       # Added: all speed times
        ])

def calculate_pixel_accuracy(image1, image2):
    total_pixels = image1.size
    matching_pixels = np.sum(image1 == image2)
    accuracy = (matching_pixels / total_pixels) * 100
    return accuracy

if __name__ == '__main__':
    prepare_folder()
    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()

    # No longer read the picture, but read a .mat file organized by matlab. 
    # In this case, there will be a displacement in the serial number when reading, 
    # because the first 15 in this .mat file are a piece of empty data (equivalent to the initial setting of the memristor)
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
        # Note the serial number relationship. Only mem_state2 is needed for subsequent processing. 
        # The serial number rule is the maximum serial number of the input picture, which is (i+1)+offset
        mem_state1 = mem_state[:, :, OFFSET + i].astype(np.double)
        mem_state2 = mem_state[:, :, OFFSET + i + 1].astype(np.double) # The value actually passed
        rgbimfile1 = rgbimages[i]
        rgbimfile2 = rgbimages[i + 1]
        filename1 = os.path.basename(rgbimfile1)
        filename2 = os.path.basename(rgbimfile2)
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Calculation between {filename1} and {filename2}\n')

        # does not read the image but the matrix information, and normalizes the values ​​in the matrix between 0-255
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
        print("The first, i+1, to, i+2, picture", mem_task_times[0], original_task_times[0], accuracy1, accuracy2)
        # Compare flow computation time (first value)
        # Compare flow computation time and write metrics
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
            vel_times_str          # Add all speed times
        ])
        
        # Write to CSV file
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

        # Still keep writing to the text file
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Segmentation time: Original={pred_orig_time:.4f}s, Mem={pred_mem_time:.4f}s, Combination={com_mem_time:.4f}s\n'
                        f'Accuracy: Original={current_raft_ssim:.4f}, Mem={current_memraft_ssim:.4f}\n')

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

    print(f"our method average accuracy: {total_accuracy1 / cnt}")
    print(f"original method average accuracy: {total_accuracy2 / cnt}")
    with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
        file.write(f'Total average accuracy of our method : {total_accuracy1 / cnt}, Total average accuracy of original farneback : {total_accuracy2 / cnt}\n')