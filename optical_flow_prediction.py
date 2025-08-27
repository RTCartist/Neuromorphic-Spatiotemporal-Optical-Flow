import cv2
import numpy as np
import time
import os
import flow_viz
from PIL import Image
from skimage.metrics import structural_similarity
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
    # imgpil.save(imgname)
    # MODIFY WSB to increase the clarity of saved pictures during visualization
    imgpil.save(imgname, 'PNG', quality=100, dpi=(300, 300))  # Use PNG format, highest quality, 300DPI

# Default data paths (can be overridden by command-line arguments or a config file)
# Note: Replace these with your actual data paths
RGBPATH = 'data/grasp/RGB'
IMGLIST = 'data/grasp/imgs.txt'
MEMMATPATH = 'data/grasp/constructed_3D_matrix.mat'

# Output paths
OUTPUT_DIR = 'output/grasp_predict'
SAVEPATH = os.path.join(OUTPUT_DIR, 'predictours')
SAVEPATH2 = os.path.join(OUTPUT_DIR, 'originalimg1')
VISPATH_MEM = os.path.join(OUTPUT_DIR, 'mem_visflow')
VISPATH_ORI = os.path.join(OUTPUT_DIR, 'ori_visflow')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'time.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_predict.csv')

# MEM_FLOW_PATH = '/ibex/user/zhaol0c/wsbckpt/results/traditional/UAV_new2/mem_flow.npy'
# ORI_FLOW_PATH = '/ibex/user/zhaol0c/wsbckpt/results/traditional/UAV_new2/ori_flow.npy'

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

mem_opticalflow_times = []
mem_cal_times = []
mem_velocity_times = []

mem_task_times = []
mem_combination_times = []

original_opticalflow_times = []
original_task_times = []
# add data that needs to be saved
mem_flow =[]
ori_flow = []

# tabletennis farneback parameters
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

# uav farneback parameters
# farneback_params = {
#         'pyr_scale': 0.6,
#         'levels': 3,
#         'winsize': 3,
#         'iterations': 3,
#         'poly_n': 10,
#         'poly_sigma': 1.05,
#         'flags': 0
#     }

def calculateIntegralError(prediction, true):
    ssim = structural_similarity(true[:,:,2], prediction[:,:,2], data_range=255.0)
    return ssim

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
    new_frame_rgb = next_frame.copy()

    if num_labels > 1:
        if EST_FLAG == 1:
            if MERGE_FLAG:
                # Merge processing mode
                padding = PADDING
                x_min = max(0, min(region[0] for region in regions_info) - padding)
                y_min = max(0, min(region[1] for region in regions_info) - padding)
                x_max = min(w, max(region[2] for region in regions_info) + padding)
                y_max = min(h, max(region[3] for region in regions_info) + padding)
                
                region_h = y_max - y_min
                region_w = x_max - x_min
                
                x_coords = np.tile(np.arange(x_min, x_max), region_h)
                y_coords = np.repeat(np.arange(y_min, y_max), region_w)
                flow_region = flow[y_min:y_max, x_min:x_max]
                
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                
                flow_map = np.column_stack((x_coords, y_coords)) + flow_region.reshape(-1, 2)
                flow_map = flow_map.reshape(region_h, region_w, 2).astype(np.float32)
                
                # Corresponding prediction task function
                for channel in range(3):
                    remapped = cv2.remap(
                        next_frame[:,:,channel],
                        flow_map[..., 0],
                        flow_map[..., 1],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    new_frame_rgb[y_min:y_max, x_min:x_max, channel] = remapped
            else:
                # Separate processing
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                
                for x_min, y_min, x_max, y_max in regions_info:
                    region_h = y_max - y_min
                    region_w = x_max - x_min
                    
                    x_coords = np.tile(np.arange(x_min, x_max), region_h)
                    y_coords = np.repeat(np.arange(y_min, y_max), region_w)
                    flow_region = flow[y_min:y_max, x_min:x_max]
                    
                    flow_map = np.column_stack((x_coords, y_coords)) + flow_region.reshape(-1, 2)
                    flow_map = flow_map.reshape(region_h, region_w, 2).astype(np.float32)
                    
                    # Corresponding prediction task function
                    for channel in range(3):
                        remapped = cv2.remap(
                            next_frame[:,:,channel],
                            flow_map[..., 0],
                            flow_map[..., 1],
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        new_frame_rgb[y_min:y_max, x_min:x_max, channel] = remapped        
        else:  # EST_FLAG == 2
            # Process directly using the merged area
            x_min, y_min, x_max, y_max = regions_info
            region_h = y_max - y_min
            region_w = x_max - x_min
            
            end_time_combined = time.time()
            mem_combination_times.append(end_time_combined - start_time_combined)
            
            x_coords = np.tile(np.arange(x_min, x_max), region_h)
            y_coords = np.repeat(np.arange(y_min, y_max), region_w)
            flow_region = flow[y_min:y_max, x_min:x_max]
            
            flow_map = np.column_stack((x_coords, y_coords)) + flow_region.reshape(-1, 2)
            flow_map = flow_map.reshape(region_h, region_w, 2).astype(np.float32)
            
            # Corresponding prediction task function
            for channel in range(3):
                remapped = cv2.remap(
                    next_frame[:,:,channel],
                    flow_map[..., 0],
                    flow_map[..., 1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                new_frame_rgb[y_min:y_max, x_min:x_max, channel] = remapped
 
    else:
        end_time_combined = time.time()
        mem_combination_times.append(end_time_combined - start_time_combined)
    
    end_time = time.time()
    mem_task_times.append(end_time - start_time)
    
    return new_frame_rgb

def create_combined_mask(image_height, image_width, num_labels, regions_info):
    """
    Create a merged mask based on the detected area information
    
    Args:
        image_height (int): image height
        image_width (int): image width
        num_labels (int): number of detected regions (including background)
        regions_info (list): list of region information, each element is (x_start, y_start, x_end, y_end)
    
    Returns:
        tuple: (mask, (x_min, y_min, x_max, y_max))
            - mask: merged mask image
            - (x_min, y_min, x_max, y_max): boundary coordinates of the merged region
    """
    # Create blank mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # If no region is detected, return an empty mask directly
    if num_labels <= 1:
        return mask, (0, 0, 0, 0)
    
    # Initialize boundary values
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Find the largest bounding rectangle of all regions
    for region in regions_info:
        x_start, y_start, x_end, y_end = region
        x_min = min(x_min, x_start)
        y_min = min(y_min, y_start)
        x_max = max(x_max, x_end)
        y_max = max(y_max, y_end)
    
    # Mark the merged area in the mask
    mask[y_min:y_max, x_min:x_max] = 1
    
    return mask, (x_min, y_min, x_max, y_max)

def prepare_folder():
    os.makedirs(SAVEPATH, exist_ok=True)
    os.makedirs(SAVEPATH2, exist_ok=True)
    os.makedirs(VISPATH_MEM, exist_ok=True)
    os.makedirs(VISPATH_ORI, exist_ok=True)
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
            'Original_Pred_Time',
            'Mem_Pred_Time',
            'Combination_Time',
            'Original_SSIM',
            'Mem_SSIM',
            'Region_Percent',
            'Cal_Times',           # Added: all calculation times
            'Velocity_Times'       # Added: all speed times
        ])

def process_mem_state(state):
    """Process single memristor state"""
    with np.errstate(divide='ignore', invalid='ignore'):
        state = - 3366 / np.log10(state) - 306
    return np.clip(state, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    prepare_folder()
    cnt = 0
    ssim1 = 0
    ssim2 = 0
    
    # Set a picture range and read
    # with open(IMGLIST, 'w') as f:
    #     for i in range(1, 101):
    #         f.write(f'{i}.jpg\n')

    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()   

    # No longer read pictures, but read a .mat file organized by matlab. 
    # In this case, there will be a displacement in the serial number when reading, 
    # because the first 15 in this .mat file are a piece of empty data (equivalent to the initial setting of the memristor)
    mem_data = scipy.io.loadmat(MEMMATPATH)
    mem_state = mem_data['constructed3DMatrix']

    rgbimages = [os.path.join(RGBPATH,i) for i in imgs]
    # for windows '\\' for linux '/'
    rgbimages = sorted(rgbimages, key=lambda x: int(x.split('\\')[-1].split('.')[0]))

    # Note that this is a prediction task, and the idx range is range(len(imgs)-2)
    for idx in range(len(imgs) - 2):
        # Note the serial number relationship. Only mem_state2 is needed for subsequent processing. 
        # The serial number rule is the maximum serial number of the input picture, which is (i+1)+offset
        mem_state1 = mem_state[:, :, OFFSET+ idx].astype(np.double)
        mem_state2 = mem_state[:, :, OFFSET+ idx + 1].astype(np.double)
        rgbimfile1 = rgbimages[idx]
        rgbimfile2 = rgbimages[idx + 1]
        filename1 = os.path.basename(rgbimfile1)
        filename2 = os.path.basename(rgbimfile2)
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Calculation between {filename1} and {filename2}\n')

        # Read the matrix information instead of the image, and normalize the values ​​in the matrix between 0-255
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
        prev_frame = cv2.imread(rgbimages[idx])
        next_frame = cv2.imread(rgbimages[idx + 1])

        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        pixel_width = MEMSIZE
        pixel_height = MEMSIZE
        h, w = next_frame.shape[:2]

        if FLAG == 1:
            flow, cal_times, vel_times, region_list, num_labels, regions_info = opticalFlow3D(memimg1, memimg2, prev_frame_gray, next_frame_gray, pixel_width, pixel_height)
        else:
            flow, cal_times, vel_times, region_list, (x_start, y_start, x_end, y_end) = opticalFlow3D(memimg1, memimg2, prev_frame_gray, next_frame_gray, pixel_width, pixel_height)

        # Invert optical flow (only for farneback)
        flow = - flow

        # write to txt file
        if len(region_list) == 0:
            with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                file.write(f"Mem Flow Calculation: \n Region covers 0% of the full image\n")
                file.write(f" preprocess time is {cal_times[0]:.4f}s\n")
            with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                file.write(f" farneback running time is 0s\n")
        else:
            for mem_t, vel_t, region_percentage in zip(cal_times, vel_times, region_list):
                with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                    file.write(f"Mem Flow Calculation: \n Region covers {region_percentage:.2f}% of the full image\n")
                    file.write(f" preprocess time is {mem_t:.4f}s\n")
                with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                    file.write(f" farneback running time is {vel_t:.4f}s\n")

        flow = flow.astype(np.float32)
        cnt += 1

        # save flow vis map
        # for windows '\\' for linux '/'
        viz(flow, os.path.join(VISPATH_MEM, rgbimfile2.split("\\")[-1]))
        # save optical flow
        mem_flow.append(flow)

        # Use in main loop
        # Pass different parameters according to the FLAG value
        if FLAG == 1:
            new_frame_rgb = task_results(
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
            new_frame_rgb = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels,
                (x_start, y_start, x_end, y_end),  # Directly pass the area boundary coordinates
                EST_FLAG=FLAG,
                MERGE_FLAG=True  # This parameter is invalid when FLAG=2
            )
          
        # calculate ssim (read the future picture, i.e. i+2)
        if cnt != 0:
            # breakpoint()
            rgbimg3=cv2.imread(rgbimages[idx + 2])
            memraft_ssim = calculateIntegralError(new_frame_rgb, rgbimg3)
            filename3 = os.path.basename(rgbimages[idx + 2])
            ssim1 += memraft_ssim
        
        # for windows '\\' for linux '/'
        predictimgpath = os.path.join(SAVEPATH, rgbimfile2.split("\\")[-1])
        cv2.imwrite(predictimgpath, new_frame_rgb)

        ################# original ######################
        start_time = time.time()
        flow1 = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, **farneback_params)
        end_time = time.time()
        original_opticalflow_times.append(end_time - start_time)
        print(f"Direct Optical Flow Calculation Time for Image Pair {idx}: {end_time - start_time} seconds")

        # Invert optical flow (only for farneback)
        flow1 = - flow1
        # save original optical flow
        viz(flow1, os.path.join(VISPATH_ORI, rgbimfile2.split("\\")[-1]))
        ori_flow.append(flow1)

        start_time = time.time()
        flow1 = flow1.astype(np.float32)
        flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w))) + flow1.reshape(-1, 2)
        flow_map = flow_map.reshape(h, w, 2).astype(np.float32)

        b_remapped = cv2.remap(next_frame[:,:,0], flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)
        g_remapped = cv2.remap(next_frame[:,:,1], flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)
        r_remapped = cv2.remap(next_frame[:,:,2], flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)

        # Merge the channels back together
        new_frame_rgb = cv2.merge([b_remapped, g_remapped, r_remapped])
        end_time = time.time()
        original_task_times.append(end_time - start_time)

        # for windows '\\' for linux '/'
        predictimgpath = os.path.join(SAVEPATH2, rgbimfile2.split("\\")[-1])
        cv2.imwrite(predictimgpath, new_frame_rgb)

        # calculate ssim
        # modify variable name
        if cnt != 0:
            rgbimg3=cv2.imread(rgbimages[idx + 2])
            raft_ssim = calculateIntegralError(new_frame_rgb, rgbimg3)
            filename3 = os.path.basename(rgbimages[idx + 2])
            ssim2 += raft_ssim

        # Compare flow computation time and write metrics
        flow_orig_time = original_opticalflow_times[0]
        flow_mem_time = mem_opticalflow_times[0]
        flow_improvement = flow_orig_time - flow_mem_time
        flow_improvement_percent = (flow_improvement / flow_orig_time) * 100
        # flow_improvement_percent = 0
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

        # Compare prediction & SSIM time
        if cnt < len(rgbimages) - 1:
            pred_orig_time = original_task_times[0]
            pred_mem_time = mem_task_times[0]
            com_mem_time = mem_combination_times[0]
            current_raft_ssim = raft_ssim
            current_memraft_ssim = memraft_ssim
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
            
            # Still keep writing to the text file
            with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                file.write(f'Prediction & SSIM time: Original={pred_orig_time:.4f}s, Mem={pred_mem_time:.4f}s, Combination={com_mem_time:.4f}s\n'
                          f'SSIM results: Original={current_raft_ssim:.4f}, Mem={current_memraft_ssim:.4f}\n')
        else:
            # If it is the last frame, fill in NA
            csv_row.extend(['NA', 'NA', 'NA', 'NA', 'NA'])

        # Write to CSV file
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

        # Clear list
        mem_opticalflow_times = []
        mem_cal_times = []
        mem_velocity_times = []

        mem_task_times = []
        mem_combination_times = []

        original_opticalflow_times = []
        original_task_times = []   
    
    # save optical flow
    # np.save(MEM_FLOW_PATH, np.array(mem_flow))
    # np.save(ORI_FLOW_PATH, np.array(ori_flow))
    print(f'mem Farneback average ssim is {ssim1/(cnt-1)}, original Farneback average ssim is {ssim2/(cnt-1)}\n')
    print("Processing complete.")
    with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
        file.write(f'mem Farneback average ssim is {ssim1/(cnt-1)}, original Farneback average ssim is {ssim2/(cnt-1)}\n')