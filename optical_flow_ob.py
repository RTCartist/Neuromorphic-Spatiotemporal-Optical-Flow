import time
import cv2
import numpy as np
import os
# import drawFlow
import scipy.io
from numba import njit
import csv

# data path
# Default data paths (can be overridden by command-line arguments or a config file)
# Note: Replace these with your actual data paths
FRAME_PATH_BASE = 'data/grasp/RGB'
MASK_PATH_BASE = 'data/grasp/gtmask'
IMGLIST = 'data/grasp/imgs.txt'
MEMMATPATH = 'data/grasp/constructed_3D_matrix.mat'

# Output paths
OUTPUT_DIR = 'output/grasp_ob'
OB_SAVE_PATH1 = os.path.join(OUTPUT_DIR, 'bgr')
OB_SAVE_PATH2 = os.path.join(OUTPUT_DIR, 'bbox')
OB_SAVE_PATH3 = os.path.join(OUTPUT_DIR, 'originalbbox')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'farneback_track.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_ob.csv')


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
#         'pyr_scale': 0.6,
#         'levels': 3,
#         'winsize': 4,
#         'iterations': 2,
#         'poly_n': 1,
#         'poly_sigma': 1.05,
#         'flags': 0
#     }

def py_cpu_nms(dets, thresh):
    # coordinates of the bounding box
    x1 = dets[:, 0]  # all rows, first column
    y1 = dets[:, 1]  # all rows, second column
    x2 = dets[:, 2]  # all rows, third column
    y2 = dets[:, 3]  # all rows, fourth column
    # Calculate the area of the bounding box
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)  # (fourth column - second column + 1) * (third column - first column + 1)
    # Execution degree, confidence score of the bounding box
    scores = dets[:, 4]  # all rows, fifth column

    keep = []  # keep

    # Sort by the confidence score of the bounding box. Add [::-1] at the end to mean reverse order. If there is no [::-1], argsort will return from small to large.
    index = scores.argsort()[::-1]  # Sort the fifth column of all rows from largest to smallest and return the index value

    # Iterate bounding boxes
    while index.size > 0:  # 6 > 0,      3 > 0,      2 > 0
        i = index[0]  # every time the first is the biggest, add it directly
        keep.append(i)  # save
        # Calculate the ordinate of the intersection point on the union (IOU)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])  # index[1:] starts from the number with subscript 1 until the end
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # Calculate the intersection area on the union
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h  # Overlap, intersection

        # IoU: The essence of intersection-over-union is to search for local maximums and suppress non-maximum elements. That is, the intersection of two bounding boxes is divided by their union.
        #          Overlap / (Area[i] + Area[Index[1:]] - Overlap)
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # The overlapping part is the intersection, iou = intersection / union
        # print("ious", ious)
        #               ious <= 0.7
        idx = np.where(ious <= thresh)[0]  # determine the threshold
        # print("idx", idx)
        index = index[idx + 1]  # because the subscript starts from 1
    return keep  # return the saved value

def get_max_bbox_from_mask(mask_path):
    """
    Get the coordinates of the largest bounding box from the mask image
    
    Args:
        mask_path (str): The path of the mask image
        
    Returns:
        tuple: (x1, y1, x2, y2) The upper-left and lower-right coordinates of the largest bounding box
               If no bounding box is found, return None
    """
    # Read black and white image
    image = cv2.imread(mask_path)
    if image is None:
        return None
        
    # Convert to grayscale and binarize
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest bounding box
    max_rect = None
    max_area = 0
    for contour in contours:
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the area of the bounding rectangle
        area = w * h
        
        # Update the largest bounding rectangle
        if area > max_area:
            max_area = area
            max_rect = (x, y, w, h)
    
    # If a bounding box is found, return the coordinates
    if max_rect is not None:
        x, y, w, h = max_rect
        return (x, y, x + w, y + h)
    
    return None

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

def process_flow_region_tracking(mag, ang, frame, region_coords):
    """
    Process the optical flow data of a specific region and extract the target bounding box
    
    Args:
        flow_region: optical flow data in the region
        frame: the complete frame to be processed
        region_coords: (x_min, y_min, x_max, y_max) region coordinates
        
    Returns:
        tuple: (processed_frame, boxes) processed frame and list of detected bounding boxes
    """   
    # Create HSV image
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to BGR and grayscale
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    draw = cv2.morphologyEx(draw, cv2.MORPH_CLOSE, kernel)
    
    # Thresholding and contour detection
    _, draw = cv2.threshold(draw, SEG_TH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process bounding boxes
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x + region_coords[0], y + region_coords[1], 
                     x + w + region_coords[0], y + h + region_coords[1], 
                     cv2.contourArea(c)])
    
    boxes = np.array(boxes)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return frame, []
    
    # Non-maximum suppression
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    # orginal RFAT&GMflow
    # keep = py_cpu_nms(boxes, 0.2)
    keep = py_cpu_nms(boxes, 0.2)

    # Draw bounding boxes and return results
    final_boxes = []
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx, :4]
        final_boxes.append([x1, y1, x2, y2])
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    
    return frame, final_boxes

def task_results(prev_frame, next_frame, flow, num_labels, regions_info, EST_FLAG, MERGE_FLAG):
    """
    Process optical flow results and perform object tracking
    
    Args:
        prev_frame: previous frame
        next_frame: current frame
        flow: optical flow data
        num_labels: number of connected components
        regions_info: region information
        EST_FLAG: 1 or 2 indicates the region estimation method
        MERGE_FLAG: Effective when EST_FLAG=1, True means merged processing, False means separate processing
    
    Returns:
        tuple: (processed_frame, pred_boxes) processed frame and list of predicted bounding boxes
    """
    start_time = time.time()
    start_time_combined = time.time()
    h, w = prev_frame.shape[:2]
    next_frame_copy = next_frame.copy()
    
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
                
                flow_region = flow[y_min:y_max, x_min:x_max]
                mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                next_frame_copy, pred_boxes = process_flow_region_tracking(
                    mag, ang, next_frame_copy, (x_min, y_min, x_max, y_max))
            else:
                # Separate processing
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                all_boxes = []
                
                for x_min, y_min, x_max, y_max in regions_info:
                    flow_region = flow[y_min:y_max, x_min:x_max]
                    mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                    next_frame_copy, region_boxes = process_flow_region_tracking(
                        mag, ang, next_frame_copy, (x_min, y_min, x_max, y_max))
                    all_boxes.extend(region_boxes)
                pred_boxes = all_boxes
        else:
            # EST_FLAG == 2
            x_min, y_min, x_max, y_max = regions_info
            
            end_time_combined = time.time()
            mem_combination_times.append(end_time_combined - start_time_combined)
            
            flow_region = flow[y_min:y_max, x_min:x_max]
            mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
            next_frame_copy, pred_boxes = process_flow_region_tracking(
                mag, ang, next_frame_copy, (x_min, y_min, x_max, y_max))
    else:
        end_time_combined = time.time()
        mem_combination_times.append(end_time_combined - start_time_combined)
        pred_boxes = []
    
    end_time = time.time()
    mem_task_times.append(end_time - start_time)
    
    return next_frame_copy, pred_boxes

def create_directories():
    os.makedirs(OB_SAVE_PATH1, exist_ok=True)
    os.makedirs(OB_SAVE_PATH2, exist_ok=True)
    os.makedirs(OB_SAVE_PATH3, exist_ok=True)
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
            'Original_OB_Time',
            'Mem_OB_Time',
            'Combination_Time',
            'Original_IoU',
            'Mem_IoU',
            'Region_Percent',
            'Cal_Times',           # Added: all calculation times
            'Velocity_Times'       # Added: all speed times
        ])

if __name__ == '__main__':
    create_directories()
    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()

    # no longer reads pictures, but reads a .mat file organized by matlab. 
    # In this case, there will be a displacement in the serial number when reading, 
    # because the first 15 in this .mat file are a piece of empty data (equivalent to the initial setting of the memristor)
    mem_data = scipy.io.loadmat(MEMMATPATH)
    mem_state = mem_data['constructed3DMatrix']

    rgbimages = [os.path.join(FRAME_PATH_BASE,i) for i in imgs]
    rgbimages = sorted(rgbimages, key=lambda x: int(x.split('\\')[-1].split('.')[0]))

    a, b = 0, 0
    cnt = 0
    agtime1 = 0
    agtime2 = 0
    
    for i in range(len(imgs) - 2):
        # Note the serial number relationship. Only mem_state2 is needed for subsequent processing. 
        # The serial number rule is the maximum serial number of the input picture, which is (i+1)+offset
        mem_state1 = mem_state[:, :, OFFSET+i].astype(np.double)
        mem_state2 = mem_state[:, :, OFFSET+i+1].astype(np.double) # The value actually passed
        rgbimfile1 = rgbimages[i]
        rgbimfile2 = rgbimages[i+1]
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
        next_frame = cv2.imread(rgbimages[i+1])

        # Convert RGB image to grayscale
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        # According to the number of pixels in the original image corresponding to one pixel in the image processed by the memristor
        pixel_width = MEMSIZE  # Note modification
        pixel_height = MEMSIZE  # Note modification
        h, w = next_frame.shape[:2]

        if FLAG == 1:
            flow, cal_times, vel_times, region_list, num_labels, regions_info = opticalFlow3D(memimg1, memimg2, prev_frame_gray, next_frame_gray, pixel_width, pixel_height)
        else:
            flow, cal_times, vel_times, region_list, (x_start, y_start, x_end, y_end) = opticalFlow3D(memimg1, memimg2, prev_frame_gray, next_frame_gray, pixel_width, pixel_height)
        
        # Invert the optical flow direction (only Farneback needs to be inverted)
        flow = -flow  # Invert the optical flow in both x and y directions at the same time

        # Process results
        if FLAG == 1:
            next_frame_processed, pred_boxes = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                regions_info, 
                EST_FLAG=FLAG,
                MERGE_FLAG=True
            )
        else:
            if x_start == 0 and y_start == 0 and x_end == 0 and y_end == 0:
                num_labels = 1  # No valid area
            else:
                num_labels = 2  # There is a valid area
            next_frame_processed, pred_boxes = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                (x_start, y_start, x_end, y_end),
                EST_FLAG=FLAG,
                MERGE_FLAG=True
            )

        # Get the largest bounding box. Note the serial number relationship. This should point to the next frame.
        bbox = get_max_bbox_from_mask(f"{MASK_PATH_BASE}/{imgs[i+1]}")
        gt_frame = next_frame.copy()  # Create a new copy to draw the real bounding box

        if bbox is not None:
            x1_max, y1_max, x2_max, y2_max = bbox
            # Draw the real bounding box, using a different color (such as red) to distinguish it from the prediction box
            gt_frame = np.ascontiguousarray(gt_frame, dtype=np.uint8)
            cv2.rectangle(gt_frame, (int(x1_max), int(y1_max)), 
                         (int(x2_max), int(y2_max)), (0, 0, 255), 2)  # BGR format, red
            
            # Add label text
            cv2.putText(gt_frame, 'Ground Truth', (int(x1_max), int(y1_max)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Save the image with the real bounding box
        # breakpoint()
        cv2.imwrite(f"{OB_SAVE_PATH1}/{imgs[i+1]}", gt_frame)

        # Calculate IoU
        if bbox is not None and pred_boxes:
            total_iou = 0
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                # Calculate intersection
                x1_intersect = max(x1, x1_max)
                y1_intersect = max(y1, y1_max)
                x2_intersect = min(x2, x2_max)
                y2_intersect = min(y2, y2_max)
                
                intersection_area = max(0, x2_intersect - x1_intersect + 1) * max(0, y2_intersect - y1_intersect + 1)
                box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                max_box_area = (x2_max - x1_max + 1) * (y2_max - y1_max + 1)
                union_area = box_area + max_box_area - intersection_area
                
                iou = intersection_area / union_area
                total_iou += iou
            
            average_iou = total_iou / len(pred_boxes)
        else:
            average_iou = 0
        
        cv2.imwrite(f"{OB_SAVE_PATH2}/{imgs[i+1]}", next_frame_processed)
        ############################################################################################
        # Farneback
        start_time = time.time()
        # Use cv2.calcOpticalFlowFarneback to calculate dense optical flow
        flow1 = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, **farneback_params)
        end_time = time.time()
        time2 = end_time - start_time
        original_opticalflow_times.append(end_time - start_time)

        # Invert the optical flow direction (only Farneback needs to be inverted)
        flow1 = -flow1  # Invert the optical flow in both x and y directions at the same time

        start_time = time.time()
        next_frame_copy = next_frame.copy()
        
        mag, ang = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
        next_frame_processed, pred_boxes = process_flow_region_tracking(
            mag, ang, next_frame_copy, (0, 0, 0, 0))

        total_iou = 0
        total_weight = 0
        if bbox is not None and pred_boxes:
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                # Calculate intersection
                x1_intersect = max(x1, x1_max)
                y1_intersect = max(y1, y1_max)
                x2_intersect = min(x2, x2_max)
                y2_intersect = min(y2, y2_max)
                
                intersection_area = max(0, x2_intersect - x1_intersect + 1) * max(0, y2_intersect - y1_intersect + 1)
                box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                max_box_area = (x2_max - x1_max + 1) * (y2_max - y1_max + 1)
                union_area = box_area + max_box_area - intersection_area
                
                # Use area as weight
                weight = 1
                iou = intersection_area / union_area
                total_iou += iou * weight
                total_weight += weight

            average_iou2 = total_iou / total_weight if total_weight > 0 else 0
        else:
            average_iou2 = 0
            next_frame_processed = np.zeros_like(next_frame)

        end_time = time.time()
        original_task_times.append(end_time - start_time)

        cv2.imwrite(f"{OB_SAVE_PATH3}/{imgs[i+1]}", next_frame_processed)

        print("The processing time of the i+1 to i+2 pictures", mem_task_times[0], original_task_times[0])
        print("i+1 to i+2 iou", average_iou, average_iou2)
        # Compare flow computation time (first value)
        flow_orig_time = original_opticalflow_times[0]
        flow_mem_time = mem_opticalflow_times[0]
        flow_improvement = flow_orig_time - flow_mem_time
        flow_improvement_percent = (flow_improvement / flow_orig_time) * 100
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
        current_raft_ssim = average_iou2
        current_memraft_ssim = average_iou

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
        
        mem_opticalflow_times = []
        mem_cal_times = []
        mem_velocity_times = []

        mem_task_times = []
        mem_combination_times = []

        original_opticalflow_times = []
        original_task_times = []  

        a = a + average_iou
        b = b + average_iou2
        cnt += 1

    with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
        file.write(f'Total average iou of our method : {a / cnt}, Total average iou of original farneback : {b / cnt}\n')