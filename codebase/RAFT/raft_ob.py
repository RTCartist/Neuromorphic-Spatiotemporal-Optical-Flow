import os
import cv2
import numpy as np
import raft_mem_v0_r
import argparse
import torch
from PIL import Image
from raft import RAFT
import torchvision.transforms as T
import time
from einops import rearrange
from utils.utils import InputPadder
# WSB Modify 新增包
import scipy.io
from numba import njit
import csv

# input data path
DEVICE = 'cuda'
FRAME_PATH_BASE = 'data/grasp/RGB'
MASK_PATH_BASE = 'data/grasp/gtmask'
IMGLIST = 'data/grasp/imgs.txt'
MEMMATPATH = 'data/grasp/constructed_3D_matrix.mat'

# output data path
OUTPUT_DIR = 'output/grasp_ob'
OB_SAVE_PATH1 = os.path.join(OUTPUT_DIR, 'ob')
OB_SAVE_PATH2 = os.path.join(OUTPUT_DIR, 'bbox')
OB_SAVE_PATH3 = os.path.join(OUTPUT_DIR, 'originalbbox')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'raft_track.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_ob.csv')

MEMSIZE=80
OFFSET = 0

EXTEND_HEIGHT_UPPER = 20
EXTEND_HEIGHT_LOWER = 20
EXTEND_WIDTH_LEFT = 20
EXTEND_WIDTH_RIGHT = 20
THRES = 250
CONNECT  = 4
FLAG = 2
PADDING = 20
SEG_TH = 1

THRESHOLD_DRAW=20

mem_opticalflow_times = []
mem_cal_times = []
mem_velocity_times = []

mem_task_times = []
mem_combination_times = []

original_opticalflow_times = []
original_task_times = []

def py_cpu_nms(dets, thresh):
    # coordinates of the bounding box
    x1 = dets[:, 0]  # all rows first column
    y1 = dets[:, 1]  # all rows second column
    x2 = dets[:, 2]  # all rows third column
    y2 = dets[:, 3]  # all rows fourth column
    # calculate the area of the bounding box
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)  # (fourth column - second column + 1) * (third column - first column + 1)
    # confidence score of the bounding box
    scores = dets[:, 4]  # all rows fifth column

    keep = [] 
    index = scores.argsort()[::-1]  

    while index.size > 0:  
        i = index[0]  
        keep.append(i)  
        x11 = np.maximum(x1[i], x1[index[1:]])  
        y11 = np.maximum(y1[i], y1[index[1:]]) 
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  
        h = np.maximum(0, y22 - y11 + 1)  
        overlaps = w * h  

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  
        idx = np.where(ious <= thresh)[0]  
        index = index[idx + 1]  

    return keep  

def get_max_bbox_from_mask(mask_path):
    """
    get the coordinates of the maximum bounding box from the mask image
    
    Args:
        mask_path (str): the path of the mask image
        
    Returns:
        tuple: (x1, y1, x2, y2) the coordinates of the left top and right bottom of the maximum bounding box
                if no bounding box is found, return None
    """
    # read the black and white image
    image = cv2.imread(mask_path)
    image = cv2.resize(image, (int(image.shape[1] /3), int(image.shape[0] /3)))
    if image is None:
        return None
        
    # convert to grayscale and binary
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the maximum bounding box
    max_rect = None
    max_area = 0
    for contour in contours:
        # calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # calculate the area of the bounding rectangle
        area = w * h
        
        # update the maximum bounding rectangle
        if area > max_area:
            max_area = area
            max_rect = (x, y, w, h)
    
    # if the bounding box is found, return the coordinates
    if max_rect is not None:
        x, y, w, h = max_rect
        return (x, y, x + w, y + h)
    
    return None

def create_directories():
    os.makedirs(OB_SAVE_PATH1, exist_ok=True)
    os.makedirs(OB_SAVE_PATH2, exist_ok=True)
    os.makedirs(OB_SAVE_PATH3, exist_ok=True)
    with open(SAVETXTPATH, 'w', encoding='utf-8') as file:
        pass
    # create and initialize the CSV file
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
            'Cal_Times',           
            'Velocity_Times'       
        ])

def preprocess(batch):
    transforms = T.Compose(
        [
            T.Resize(size=(int(batch.size[1] /3), int(batch.size[0] /3))),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ]
    )
    batch = transforms(batch)
    return batch

def load_image(imfile):
    # the shape of the output image: [1, c, h, w]
    img = Image.open(imfile).convert('RGB')
    img = preprocess(img)*255.0
    return img[None].to(DEVICE)

def compute_segmentation_mask(flow, threshold=1.0):
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = (magnitude > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def process_images(frame_path1, frame_path2):
    rgbimg1 = load_image(frame_path1)
    rgbimg2 = load_image(frame_path2)
    return rgbimg1, rgbimg2

def prepare_hsv(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (int(img.shape[1] /3), int(img.shape[0] /3)))
    hsv = np.zeros_like(img)
    hsv[..., 1] = 255
    return hsv

def runraft(model, img1, img2):
    padder = InputPadder(img1.shape)
    image1, image2 = padder.pad(img1, img2)

    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    flow_up_unpad = padder.unpad(flow_up)
    outs = flow_up_unpad[0].permute(1, 2, 0).cpu().detach().numpy()
    return outs

@njit
def update_transition_pic(prev_memristor, transition_pic, thres):
    for i in range(prev_memristor.shape[0]):
        for j in range(prev_memristor.shape[1]):
            if prev_memristor[i, j] >= thres:
                transition_pic[i, j] = 255
    return transition_pic

def process_separate_regions(model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    region_list = []
    regions_info = []
    h, w = rgbimg1.shape[2:]

    for i in range(1, len(stats)):
        x, y, a, b = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # calculate the region boundary
        region_start_time = time.time()
        x_start = max(x * pixel_width - EXTEND_WIDTH_LEFT, 0)
        y_start = max(y * pixel_height - EXTEND_HEIGHT_UPPER, 0)
        x_end = min((x + a) * pixel_width + EXTEND_WIDTH_RIGHT, w)
        y_end = min((y + b) * pixel_height + EXTEND_HEIGHT_LOWER, h)
        
        regions_info.append((x_start, y_start, x_end, y_end))
        prev_region = rgbimg1[:,:,y_start:y_end, x_start:x_end]
        next_region = rgbimg2[:,:,y_start:y_end, x_start:x_end]

        region_end_time = time.time()
        mem_cal_times.append(region_end_time - region_start_time)   

        # calculate the region percentage
        region_pixels = prev_region.shape[2] * prev_region.shape[3]
        total_pixels = rgbimg1.shape[2] * rgbimg1.shape[3]
        region_percentage = (region_pixels / total_pixels) * 100
        if prev_region.shape[2] < 64 or prev_region.shape[3] < 64 or next_region.shape[2] < 64 or next_region.shape[3] < 64:
            start_time = time.time()
            continue
        region_list.append(region_percentage)
        print(f"Region covers {region_percentage:.2f}% of the full image\n"
                f"Memristor process time is {region_end_time - region_start_time}")
        
        # calculate the optical flow
        if prev_region.shape[2] > 0 and prev_region.shape[3] > 0 and next_region.shape[2] > 0 and next_region.shape[3] > 0:
            start_time = time.time()
            current_flow = runraft(model, prev_region, next_region)
            flow[y_start:y_end, x_start:x_end] = current_flow
            end_time = time.time()
            mem_velocity_times.append(end_time - start_time)
            print(f"Separate opticalflow process time is {end_time - start_time}")
        else:
            mem_velocity_times.append(0)
            
    return flow, mem_cal_times, mem_velocity_times, region_list, len(stats), regions_info

def process_merged_region(model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    region_list = []
    h, w = rgbimg1.shape[2:]
    start_time = time.time()

    x_min = min(stats[i, cv2.CC_STAT_LEFT] for i in range(1, len(stats)))
    y_min = min(stats[i, cv2.CC_STAT_TOP] for i in range(1, len(stats)))
    x_max = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in range(1, len(stats)))
    y_max = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, len(stats)))
    
    # calculate the extended boundary
    x_start = max(x_min * pixel_width - EXTEND_WIDTH_LEFT, 0)
    y_start = max(y_min * pixel_height - EXTEND_HEIGHT_UPPER, 0)
    x_end = min(x_max * pixel_width + EXTEND_WIDTH_RIGHT, w)
    y_end = min(y_max * pixel_height + EXTEND_HEIGHT_LOWER, h)
    
    # extract the region
    prev_region = rgbimg1[:,:,y_start:y_end, x_start:x_end]
    next_region = rgbimg2[:,:,y_start:y_end, x_start:x_end]

    end_time = time.time()
    mem_cal_times.append(end_time - start_time)

    # calculate the region percentage
    region_pixels = prev_region.shape[2] * prev_region.shape[3]
    total_pixels = rgbimg1.shape[2] * rgbimg1.shape[3]
    region_percentage = (region_pixels / total_pixels) * 100   
    region_list.append(region_percentage)
    print(f"Merged region covers {region_percentage:.2f}% of the full image\n"
          f"Memristor process time is {end_time - start_time}")
    
    # calculate the optical flow
    if prev_region.shape[2] > 0 and prev_region.shape[3] > 0 and next_region.shape[2] > 0 and next_region.shape[3] > 0:
        start_time = time.time()
        current_flow = runraft(model, prev_region, next_region)
        flow[y_start:y_end, x_start:x_end] = current_flow
        end_time = time.time()
        mem_velocity_times.append(end_time - start_time)
        print(f"Merged opticalflow process time is {end_time - start_time}")
 
    return flow, mem_cal_times, mem_velocity_times, region_list, (x_start, y_start, x_end, y_end)

def opticalFlow3D(model, memimg1, memimg2, rgbimg1, rgbimg2, pixel_width, pixel_height):
    start_time = time.time()
    h, w = rgbimg1.shape[2:]
    flow = np.zeros((h, w, 2))
    
    # create the transition image
    transition_pic = np.zeros((int(h / pixel_height), int(w / pixel_width)))
    transition_pic = update_transition_pic(memimg2, transition_pic, THRES)
    transition_pic = transition_pic.astype(np.uint8)

    # find the connected regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        transition_pic, connectivity=CONNECT)

    # if no region is found, return
    if num_labels == 1:
        end_time = time.time()
        mem_cal_times.append(end_time - start_time)
        mem_opticalflow_times.append(end_time - start_time)
        if FLAG == 1:
            return flow, mem_cal_times, mem_opticalflow_times, [], num_labels, []
        else:
            return flow, mem_cal_times, mem_opticalflow_times, [], (0, 0, 0, 0)

    # if there are regions to be processed, choose the processing method according to FLAG
    if FLAG == 1:
        flow, cal_times, vel_times, region_list, num_labels, regions_info = process_separate_regions(
            model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height)
    else:
        flow, cal_times, vel_times, region_list, regions_info = process_merged_region(
            model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height)
    
    # add the total processing time measurement
    end_time = time.time()
    mem_opticalflow_times.append(end_time - start_time)
    
    # return the result
    if FLAG == 1:
        return flow, cal_times, vel_times, region_list, num_labels, regions_info
    else:
        return flow, cal_times, vel_times, region_list, regions_info

def process_flow_region_tracking(mag, ang, frame, region_coords):
    """
    process the optical flow data of a specific region and extract the target boundary box
    
    Args:
        flow_region: the optical flow data in the region
        frame: the complete frame to be processed
        region_coords: (x_min, y_min, x_max, y_max) the coordinates of the region
        
    Returns:
        tuple: (processed_frame, boxes) the processed frame and the list of detected boundary boxes
    """
    # create the HSV image
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # convert to BGR and grayscale
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    draw = cv2.morphologyEx(draw, cv2.MORPH_CLOSE, kernel)
    
    # threshold processing and contour detection
    _, draw = cv2.threshold(draw, SEG_TH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # process the boundary box
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
    
    # non-maximum suppression
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    keep = py_cpu_nms(boxes, 0.2)
    
    # draw the boundary box and return the result
    final_boxes = []
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx, :4]
        final_boxes.append([x1, y1, x2, y2])
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    
    return frame, final_boxes

def task_results(prev_frame, next_frame, flow, num_labels, regions_info, EST_FLAG, MERGE_FLAG):
    """
    process the optical flow result and execute the target tracking
    
    Args:
        prev_frame: the previous frame
        next_frame: the current frame
        flow: the optical flow data
        num_labels: the number of connected regions
        regions_info: the region information
        EST_FLAG: 1 or 2 represents the region estimation method
        MERGE_FLAG: when EST_FLAG=1, it takes effect True represents merging processing False represents separate processing
    
    Returns:
        tuple: (processed_frame, pred_boxes) the processed frame and the list of predicted boundary boxes
    """
    h, w = prev_frame.shape[2:]
    # breakpoint()
    next_frame_copy = rearrange(next_frame[0], 'c h w -> h w c')
    next_frame_copy = next_frame_copy.cpu().numpy()
    start_time = time.time()
    start_time_combined = time.time()
    
    if num_labels > 1:
        if EST_FLAG == 1:
            if MERGE_FLAG:
                # merging processing mode
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
                # separate processing
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
            # EST_FLAG = 2
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

def get_ob(flow, hsv, x1_max, y1_max, x2_max, y2_max, kernel, next_frame):
    global original_task_times
    # breakpoint()
    # convert the optical flow to polar coordinates
    start_time = time.time()
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert the HSV image to BGR image
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # convert the BGR image to grayscale
    draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # morphological opening
    draw = cv2.morphologyEx(draw, cv2.MORPH_CLOSE, kernel)

    # threshold processing
    _, draw = cv2.threshold(draw, SEG_TH, 255, cv2.THRESH_BINARY)
    # find contours
    contours, _ = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # save the boundary box information of the contours
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h, cv2.contourArea(c)])
    boxes = np.array(boxes)
    end_time = time.time()
    # breakpoint()
    original_task_times.append(end_time - start_time)

    # ensure boxes is a 2D array
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        # return a pure black image and average_iou1=0
        bgr = np.zeros_like(bgr)
        next_frame = np.zeros_like(next_frame)
        average_iou1 = 0
        return bgr, next_frame, average_iou1

    # sort the boundary boxes by the confidence score
    # breakpoint()
    boxes = boxes[boxes[:, 4].argsort()[::-1]]

    # non-maximum suppression
    keep = py_cpu_nms(boxes, 0.2)

    total_iou = 0
    total_weight = 0
    # draw the final boundary box
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx, :4]
        # calculate the coordinates of the intersection region
        x1_intersect = max(x1, x1_max)
        y1_intersect = max(y1, y1_max)
        x2_intersect = min(x2, x2_max)
        y2_intersect = min(y2, y2_max)
        # calculate the area of the intersection region
        intersection_area = max(0, x2_intersect - x1_intersect + 1) * max(0, y2_intersect - y1_intersect + 1)

        # calculate the area of the union region
        box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        max_box_area = (x2_max - x1_max + 1) * (y2_max - y1_max + 1)
        union_area = box_area + max_box_area - intersection_area

        weight = 1

        # calculate IoU
        iou = intersection_area / union_area

        # weighted accumulation of IoU
        total_iou = total_iou + iou * weight

        # accumulate the weight
        total_weight = total_weight + weight

        next_frame = np.ascontiguousarray(next_frame, dtype=np.uint8)
        cv2.rectangle(next_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

    # calculate the weighted average IoU
    average_iou1 = total_iou / total_weight

    return bgr, next_frame, average_iou1

def main(model):
    global mem_opticalflow_times, mem_cal_times, mem_velocity_times, mem_task_times, mem_combination_times, original_task_times
    mem_opticalflow_times = []
    mem_cal_times = []
    mem_velocity_times = []

    mem_task_times = []
    mem_combination_times = []

    original_opticalflow_times = []
    original_task_times = []

    create_directories()
    hsv = prepare_hsv(f"{FRAME_PATH_BASE}/11.jpg")
    iou_our = 0
    iou_original = 0
    average_time1 = 0
    average_time2 = 0
    # create the structural element
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # read from txt file
    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()
    mem_data = scipy.io.loadmat(MEMMATPATH)
    mem_state = mem_data['constructed3DMatrix']

    rgbimages = [os.path.join(FRAME_PATH_BASE,i) for i in imgs]
    rgbimages = sorted(rgbimages, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    cnt=0
    a, b = 0, 0

    for idx in range(len(rgbimages)-2):
        mem_state1 = mem_state[:, :, OFFSET+idx].astype(np.double)
        mem_state2 = mem_state[:, :, OFFSET+idx+1].astype(np.double)
        rgbimfile1 = rgbimages[idx]
        rgbimfile2 = rgbimages[idx+1]
        filename1 = os.path.basename(rgbimfile1)
        filename2 = os.path.basename(rgbimfile2)
        rgbimg1, rgbimg2 = process_images(rgbimfile1, rgbimfile2)
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                file.write(f'Calculation between {filename1} and {filename2}\n')

        with np.errstate(divide='ignore', invalid='ignore'):  
            mem_state1 = - 3366 / np.log10(mem_state1) - 306
        mem_state1 = np.clip(mem_state1, 0, 255)
        # Convert to uint8
        mem_state1 = mem_state1.astype(np.uint8)
        with np.errstate(divide='ignore', invalid='ignore'):  
            mem_state2 = - 3366 / np.log10(mem_state2) - 306
        mem_state2 = np.clip(mem_state2, 0, 255)
        # Convert to uint8
        mem_state2 = mem_state1.astype(np.uint8)
        memimg1 = mem_state1  
        memimg2 = mem_state2  

        prev_frame = rgbimg1
        next_frame = rgbimg2

        pixel_width = MEMSIZE
        pixel_height = MEMSIZE

        pixel_width = int(pixel_width /3)   
        pixel_height=int(pixel_height/3)
        h, w = next_frame.shape[2:]

        if FLAG == 1:
            flow, cal_times, vel_times, region_list, num_labels, regions_info = opticalFlow3D(model, memimg1, memimg2, prev_frame, next_frame, pixel_width, pixel_height)
        else:
            flow, cal_times, vel_times, region_list, (x_start, y_start, x_end, y_end) = opticalFlow3D(model, memimg1, memimg2, prev_frame, next_frame, pixel_width, pixel_height)

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
                num_labels = 1  # no valid region
            else:
                num_labels = 2  # one valid region
            next_frame_processed, pred_boxes = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                (x_start, y_start, x_end, y_end),
                EST_FLAG=FLAG,
                MERGE_FLAG=True
            )

        bbox = get_max_bbox_from_mask(f"{MASK_PATH_BASE}/{imgs[idx+1]}")
        gt_frame = rearrange(next_frame[0], 'c h w -> h w c')
        gt_frame = gt_frame.cpu().numpy()  # create a new copy for drawing the ground truth boundary box

        if bbox is not None:
            x1_max, y1_max, x2_max, y2_max = bbox
            gt_frame = np.ascontiguousarray(gt_frame, dtype=np.uint8)
            cv2.rectangle(gt_frame, (int(x1_max), int(y1_max)), 
                         (int(x2_max), int(y2_max)), (0, 0, 255), 2)  # BGR格式，红色
            
            cv2.putText(gt_frame, 'Ground Truth', (int(x1_max), int(y1_max)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{OB_SAVE_PATH1}/{imgs[idx+1]}", gt_frame)

        # calculate IoU
        if bbox is not None and pred_boxes:
            total_iou = 0
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                # calculate the intersection
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
        # breakpoint()
        next_frame_processed = cv2.cvtColor(next_frame_processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{OB_SAVE_PATH2}/{imgs[idx+1]}", next_frame_processed)

        ################## original #############
        start_time = time.time()
        floworiginal = runraft(model, prev_frame, next_frame)
        end_time = time.time()
        original_opticalflow_times.append(end_time - start_time)

        # calculate iou
        start_time = time.time()
        rgb1forob = rearrange(next_frame[0], 'c h w -> h w c')
        mag, ang = cv2.cartToPolar(floworiginal[..., 0], floworiginal[..., 1])
        next_frame_processed, pred_boxes = process_flow_region_tracking(
            mag, ang, rgb1forob.cpu().numpy(), (0, 0, 0, 0))

        end_time = time.time()
        original_task_times.append(end_time - start_time)

        total_iou = 0
        total_weight = 0
        if bbox is not None and pred_boxes:
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                # calculate the intersection
                x1_intersect = max(x1, x1_max)
                y1_intersect = max(y1, y1_max)
                x2_intersect = min(x2, x2_max)
                y2_intersect = min(y2, y2_max)
                
                intersection_area = max(0, x2_intersect - x1_intersect + 1) * max(0, y2_intersect - y1_intersect + 1)
                box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                max_box_area = (x2_max - x1_max + 1) * (y2_max - y1_max + 1)
                union_area = box_area + max_box_area - intersection_area
                
                # use the area as the weight
                weight = 1
                iou = intersection_area / union_area
                total_iou += iou * weight
                total_weight += weight

            average_iou2 = total_iou / total_weight if total_weight > 0 else 0
        else:
            average_iou2 = 0
            next_frame_processed = np.zeros_like(rgb1forob.cpu().numpy())

        print(f"original average iou is: {average_iou2}")

        next_frame_processed = cv2.cvtColor(next_frame_processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{OB_SAVE_PATH3}/{imgs[idx+1]}", next_frame_processed)

        # breakpoint()
        print("第", idx+1, "到", idx+2, "张图片处理时间", mem_task_times[0], original_task_times[0])
        print("第", idx+1, "到", idx+2, "iou", average_iou, average_iou2)
        # Compare flow computation time (first value)
        flow_orig_time = original_opticalflow_times[0]
        flow_mem_time = mem_opticalflow_times[0]
        flow_improvement = flow_orig_time - flow_mem_time
        flow_improvement_percent = (flow_improvement / flow_orig_time) * 100
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Flow computation time: Original={flow_orig_time:.4f}s, Mem={flow_mem_time:.4f}s, \n'
                        f' Improvement={flow_improvement:.4f}s ({flow_improvement_percent:.2f}%)\n')
        
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

        # convert the time list to a semicolon-separated string
        cal_times_str = ';'.join([f'{t:.4f}' for t in region_list])
        cal_times_str = ';'.join([f'{t:.4f}' for t in mem_cal_times])
        vel_times_str = ';'.join([f'{t:.4f}' for t in mem_velocity_times])

        csv_row.extend([
            f'{pred_orig_time:.4f}',
            f'{pred_mem_time:.4f}',
            f'{com_mem_time:.4f}',
            f'{current_raft_ssim:.4f}',
            f'{current_memraft_ssim:.4f}',
            region_list,           
            cal_times_str,         
            vel_times_str          
        ])

         # write to the CSV file
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
        file.write(f'Total average iou of our method : {a / cnt}, Total average iou of original raft : {b / cnt}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # load the model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    main(model)