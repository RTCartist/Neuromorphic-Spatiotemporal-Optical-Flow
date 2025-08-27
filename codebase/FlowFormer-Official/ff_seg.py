import sys
sys.path.append('core')

import time
import cv2
import numpy as np
import os
import argparse
import torch
from PIL import Image
from raft import RAFT
import torchvision.transforms as T
from utils.utils import InputPadder
from core.FlowFormer import build_flowformer
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
# wsb 新增加两个包
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
OUTPUT_DIR = 'output/grasp_seg'
SEG_SAVE_PATH = os.path.join(OUTPUT_DIR, 'segimg')
SEG_SAVE_PATH2 = os.path.join(OUTPUT_DIR, 'originalsegimg')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'raft_seg.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_seg.csv')
MEMSIZE=80
OFFSET = 0

MEMSIZE=80
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

def calculate_pixel_accuracy(image1, image2):
    total_pixels = image1.size
    matching_pixels = np.sum(image1 == image2)
    accuracy = (matching_pixels / total_pixels) * 100
    return accuracy


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

def runflowformer(model, img1, img2):
    try:
        with torch.no_grad():
            padder = InputPadder(img1.shape)
            image1, image2 = padder.pad(img1, img2)
            
            # add sync point
            torch.cuda.synchronize()
            flowpre = model(image1, image2)
            
            # add sync point
            torch.cuda.synchronize()
            
            flow_up_unpad = padder.unpad(flowpre[0])
            outs = flow_up_unpad[0].permute(1, 2, 0).cpu().detach().numpy()
            
            # clear and sync
            del flowpre, flow_up_unpad, image1, image2
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            return outs
    except Exception as e:
        print(f"Error in runflowformer: {e}")
        torch.cuda.empty_cache()
        raise e

def prepare_hsv(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (int(img.shape[1] /3), int(img.shape[0] /3)))
    hsv = np.zeros_like(img)
    hsv[..., 1] = 255
    return hsv

def process_images(frame_path1, frame_path2):
    rgbimg1 = load_image(frame_path1)
    rgbimg2 = load_image(frame_path2)
    return rgbimg1, rgbimg2

@njit
def update_transition_pic(prev_memristor, transition_pic, thres):
    for i in range(prev_memristor.shape[0]):
        for j in range(prev_memristor.shape[1]):
            if prev_memristor[i, j] >= thres:
                transition_pic[i, j] = 255
    return transition_pic

# separate processing FLAG=1
def process_separate_regions(model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    """separate processing FLAG=1"""
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
            current_flow = runflowformer(model, prev_region, next_region)
            flow[y_start:y_end, x_start:x_end] = current_flow
            end_time = time.time()
            mem_velocity_times.append(end_time - start_time)
            print(f"Separate opticalflow process time is {end_time - start_time}")
        else:
            mem_velocity_times.append(0)
            
    return flow, mem_cal_times, mem_velocity_times, region_list, len(stats), regions_info


# merged processing FLAG=2
def process_merged_region(model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    """merged processing FLAG=2"""
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
        current_flow = runflowformer(model, prev_region, next_region)
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

    # find the connected region
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

    # there is a region to be processed, select the processing method according to FLAG
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

def task_results(prev_frame, next_frame, flow, num_labels, regions_info, EST_FLAG, MERGE_FLAG):
    """
    process the optical flow result
    FLAG: 1 or 2 represents the region estimation method
    MERGE_FLAG: when FLAG=1, True represents merged processing, False represents separate processing
    """
    start_time = time.time()
    start_time_combined = time.time()
    h, w = prev_frame.shape[2:]
    motion_binary = np.zeros((h, w), dtype=np.uint8)

    # create the HSV image for the optical flow visualization
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    if num_labels > 1:
        if EST_FLAG == 1:
            if MERGE_FLAG:
                # merged processing mode
                padding = PADDING
                x_min = max(0, min(region[0] for region in regions_info) - padding)
                y_min = max(0, min(region[1] for region in regions_info) - padding)
                x_max = min(w, max(region[2] for region in regions_info) + padding)
                y_max = min(h, max(region[3] for region in regions_info) + padding)
                
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                
                # process the selected region's optical flow
                flow_region = flow[y_min:y_max, x_min:x_max]
                mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                
                # process the selected region
                motion_binary[y_min:y_max, x_min:x_max] = process_flow_region(mag, ang)
            else:
                # separate processing
                end_time_combined = time.time()
                mem_combination_times.append(end_time_combined - start_time_combined)
                
                for x_min, y_min, x_max, y_max in regions_info:
                    flow_region = flow[y_min:y_max, x_min:x_max]
                    mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                    
                    # process the current region
                    motion_binary[y_min:y_max, x_min:x_max] = process_flow_region(mag, ang)

        else:  # EST_FLAG == 2
            # directly use the merged region processing
            x_min, y_min, x_max, y_max = regions_info
            
            end_time_combined = time.time()
            mem_combination_times.append(end_time_combined - start_time_combined)
            
            flow_region = flow[y_min:y_max, x_min:x_max]
            mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
            
            # process the selected region
            motion_binary[y_min:y_max, x_min:x_max] = process_flow_region(mag, ang)
 
    else:
        end_time_combined = time.time()
        mem_combination_times.append(end_time_combined - start_time_combined)
    
    end_time = time.time()
    mem_task_times.append(end_time - start_time)
    
    return motion_binary

def process_flow_region(mag, ang):
    """
    process the optical flow data of the specific region and generate the motion segmentation mask
    """
    # create the HSV image of the specific region
    region_hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    region_hsv[..., 1] = 255
    
    # map the angle to the hue channel
    region_hsv[..., 0] = ang * 180 / np.pi / 2
    
    # normalize the flow magnitude to the brightness channel
    region_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # convert to BGR and then to grayscale
    bgr = cv2.cvtColor(region_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # thresholding
    threshold = SEG_TH
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # create the motion mask
    motion_mask = np.zeros_like(gray)
    motion_mask[mag > threshold] = 255
    
    # morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    for _ in range(5):  # 5 iterations
        motion_mask = cv2.dilate(motion_mask, kernel)
        motion_mask = cv2.erode(motion_mask, kernel)
    
    # final binary
    _, motion_binary = cv2.threshold(motion_mask, 1, 255, cv2.THRESH_BINARY)
    
    return motion_binary

def calculate_seg(flow, hsv):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # map the angle to the hue channel
    hsv[..., 0] = ang * 180 / np.pi / 2

    # normalize the flow magnitude to the brightness channel
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert the HSV image to the BGR image
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # convert the color image to the grayscale image
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # use thresholding to convert the grayscale image to the binary image
    threshold = SEG_TH  # threshold, adjust according to实际情况
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # create a blank image with the same size as the input image
    motion_mask = np.zeros_like(bgr)

    # for each pixel, judge the optical flow, if the magnitude of the flow is greater than the threshold, mark the pixel as the motion region
    motion_mask[mag > threshold] = 255

    # morphological operation on the motion region, to fill the holes or remove the noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    num_iterations = 5  # adjust the number of iterations according to实际情况
    for _ in range(num_iterations):
        motion_mask = cv2.dilate(motion_mask, kernel)
        motion_mask = cv2.erode(motion_mask, kernel)

    # convert the binary image of the motion region to the grayscale image
    motion_gray = cv2.cvtColor(motion_mask, cv2.COLOR_BGR2GRAY)

    # thresholding the grayscale image, get the final binary image
    _, motion_binary = cv2.threshold(motion_gray, 1, 255, cv2.THRESH_BINARY)

    return motion_binary

def prepare_folder():
    os.makedirs(SEG_SAVE_PATH, exist_ok=True)
    os.makedirs(SEG_SAVE_PATH2, exist_ok=True)
    with open(SAVETXTPATH, 'w', encoding='utf-8') as file:
        pass

    # 创建并初始化 CSV 文件
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
            'Cal_Times',            
            'Velocity_Times'        
        ])

def main(model):
    global original_opticalflow_times, original_task_times, mem_task_times, mem_opticalflow_times, mem_combination_times, mem_cal_times, mem_velocity_times
    mem_opticalflow_times = []
    mem_cal_times = []
    mem_velocity_times = []
    mem_task_times = []
    mem_combination_times = []
    original_opticalflow_times = []
    original_task_times = []

    prepare_folder()
    hsv = prepare_hsv(f"{FRAME_PATH_BASE}/1.jpg")
    total_accuracy1 = 0
    total_accuracy2 = 0
    average_time1 = 0
    average_time2 = 0
    # breakpoint()

     # read from txt file
    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()
    mem_data = scipy.io.loadmat(MEMMATPATH)
    mem_state = mem_data['constructed3DMatrix']

    rgbimages = [os.path.join(FRAME_PATH_BASE,i) for i in imgs]
    rgbimages = sorted(rgbimages, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    cnt=0

    for idx in range(len(rgbimages)-2):
        mem_state1 = mem_state[:, :, OFFSET+idx].astype(np.double)
        mem_state2 = mem_state[:, :, OFFSET+idx+1].astype(np.double)
        rgbimfile1 = rgbimages[idx]
        rgbimfile2 = rgbimages[idx+1]
        filename1 = os.path.basename(rgbimfile1)
        filename2 = os.path.basename(rgbimfile2)
        rgbimg1, rgbimg2 = process_images(rgbimfile1, rgbimfile2)

        # breakpoint()
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                file.write(f'Calculation between {filename1} and {filename2}\n')

        # not read the image but the matrix information, and normalize the matrix value to 0-255
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

        # assign the rgbimg1 and rgbimg2 to prev_frame and next_frame
        prev_frame = rgbimg1
        next_frame = rgbimg2

        # set the pixel size parameter
        pixel_width = MEMSIZE
        pixel_height = MEMSIZE
        h, w = next_frame.shape[2:]

         # calculate the optical flow(memristor-accelerated)
        if FLAG == 1:
            flow, cal_times, vel_times, region_list, num_labels, regions_info = opticalFlow3D(model, memimg1, memimg2, prev_frame, next_frame, pixel_width, pixel_height)
        else:
            flow, cal_times, vel_times, region_list, (x_start, y_start, x_end, y_end) = opticalFlow3D(model, memimg1, memimg2, prev_frame, next_frame, pixel_width, pixel_height)

        # task processing result
        if FLAG == 1:
            motion_binary = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                regions_info, 
                EST_FLAG=FLAG,
                MERGE_FLAG=True  # when EST_FLAG=1, True represents merged processing, False represents separate processing
            )
        else:
            # prevent error
            if x_start == 0 and y_start == 0 and x_end == 0 and y_end == 0:
                num_labels = 1  # no valid region
            else:
                num_labels = 2  # one valid region
            # when FLAG == 2, pass the single region information tuple
            motion_binary = task_results(
                prev_frame, 
                next_frame, 
                flow, 
                num_labels, 
                (x_start, y_start, x_end, y_end),  # 直接传递区域边界坐标
                EST_FLAG=FLAG,
                MERGE_FLAG=True  # FLAG=2时此参数无效
            )

        cv2.imwrite(f"{SEG_SAVE_PATH}/{imgs[idx + 1]}", motion_binary)

        image = cv2.imread(f"{MASK_PATH_BASE}/{imgs[idx  + 1]}")
        image = cv2.resize(image, (int(image.shape[1] /3), int(image.shape[0] /3)))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 256, cv2.THRESH_BINARY)

        # original flowformer
        start_time = time.time()
        floworiginal = runflowformer(model, rgbimg1, rgbimg2)
        end_time = time.time()
        original_opticalflow_times.append(end_time - start_time)

        # seg task
        start_time = time.time()
        motion_binaryoriginal = calculate_seg(floworiginal, hsv)
        end_time = time.time()       
        original_task_times.append(end_time - start_time)

        cv2.imwrite(f"{SEG_SAVE_PATH2}/{imgs[idx + 1]}", motion_binaryoriginal)
        accuracy1 = calculate_pixel_accuracy(motion_binary, binary_image)
        total_accuracy1 += accuracy1
        accuracy2 = calculate_pixel_accuracy(motion_binaryoriginal, binary_image)
        total_accuracy2 += accuracy2

        print(f"Frame {idx+1} to {idx+2}, mem task time: {mem_task_times[0]}, original task time: {original_task_times[0]}, accuracy1: {accuracy1}, accuracy2: {accuracy2}")

        flow_orig_time = original_opticalflow_times[0]
        flow_mem_time = mem_opticalflow_times[0]
        flow_improvement = flow_orig_time - flow_mem_time
        flow_improvement_percent = (flow_improvement / flow_orig_time) * 100

        # still keep the text file writing
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Flow computation time: Original={flow_orig_time:.4f}s, Mem={flow_mem_time:.4f}s, \n'
                      f' Improvement={flow_improvement:.4f}s ({flow_improvement_percent:.2f}%)\n')

        # collect the CSV data
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
        
        # convert the time list to a semicolon-separated string
        cal_times_str = ';'.join([f'{t:.4f}' for t in region_list])
        cal_times_str = ';'.join([f'{t:.4f}' for t in mem_cal_times])
        vel_times_str = ';'.join([f'{t:.4f}' for t in mem_velocity_times])

        # continue to add to the CSV row
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
        
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'Segmentation time: Original={pred_orig_time:.4f}s, Mem={pred_mem_time:.4f}s, Combination={com_mem_time:.4f}s\n'
                        f'Accuracy: Original={current_raft_ssim:.4f}, Mem={current_memraft_ssim:.4f}\n')

        # f.write(f"{i} {time1} {time2} {accuracy1} {accuracy2}\n")
        # convert the image to the color image
        image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        motion_binary_color = cv2.cvtColor(motion_binary, cv2.COLOR_GRAY2BGR)
        motion_binaryoriginal_color = cv2.cvtColor(motion_binaryoriginal, cv2.COLOR_GRAY2BGR)

        cnt += 1
        # clear the list
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
        file.write(f'Total average accuracy of our method : {total_accuracy1 / cnt}, Total average accuracy of original raft : {total_accuracy2 / cnt}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # cfg = get_cfg()
    if args.small:
        cfg = get_small_things_cfg()
    else:
        cfg = get_things_cfg()
    cfg.update(vars(args))
    # load the model
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    print(args)
    model.to(DEVICE)
    model.eval()
    main(model)