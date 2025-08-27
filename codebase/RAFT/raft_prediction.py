import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
# WSB MODIFY 新增加三个包
import scipy.io
import csv
from numba import njit

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import time
import torchvision.transforms as T
from skimage.metrics import structural_similarity
from numpy import mgrid, vstack, int32

# input data path
DEVICE = 'cuda'
RGBPATH = 'data/grasp/RGB'
IMGLIST = 'data/grasp/imgs.txt'
MEMMATPATH = 'data/grasp/constructed_3D_matrix.mat'

# output data path
OUTPUT_DIR = 'output/grasp_prediction'
SAVEPATH= os.path.join(OUTPUT_DIR, 'predictours')
SAVEPATH2= os.path.join(OUTPUT_DIR, 'originalimg1')
VISPATH_MEM = os.path.join(OUTPUT_DIR, 'mem_visflow')
VISPATH_ORI = os.path.join(OUTPUT_DIR, 'ori_visflow')
SAVETXTPATH = os.path.join(OUTPUT_DIR, 'raft_time.txt')
CSV_PATH = os.path.join(OUTPUT_DIR, 'metrics_predict.csv')

# save the flow information, which might too large. omit default
# MEM_FLOW_PATH = os.path.join(OUTPUT_DIR, 'mem_flow.npy')
# ORI_FLOW_PATH = os.path.join(OUTPUT_DIR, 'ori_flow.npy')

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

# calculate the error of the optical flow
def calculateIntegralError(prediction, true):
    # Convert prediction to torch tensor from numpy array
    prediction = torch.from_numpy(np.array(prediction))
    
    true = torch.from_numpy(np.array(true.cpu()))
    true = true.squeeze(0).permute(1, 2, 0)
    ssim = structural_similarity(true.cpu().numpy()[:,:,2], prediction.cpu().numpy()[:,:,2], data_range=255.0)
    return ssim

def prepare_folder():
    os.makedirs(VISPATH_MEM, exist_ok=True)
    os.makedirs(VISPATH_ORI, exist_ok=True)
    os.makedirs(SAVEPATH, exist_ok=True)
    os.makedirs(SAVEPATH2, exist_ok=True)
    with open(SAVETXTPATH, 'w', encoding='utf-8') as file:
        pass

    # create and initialize CSV file
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
            'Cal_Times',           
            'Velocity_Times'      
        ])

def preprocess(batch):
    transforms = T.Compose(
        [
            # T.Resize(size=(int(batch.size[1] /3), int(batch.size[0] /3))),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ]
    )
    batch = transforms(batch)
    return batch

def viz(flo, imgname):  
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    #save by name
    img_rgb = flo[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    imgpil = Image.fromarray(img_rgb_uint8)
    imgpil.save(imgname)

def load_image(imfile):
    img = Image.open(imfile).convert('RGB')
    img = preprocess(img)*255.0        
    return img[None].to(DEVICE)

def runraft(model, img1, img2):
    padder = InputPadder(img1.shape)
    image1, image2 = padder.pad(img1, img2)

    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    flow_up_unpad = padder.unpad(flow_up)
    outs = flow_up_unpad[0].permute(1, 2, 0).cpu().detach().numpy()
    return outs

# Add a new function to compare with the threshold to create a mask for speed estimation
@njit
def update_transition_pic(prev_memristor, transition_pic, thres):
    for i in range(prev_memristor.shape[0]):
        for j in range(prev_memristor.shape[1]):
            if prev_memristor[i, j] >= thres:
                transition_pic[i, j] = 255
    return transition_pic

def process_separate_regions(model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height):
    """Separate processing FLAG=1"""
    region_list = []
    regions_info = []
    h, w = rgbimg1.shape[2:]

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
        prev_region = rgbimg1[:,:,y_start:y_end, x_start:x_end]
        next_region = rgbimg2[:,:,y_start:y_end, x_start:x_end]

        region_end_time = time.time()
        mem_cal_times.append(region_end_time - region_start_time)   

        # Calculate region percentage
        region_pixels = prev_region.shape[2] * prev_region.shape[3]
        total_pixels = rgbimg1.shape[2] * rgbimg1.shape[3]
        region_percentage = (region_pixels / total_pixels) * 100
        if prev_region.shape[2] < 64 or prev_region.shape[3] < 64 or next_region.shape[2] < 64 or next_region.shape[3] < 64:
            start_time = time.time()
            continue
        region_list.append(region_percentage)
        print(f"Region covers {region_percentage:.2f}% of the full image\n"
                f"Memristor process time is {region_end_time - region_start_time}")
        
        # Calculate optical flow
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
    """Merge processing FLAG=2"""
    region_list = []
    h, w = rgbimg1.shape[2:]
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
    prev_region = rgbimg1[:,:,y_start:y_end, x_start:x_end]
    next_region = rgbimg2[:,:,y_start:y_end, x_start:x_end]

    end_time = time.time()
    mem_cal_times.append(end_time - start_time)

    # Calculate region percentage
    region_pixels = prev_region.shape[2] * prev_region.shape[3]
    total_pixels = rgbimg1.shape[2] * rgbimg1.shape[3]
    region_percentage = (region_pixels / total_pixels) * 100   
    region_list.append(region_percentage)
    print(f"Merged region covers {region_percentage:.2f}% of the full image\n"
          f"Memristor process time is {end_time - start_time}")
    
    # Calculate optical flow
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

    # If there are regions to be processed, select the processing method according to FLAG
    if FLAG == 1:
        flow, cal_times, vel_times, region_list, num_labels, regions_info = process_separate_regions(
            model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height)
    else:
        flow, cal_times, vel_times, region_list, regions_info = process_merged_region(
            model, stats, rgbimg1, rgbimg2, flow, pixel_width, pixel_height)
    
    # Add total processing time measurement
    end_time = time.time()
    mem_opticalflow_times.append(end_time - start_time)
    
    # Return results
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
    h, w = prev_frame.shape[2:]
    new_frame_rgb = next_frame.clone().cpu().detach().numpy()[:, [2, 1, 0], :, :]

    if num_labels > 1:
        if EST_FLAG == 1:
            if MERGE_FLAG:
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

                for channel in range(3):
                    remapped = cv2.remap(
                        np.array(next_frame[0,2-channel, :,:].cpu().detach()),
                        flow_map[..., 0],
                        flow_map[..., 1],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    new_frame_rgb[:, channel,y_min:y_max, x_min:x_max] = remapped
            else:
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
                    
                    for channel in range(3):
                        remapped = cv2.remap(
                            np.array(next_frame[0,2-channel, :,:].cpu().detach()),
                            flow_map[..., 0],
                            flow_map[..., 1],
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        new_frame_rgb[:, channel, y_min:y_max, x_min:x_max] = remapped        
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
                    np.array(next_frame[0,2-channel, :,:].cpu().detach()),
                    flow_map[..., 0],
                    flow_map[..., 1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                new_frame_rgb[:, channel, y_min:y_max, x_min:x_max] = remapped
 
    else:
        end_time_combined = time.time()
        mem_combination_times.append(end_time_combined - start_time_combined)
    
    end_time = time.time()
    mem_task_times.append(end_time - start_time)
    
    return new_frame_rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # Load model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # prepare folder
    prepare_folder()

    with torch.no_grad():
        # read from txt file
        with open(IMGLIST, 'r') as f:
            imgs = f.read().splitlines()
        
        # No longer read pictures, but read a .mat file organized by matlab. 
        # In this case, there will be a displacement in the serial number when reading, 
        # because the first 15 in this .mat file are a piece of empty data (equivalent to the initial setting of the memristor)
        mem_data = scipy.io.loadmat(MEMMATPATH)
        mem_state = mem_data['constructed3DMatrix']

        rgbimages = [os.path.join(RGBPATH,i) for i in imgs]
        rgbimages = sorted(rgbimages, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        totaltime1 = 0
        totaltime2 = 0
        memprocess_time = 0
        raftprocess_time = 0
        cnt=0
        ssim1 = 0
        ssim2 = 0
        for idx in range(len(rgbimages)-2):
            mem_state1 = mem_state[:, :, OFFSET+idx].astype(np.double)
            mem_state2 = mem_state[:, :, OFFSET+idx+1].astype(np.double)
            rgbimfile1 = rgbimages[idx]
            rgbimfile2 = rgbimages[idx+1]
            filename1 = os.path.basename(rgbimfile1)
            filename2 = os.path.basename(rgbimfile2)
            with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                file.write(f'Calculation between {filename1} and {filename2}\n')
            
            # Read the matrix information instead of the image, and normalize the values ​​in the matrix between 0-255
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

            rgbimg1 = load_image(rgbimfile1)
            rgbimg2 = load_image(rgbimfile2)
            prev_frame = rgbimg1
            next_frame = rgbimg2

            pixel_width = MEMSIZE
            pixel_height = MEMSIZE
            h, w = rgbimg2.shape[2:]

            if FLAG == 1:
                flow, cal_times, vel_times, region_list, num_labels, regions_info = opticalFlow3D(model, memimg1, memimg2, prev_frame, next_frame, pixel_width, pixel_height)
            else:
                flow, cal_times, vel_times, region_list, (x_start, y_start, x_end, y_end) = opticalFlow3D(model, memimg1, memimg2, prev_frame, next_frame, pixel_width, pixel_height)
            
            # write to txt file
            if len(region_list) == 0:
                with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                    file.write(f"Mem Flow Calculation: \n Region covers 0% of the full image\n")
                    file.write(f" preprocess time is {cal_times[0]:.4f}s\n")
                with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                    file.write(f" raft running time is 0s\n")
            else:
                for mem_t, raft_t, region_percentage in zip(cal_times, vel_times, region_list):
                    with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                        file.write(f"Mem Flow Calculation: \n Region covers {region_percentage:.2f}% of the full image\n")
                        file.write(f" preprocess time is {mem_t:.4f}s\n")
                    with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                        file.write(f" raft running time is {raft_t:.4f}s\n")
            
            cnt+=1
            flow = flow.astype(np.float32)

            # save flow vis map
            viz(flow, os.path.join(VISPATH_MEM, rgbimfile2.split("/")[-1]))
            mem_flow.append(flow)

            # Use in main loop
            # Pass different parameters according to FLAG value
            if FLAG == 1:
                new_frame_rgb = task_results(
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
                    num_labels = 1  
                else:
                    num_labels = 2 
                new_frame_rgb = task_results(
                    prev_frame, 
                    next_frame, 
                    flow, 
                    num_labels, 
                    (x_start, y_start, x_end, y_end),  
                    EST_FLAG=FLAG,
                    MERGE_FLAG=True  
                )

            # After prediction, compare the effect
            if cnt != 0:
                rgbimg3=load_image(rgbimages[idx + 2])
                new_frame_rgb = new_frame_rgb.squeeze(0).transpose(1, 2, 0)
                memraft_ssim = calculateIntegralError(new_frame_rgb, rgbimg3)
                filename3 = os.path.basename(rgbimages[idx + 2])
                ssim1 += memraft_ssim

            predictimgpath = os.path.join(SAVEPATH, rgbimfile2.split("/")[-1])
            cv2.imwrite(predictimgpath, new_frame_rgb)
            
            ############## original raft ###################
            start_time = time.time()
            flow1 = runraft(model, prev_frame, next_frame)
            end_time = time.time()
            flow1 = flow1.astype(np.float32)
            original_opticalflow_times.append(end_time - start_time)

            viz(flow1, os.path.join(VISPATH_ORI, rgbimfile2.split("/")[-1]))
            ori_flow.append(flow1)

            start_time = time.time()
            flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w))) + flow1.reshape(-1, 2)
            flow_map = flow_map.reshape(h, w, 2).astype(np.float32)

            # Apply remap to each channel
            b_remapped = cv2.remap(np.array(next_frame[0, 2,:,:].cpu().detach()), flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)
            g_remapped = cv2.remap(np.array(next_frame[0, 1,:,:].cpu().detach()), flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)
            r_remapped = cv2.remap(np.array(next_frame[0, 0,:,:].cpu().detach()), flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)

            # Merge the channels back together
            new_frame_rgb = cv2.merge([b_remapped, g_remapped, r_remapped])
            end_time = time.time()
            original_task_times.append(end_time - start_time)

            # save the predicted image
            predictimgpath = os.path.join(SAVEPATH2, rgbimfile2.split("/")[-1])
            cv2.imwrite(predictimgpath, new_frame_rgb)

            # calculate ssim
            if cnt != 0:
                rgbimg3=load_image(rgbimages[idx + 2])
                raft_ssim = calculateIntegralError(new_frame_rgb, rgbimg3)
                filename3 = os.path.basename(rgbimages[idx + 2])
                ssim2 += raft_ssim

            # Compare flow computation time and write metrics
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

            # Compare prediction & SSIM time
            if cnt < len(rgbimages) - 1:
                pred_orig_time = original_task_times[0]
                pred_mem_time = mem_task_times[0]
                com_mem_time = mem_combination_times[0]
                current_raft_ssim = raft_ssim
                current_memraft_ssim = memraft_ssim
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
                
                with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
                    file.write(f'Prediction & SSIM time: Original={pred_orig_time:.4f}s, Mem={pred_mem_time:.4f}s, Combination={com_mem_time:.4f}s\n'
                            f'SSIM results: Original={current_raft_ssim:.4f}, Mem={current_memraft_ssim:.4f}\n')
            else:
                csv_row.extend(['NA', 'NA', 'NA', 'NA', 'NA'])

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

        # save the flow
        # np.save(MEM_FLOW_PATH, np.array(mem_flow))
        # np.save(ORI_FLOW_PATH, np.array(ori_flow))  
        print(f'mem raft average ssim is {ssim1/(cnt-1)}, original raft average ssim is {ssim2/(cnt-1)}\n')
        print("Processing complete.")
        with open(SAVETXTPATH, 'a', encoding='utf-8') as file:
            file.write(f'mem raft average ssim is {ssim1/(cnt-1)}, original raft average ssim is {ssim2/(cnt-1)}\n')