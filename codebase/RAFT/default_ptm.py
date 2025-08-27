import cv2
import numpy as np
import time
import os
from PIL import Image
import sys
sys.path.append('core')
from utils import flow_viz

def viz(flo, imgname):  
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    #save by name
    img_rgb = flo[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    imgpil = Image.fromarray(img_rgb_uint8)
    imgpil.save(imgname)

def draw_min_rect_rectangle(mask_path1, mask_path2):
    image1 = cv2.imread(mask_path1)
    image2 = cv2.imread(mask_path2)
    image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 256)
    save_path2 = r"C:\Users\Tongming Pu\Desktop\Visual predict article\combine.jpg"
    cv2.imwrite(save_path2, image)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.copy(image)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_path1 = r"C:\Users\Tongming Pu\Desktop\Visual predict article\box.jpg"

    # 保存图像
    cv2.imwrite(save_path1, img)


def opticalFlow(prev_memristor, next_memristor, prev_frame, next_frame, pixel_width, pixel_height):
    farneback_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }
    flow = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 2))
    h, w = next_frame.shape[:2]
    extend_height = 1
    extend_width = 1
    # 创建过度图片
    transition_pic = np.zeros((int(h / pixel_height), int(w / pixel_width)))
    for i in range(0, h, pixel_height):
        for j in range(0, w, pixel_width):
            if abs(prev_memristor[i, j] - 255) >= 1 or abs(next_memristor[i, j] - 255) >= 1:
                transition_pic[int(i / pixel_height), int(j / pixel_width)] = 255

    # 边缘检测与轮廓提取
    # 转换为 uint8 类型
    transition_pic = transition_pic.astype(np.uint8)
    thresh = cv2.Canny(transition_pic, 128, 256)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在每个矩形框内计算光流
    for cnt in contours:
        x, y, a, b = cv2.boundingRect(cnt)
        if x - extend_width < 0:
            # 左上角
            if y - extend_height < 0:
                prev_pixel = prev_frame[0:(y + b + extend_height) * pixel_height,
                             0:(x + a + extend_width) * pixel_width]
                next_pixel = next_frame[0:(y + b + extend_height) * pixel_height,
                             0:(x + a + extend_width) * pixel_width]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[0:(y + b + extend_height) * pixel_height,
                0:(x + a + extend_width) * pixel_width] = current_flow
            # 左下角
            if (y + b + extend_height) * pixel_height >= h:
                prev_pixel = prev_frame[(y - extend_height) * pixel_height:h - 1,
                             0:(x + a + extend_width) * pixel_width]
                next_pixel = next_frame[(y - extend_height) * pixel_height:h - 1,
                             0:(x + a + extend_width) * pixel_width]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[(y - extend_height) * pixel_height:h - 1,
                0:(x + a + extend_width) * pixel_width] = current_flow
            # 左侧
            if y - extend_height >= 0 and (y + b + extend_height) * pixel_height < h:
                prev_pixel = prev_frame[(y - extend_height) * pixel_height:(y + extend_height + b) * pixel_height,
                             0:(x + a + extend_width) * pixel_width]
                next_pixel = next_frame[(y - extend_height) * pixel_height:(y + extend_height + b) * pixel_height,
                             0:(x + a + extend_width) * pixel_width]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[(y - extend_height) * pixel_height:(y + extend_height + b) * pixel_height,
                0:(x + a + extend_width) * pixel_width] = current_flow
        if (x + extend_width + a) * pixel_width >= w:
            # 右上角
            if y - extend_height < 0:
                prev_pixel = prev_frame[0:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:w - 1]
                next_pixel = next_frame[0:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:w - 1]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[0:(y + b + extend_height) * pixel_height,
                (x - extend_width) * pixel_width:w - 1] = current_flow
            # 右下角
            if (y + b + extend_height) * pixel_height >= h:
                prev_pixel = prev_frame[(y - extend_height) * pixel_height:h - 1,
                             (x - extend_height) * pixel_width:w - 1]
                next_pixel = next_frame[(y - extend_height) * pixel_height:h - 1,
                             (x - extend_height) * pixel_width:w - 1]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[(y - extend_height) * pixel_height:h - 1,
                (x - extend_height) * pixel_width:w - 1] = current_flow
            # 右侧
            if y - extend_width > 0 and (y + b + extend_height) * pixel_height < h:
                prev_pixel = prev_frame[(y - extend_height) * pixel_height:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:w - 1]
                next_pixel = next_frame[(y - extend_height) * pixel_height:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:w - 1]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[(y - extend_height) * pixel_height:(y + b + extend_height) * pixel_height,
                (x - extend_width) * pixel_width:w - 1] = current_flow
        if x - extend_width >= 0 and (x + extend_width + a) * pixel_width < w:
            # 下侧
            if (y + extend_height + b) * pixel_height >= h:
                prev_pixel = prev_frame[(y - extend_height) * pixel_height:h - 1,
                             (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width]
                next_pixel = next_frame[(y - extend_height) * pixel_height:h - 1,
                             (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[(y - extend_height) * pixel_height:h - 1,
                (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width] = current_flow
            # 上侧
            if y - extend_height < 0:
                prev_pixel = prev_frame[0:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width]
                next_pixel = next_frame[0:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[0:(y + b + extend_height) * pixel_height,
                (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width] = current_flow
            # 中央
            if y - extend_height >= 0 and (y + b + extend_height) * pixel_height < h:
                prev_pixel = prev_frame[(y - extend_height) * pixel_height:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width]
                next_pixel = next_frame[(y - extend_height) * pixel_height:(y + b + extend_height) * pixel_height,
                             (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width]
                current_flow = cv2.calcOpticalFlowFarneback(prev_pixel, next_pixel, None, **farneback_params)
                flow[(y - extend_height) * pixel_height:(y + b + extend_height) * pixel_height,
                (x - extend_width) * pixel_width:(x + extend_width + a) * pixel_width] = current_flow
    return flow

if __name__ == '__main__':
    # draw_min_rect_rectangle(r"C:\Users\Tongming Pu\Desktop\result\rollling\scalepic\43.jpg",
    #                         r"C:\Users\Tongming Pu\Desktop\result\rollling\scalepic\44.jpg")
    prev_memristor = cv2.imread(r"/ibex/user/zhaol0c/wsbckpt/datasets/memimg2/15.jpg")
    next_memristor = cv2.imread(r"/ibex/user/zhaol0c/wsbckpt/datasets/memimg2/14.jpg")
    # 读取原图像
    prev_frame = cv2.imread(r"/ibex/user/zhaol0c/wsbckpt/datasets/img2/15.jpg")
    next_frame = cv2.imread(r"/ibex/user/zhaol0c/wsbckpt/datasets/img2/14.jpg")

    # 将RGB图像转化为灰度图
    prev_memristor_gray = cv2.cvtColor(prev_memristor, cv2.COLOR_RGB2GRAY)
    next_memristor_gray = cv2.cvtColor(next_memristor, cv2.COLOR_RGB2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

    # 根据忆阻器处理过后图像中一个像素对应原图像中像素个数
    pixel_width = 14
    pixel_height = 14
    h, w = next_frame.shape[:2]
    start_time = time.time()
    flow = opticalFlow(prev_memristor_gray, next_memristor_gray, prev_frame_gray, next_frame_gray, pixel_width,
                       pixel_height)
    end_time = time.time()
    print(end_time - start_time)
    flow = flow.astype(np.float32)
    viz(flow, "flowviz.jpg")

    # 通过光流场进行图像预测
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    # coords: 前面生成的坐标数组，其中包含原始图像中每个像素的位置。flow: 光流向量场，它存储了从时间 t 到 t+1 每个像素位置的运动向量。这个向量描述了相应像素在两帧之间的运动。pixel_map: 通过将光流向量添加到原始坐标上，计算出新的像素位置。这样，每个原始位置 (x, y) 通过加上对应的光流向量 (dx, dy)，得到了新位置 (x+dx, y+dy)。
    pixel_map = coords + flow
    # 然后再映射回图像，完成图像预测
    new_frame = cv2.remap(next_frame_gray, pixel_map[:,:,0], pixel_map[:,:,1], cv2.INTER_LANCZOS4)
    file = "part_predict.jpg"
    cv2.imwrite(file, new_frame)

    start_time = time.time()
    # 不经过忆阻器直接处理图像
    flow1 = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 30, 3, 5, 1.1, 0)

    # 输出程序运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print("程序执行时间为: ", execution_time, "秒")
