import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import time
import torchvision.transforms as T

DEVICE = 'cuda'

def preprocess(batch):
    transforms = T.Compose(
        [
            # T.Resize(size=(240, 320)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ]
    )
    batch = transforms(batch)
    return batch

def load_image(imfile):
    # breakpoint()
    img = Image.open(imfile).convert('RGB')
    img = preprocess(img)*255.0

    # img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = torch.from_numpy(img).permute(2, 0, 1).float()
    
    return img[None].to(DEVICE)


def viz(img, flo, imgname):
    # breakpoint()
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    #save by name
    img_rgb = img_flo[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    imgpil = Image.fromarray(img_rgb_uint8)
    imgpil.save(imgname)

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        totaltime = 0
        cnt=0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            # breakpoint()
            # image1 = torch.zeros_like(image1).to(DEVICE)
            # image2 = torch.zeros_like(image2).to(DEVICE)

            start=time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            end=time.time()
            totaltime+=end - start
            cnt+=1
            print(f"running time is {end-start}")
            tmpname=imfile1.split("/")[-1]
            tmpname=tmpname.split(".")[-2]
            imgname = f"/ibex/user/zhaol0c/wsbckpt/results/tmp/{tmpname}.jpg"
            viz(image1, flow_up,imgname)
        
        print(f"total running time is {totaltime}, and average time is {totaltime/cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
