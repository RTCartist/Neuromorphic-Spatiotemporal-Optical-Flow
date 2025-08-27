import argparse
from PIL import Image
from lang_sam import LangSAM
import numpy as np
from lang_sam.utils import draw_image
from lang_sam.utils import load_image
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Generate masks using LangSAM')
    parser.add_argument('--imglist', type=str, default='data/grasp/imgs.txt', 
                        help='Path to the image list file')
    parser.add_argument('--rgbpath', type=str, default='data/grasp/RGB', 
                        help='Path to the RGB images directory')
    parser.add_argument('--savepath', type=str, default='outputs/gtmask', 
                        help='Path to save the generated masks')
    parser.add_argument('--text_prompt', type=str, default='white car.', 
                        help='Text prompt for segmentation')
    
    args = parser.parse_args()
    
    IMGLIST = args.imglist
    RGBPATH = args.rgbpath
    SAVEPATH = args.savepath
    
    model = LangSAM()

    os.makedirs(SAVEPATH, exist_ok=True)

    with open(IMGLIST, 'r') as f:
        imgs = f.read().splitlines()
    rgbimages = [os.path.join(RGBPATH,i) for i in imgs]

    for imgpath in tqdm(rgbimages):
        imgname = imgpath.split('/')[-1]
        image_pil = Image.open(imgpath).convert("RGB")
        text_prompt = args.text_prompt
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        if len(labels) == 0:
            # Create a black image with the same size as image_pil
            black_image = Image.new("RGB", image_pil.size, (0, 0, 0))
            black_image.save(os.path.join(SAVEPATH, imgname))
        else:
            maskimg = np.uint8(masks)*255
            if maskimg.shape[0] == 1:
                maskimg = Image.fromarray(maskimg[0]).convert("L")
            else:
                # Combine multiple masks using logical OR
                combined_mask = np.any(maskimg > 0, axis=0).astype(np.uint8) * 255
                maskimg = Image.fromarray(combined_mask).convert("L")
            maskimg.save(os.path.join(SAVEPATH, imgname))
        print(f"{imgpath} is done")

    print('all ok')

if __name__ == "__main__":
    main()