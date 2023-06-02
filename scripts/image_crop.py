import cv2
import numpy as np
import argparse, os

WIDTH=150
HEIGHT=150
EPSILON=20

def crop(img_path, mask_path):
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the image based on the bounding box
        cropped_image = image[max(0, y-EPSILON):min(y+h+EPSILON, height), max(0, x-EPSILON):min(x+w+EPSILON, width)]
        
        # Resize the cropped image to the desired size
        resized_image = cv2.resize(cropped_image, (WIDTH, HEIGHT))
        # Save the resized image
        cv2.imwrite('/home/kausik/Documents/BundleTrack2.0/cropped_image.jpg', resized_image)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
  

    # Test on old dataset
    parser.add_argument('--data_dir', type=str, default='/home/kausik/Documents/BundleTrack2.0/Data/contact_nets_reduced')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--model_name', type=str, default='contact_nets_reduced')
    parser.add_argument('--model_dir', type=str, default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/YCB_Video_Dataset/CADmodels/021_bleach_cleanser/textured.obj')
    parser.add_argument('--result_dir', type=str, default='/home/kausik/Documents/BundleTrack2.0/results')
    parser.add_argument('--crop_dir', type=str, default='/home/kausik/Documents/BundleTrack2.0/Data/contact_nets_reduced/cropped')
    args = parser.parse_args()

    crop_color_path = args.crop_dir+"/rgb"
    if not os.path.exists(crop_color_path):
        raise RuntimeError(f"Make sure data_dir={crop_color_path} exists")
    crop_depth_path = args.crop_dir+"/depth"
    if not os.path.exists(crop_depth_path):
        raise RuntimeError(f"Make sure data_dir={crop_depth_path} exists")
    crop_mask_path = args.crop_dir+"/masks"
    if not os.path.exists(crop_mask_path):
        raise RuntimeError(f"Make sure data_dir={crop_mask_path} exists")

    color_path = args.data_dir+"/rgb/0108.png"
    mask_path = args.data_dir+"/masks/0108.png"
    depth_path = args.data_dir+"/depth/0001.png"

    crop(color_path, mask_path)