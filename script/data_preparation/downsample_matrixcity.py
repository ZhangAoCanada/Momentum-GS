import os
import glob
from PIL import Image
import multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.utils as vutils
from argparse import ArgumentParser

def downsample_image(image_path, output_folder):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f'Unable to read image: {image_path}, Error: {e}')
        return
    
    # Fixed resolution of 1600x900
    new_width, new_height = 1600, 900
    downsampled_img = img.resize((new_width, new_height), Image.BICUBIC)
    
    img_name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
    output_path = os.path.join(output_folder, img_name)
    
    downsampled_img_tensor = transforms.ToTensor()(downsampled_img)
    vutils.save_image(downsampled_img_tensor, output_path)
    print(f'Saved downsampled image: {output_path}')

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = glob.glob(os.path.join(input_folder, '*.jpg'), recursive=True) + glob.glob(os.path.join(input_folder, '*.JPG'), recursive=True) + glob.glob(os.path.join(input_folder, '*.png'), recursive=True) + glob.glob(os.path.join(input_folder, '*.PNG'), recursive=True)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(downsample_image, [(file, output_folder) for file in files])

if __name__ == "__main__":
    parser = ArgumentParser(description="Downsample images in the specified scene")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    args = parser.parse_args()
    
    # Process both train and test datasets
    for split in ['train', 'test']:
        input_folder = os.path.join('./', args.scene, split, 'block_all', 'input')
        output_folder = os.path.join('./', args.scene, split, 'block_all', 'input_cached')
        process_images(input_folder, output_folder)
