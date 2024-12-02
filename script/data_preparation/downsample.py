import os
import glob
from PIL import Image
import multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.utils as vutils
from argparse import ArgumentParser

def downsample_image(image_path, output_folder, scale_factor=4):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f'Unable to read image: {image_path}, Error: {e}')
        return
    
    width, height = img.size
    new_width, new_height = width // scale_factor, height // scale_factor
    downsampled_img = img.resize((new_width, new_height), Image.BICUBIC)
    
    img_name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
    output_path = os.path.join(output_folder, img_name)
    
    downsampled_img_tensor = transforms.ToTensor()(downsampled_img)
    vutils.save_image(downsampled_img_tensor, output_path)

def process_images(input_folder, output_folder, scale_factor=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = glob.glob(os.path.join(input_folder, '*.jpg'), recursive=True) + glob.glob(os.path.join(input_folder, '*.JPG'), recursive=True) + glob.glob(os.path.join(input_folder, '*.png'), recursive=True) + glob.glob(os.path.join(input_folder, '*.PNG'), recursive=True)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(downsample_image, [(file, output_folder, scale_factor) for file in files])

if __name__ == "__main__":
    parser = ArgumentParser(description="Downsample images in the specified scene")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--scale_factor", type=int, default=4, help="Factor by which to downsample the images")
    args = parser.parse_args()
    
    # Process both train and val datasets
    for split in ['train', 'val']:
        input_folder = os.path.join('./', args.scene, split, 'images')
        output_folder = os.path.join('./', args.scene, split, f'images_{args.scale_factor}')
        process_images(input_folder, output_folder, args.scale_factor)

    print(f'### Finished {args.scene}.')
