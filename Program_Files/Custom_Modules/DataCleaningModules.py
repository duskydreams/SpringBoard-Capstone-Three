#This file contains the functions needed to do data cleaning for EDA
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError #Image package for data wrangling

def safe_load_image(path):
    try:
        with Image.open(path) as img:
            # Reorient based on phone camera EXIF orientation tag
            img = ImageOps.exif_transpose(img)
            
            # Enforce 3-channel RGB (drops transparency, promotes grayscale)
            img = img.convert("RGB")
            
            # Force-load pixel data before closing file context
            img.load()
            return img
    except (UnidentifiedImageError, OSError):
        # File is corrupt, empty, or unreadable
        print("File cannot be read, please check input") 

def load_images_from_directory(directory_path, recursive=False):
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    
    # Choose flat scan or recursive search through subdirectories
    file_generator = dir_path.rglob("*") if recursive else dir_path.glob("*")
    
    loaded_images = {}  # {filename: PIL.Image}
    
    for file_path in file_generator:
        # Ignore non-images and hidden OS files
        if file_path.suffix.lower() in valid_exts and not file_path.name.startswith("."):
            img = safe_load_image(file_path)
            if img is not None:
                loaded_images[file_path.name] = img
                
    return loaded_images

def letterbox(img, target_size = (160,160), fill_color = (0,0,0)):
    return ImageOps.pad(img, target_size, color =  fill_color, method = Image.Resampling.BILINEAR) 

def image_centering(img, target_size = (160,160)): #Only works on images that need to go smaller
    width, height = img.size
    target_width,target_height = target_size

    #Edge case if image is image is too small than target
    if width < target_width or height < target_height: 
        scale = max(target_width / width, target_height / height) #determine which side is the smaller size to determine scale 
        resize_width, resize_height = int(width * scale), int(height * resize_height)
        img = img.resize((resize_width,resize_height), Image.Resampling.BILINEAR)
        width,height = img.size

    #Determine how much excess needs to be removed 
    left = (width - target_width) // 2 
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height 

    return img.crop((left,top,right,bottom))

def img_to_tensor_array(img):
    array = np.asarray(img)
    array = array.astype(np.float32) / 255.0
    array = np.transpose (array, (2,0,1))
    return array

def rgba_to_rgb_clear_background(img, bg_colors = (255,255,255)): 
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, bg_colors)
        background.paste(img, mask = img.split()[-1])
        return background
    return img.convert("RGB")