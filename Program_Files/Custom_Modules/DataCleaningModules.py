#This file contains the functions needed to do data cleaning for EDA
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError #Image package for data wrangling

def read_label_txt(data_path : file, text_path: file) -> DataFrame: 
    df_labels = pd.read_csv(text_path) #Assumes this will always be CSV

    records = []
    for row in df_labels.itertuples(index = False):
        person = str(row.person_name).strip() 
        img_num = str(row.image_num).strip()
        mask_info = str(row.mask_status).strip().lower() 

        is_masked = 0 if mask_info in {"no-mask"} else 1
        mask_type = "none" if is_masked == 0 else mask_info

        filename = f"{person}_{img_num.zfill(4)}.png"

        file_path = data_path / person / filename

        if file_path.exists(): 
            with Image.open(file_path) as img: 
                w,h = img.size
                mode = img.mode

            records.append({
            "filepath": str(file_path),
            "filename": filename,
            "person_name": person,
            "image_num": img_num,
            "is_masked": is_masked,
            "mask_type": mask_type,
            "orig_width": w,
            "orig_height": h,
            "color_mode": mode
            })

        else: 
            print(f"Warning: Image not found -> {file_path}")

    df_eda = pd.DataFrame(records)
    return df_eda

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

def letterbox(img, method, target_size = (160,160), fill_color = (0,0,0)):
    return ImageOps.pad(img, size = target_size, method = method, color =  fill_color) 


def image_centering(img, target_size = (160,160), method =Image.Resampling.LANCZOS, bleed = 0.0 ,centering =(.5,.5)): #Only works on images that need to go smaller
    return ImageOps.fit(img, target_size, method, bleed, centering )

def img_to_tensor_array(img):
    array = np.asarray(img)
    array = array.astype(np.float32) / 255.0
    array = np.transpose (array, (2,0,1)) #H,W,C -> C,H,W Height,Width,Color
    return array

def rgba_to_rgb_clear_background(img, bg_colors = (255,255,255)): 
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, bg_colors)
        background.paste(img, mask = img.split()[-1])
        return background
    return img.convert("RGB")