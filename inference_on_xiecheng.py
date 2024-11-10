from inferencer import VAPredictor


import os
from PIL import Image
import numpy as np
import cv2

from scipy.ndimage import center_of_mass

from glob import glob

import torch
import time
from tqdm import tqdm
import pandas as pd

from PIL import Image
from transformers import AutoModel, AutoTokenizer

from os.path import join

import json
import re


def analyze_image_hsv(image_path):
    # Read the image as RGB
    img = cv2.imread(image_path)
    # print(img.shape)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split the HSV channels
    h, s, v = cv2.split(hsv)
    
    # Calculate warm hue ratio
    warm_hue_mask = np.logical_or(h <= 30, h >= 110)
    warm_hue_ratio = np.sum(warm_hue_mask) / h.size
    
    # Calculate average saturation
    avg_saturation = np.mean(s)
    
    # Calculate average brightness
    avg_brightness = np.mean(v)
    
    # Calculate contrast of brightness
    brightness_contrast = np.std(v)
    
    return {
        'warm_hue_ratio': warm_hue_ratio,
        'avg_saturation': avg_saturation,
        'avg_brightness': avg_brightness,
        'brightness_contrast': brightness_contrast
    }


def compute_composition_attributes(image_path):
    # Load the image and compute saliency map
    _, saliency_map_pil_image = saliency_extractor.extract_object(image_path)
    dl_saliency_map = np.array(saliency_map_pil_image)  # min 0, max 255
    _, dl_binary_map = cv2.threshold(dl_saliency_map, 0.5, 1, cv2.THRESH_BINARY)
    # cv2.imwrite('dl_binary_map.png', dl_binary_map * 255)
    # print(saliency_map.shape, saliency_map.min(), saliency_map.max())

    saliency_detector= MR_saliency()
    cvpr13_saliency_map = saliency_detector.saliency(image_path)
    
    # Threshold the saliency map to get a binary map of the main element
    normalized_cvpr13_saliency_map = (cvpr13_saliency_map - cvpr13_saliency_map.min()) / (cvpr13_saliency_map.max() - cvpr13_saliency_map.min())
    _, cvpr13_binary_map = cv2.threshold(normalized_cvpr13_saliency_map, 0.5, 1, cv2.THRESH_BINARY)
    cvpr13_binary_map = cv2.resize(cvpr13_binary_map, (dl_binary_map.shape[1], dl_binary_map.shape[0]))
    # cv2.imwrite('cvpr13_binary_map.png', cvpr13_binary_map * 255)

    # we combine these two binary maps
    binary_map = np.logical_or(dl_binary_map, cvpr13_binary_map)
    # cv2.imwrite('binary_map.png', binary_map * 255)
    
    # Load the original image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Compute center of mass of the main element
    cy, cx = center_of_mass(binary_map)
    
    # 1. Diagonal Dominance
    diag1 = lambda x, y: y - x * (height / width)
    diag2 = lambda x, y: y + x * (height / width) - height
    
    dist1 = abs(diag1(cx, cy)) / np.sqrt(1 + (height/width)**2)
    dist2 = abs(diag2(cx, cy)) / np.sqrt(1 + (height/width)**2)
    
    diagonal_dominance = 1 - min(dist1, dist2) / (np.sqrt(height**2 + width**2) / 2)
    
    # 2. Symmetry
    left_sum = np.sum(binary_map[:, :width//2])
    right_sum = np.sum(binary_map[:, width//2:])
    symmetry = abs(float(left_sum) - float(right_sum)) / float(np.sum(binary_map) + 1)
    
    # 3. Visual Balance
    left_half = image[:, :width//2]
    right_half = image[:, width//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    color_distance = np.mean(np.sqrt(np.sum((left_half.astype(float) - right_half_flipped.astype(float))**2, axis=2)))
    visual_balance = color_distance / 255  # Normalize to [0, 1]
    
    # 4. Rule of Thirds (post-processing in image_content_yolo.py)
    intersections = [
        (width / 3, height / 3),
        (2 * width / 3, height / 3),
        (width / 3, 2 * height / 3),
        (2 * width / 3, 2 * height / 3)
    ]
    
    distances = [np.sqrt((cx - x)**2 + (cy - y)**2) for x, y in intersections]
    rule_of_thirds = [1 - d / np.sqrt(width**2 + height**2) for d in distances]
    rule_of_thirds_left_top = rule_of_thirds[0]
    rule_of_thirds_right_top = rule_of_thirds[1]
    rule_of_thirds_left_bottom = rule_of_thirds[2]
    rule_of_thirds_right_bottom = rule_of_thirds[3]

    # 5. Size Difference
    size_difference = np.sum(binary_map) / (height * width)
    
    # 6. Color Difference
    main_element_mask = binary_map.astype(bool)
    background_mask = ~main_element_mask
    
    main_element_color = np.mean(image[main_element_mask], axis=0)
    background_color = np.mean(image[background_mask], axis=0)
    
    color_difference = np.sqrt(np.sum((main_element_color - background_color)**2)) / 255
    
    # 7. Texture Difference
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    
    main_element_edges = edges[main_element_mask]
    background_edges = edges[background_mask]
    
    main_element_edge_density = np.sum(main_element_edges) / (np.sum(main_element_mask) + 1)
    background_edge_density = np.sum(background_edges) / (np.sum(background_mask) + 1)
    
    texture_difference = background_edge_density - main_element_edge_density
    
    return {
        'diagonal_dominance': diagonal_dominance,
        'symmetry': symmetry,
        'visual_balance': visual_balance,
        'rule_of_thirds_left_top': rule_of_thirds_left_top,
        'rule_of_thirds_right_top': rule_of_thirds_right_top,
        'rule_of_thirds_left_bottom': rule_of_thirds_left_bottom,
        'rule_of_thirds_right_bottom': rule_of_thirds_right_bottom,
        'size_difference': size_difference,
        'color_difference': color_difference,
        'texture_difference': texture_difference
    }


def get_image_va_attributes(predictor, image_path):

    va_dict = predictor(image_path)
    v = va_dict['valence']
    a = va_dict['arousal']

    return v, a


@torch.no_grad()
def get_image_content_attributes(image_path):
    torch.manual_seed(0)

    image = Image.open(image_path).convert('RGB')

    question = '''
    Please watch the image carefully and finish the following four tasks: count the number of persons, count the number of human faces, count the number of arbitrary animals, and extract the all the text in the image. 
    Please output the result in the json format.
    For example, the response must be in the following format:
    {
        "person": 20,
        "human_face": 6,
        "animal": 0,
        "text": ["hello", "world", ..., "brand"]
    }
    Do not recognize the statue, cartoon, or other non-real objects as humans, or as human faces, or as animals.
    Please directly output the text in one line in string type. If there is infinite text in the image, just output the first 20 words.
    If there is no text in the image, the "text" field should be an empty list, i.e., "text": [].
    Do not output other information. JSON format is the only acceptable output information.
    '''

    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=minicpm_tokenizer
    )
    print(res)

    # remove "+" in the res
    res = res.replace('+', '')

    # decompose the res to get the json output
    res = re.findall(r'\{[^}]*\}', res)
    # print(res)
    if len(res) == 0:
        res = res + "']}"
    res = json.loads(res[0])
    # print(res)

    return res


def do_job(image_path):
    product_id = os.path.basename(image_path).split('.')[0]
    print("Processing product:", product_id)

    # extract color attributes
    color_attributes = analyze_image_hsv(image_path=image_path)

    # extract composition attributes
    torch.cuda.empty_cache()
    composition_attributes = compute_composition_attributes(image_path=image_path)

    # extract VA attributes
    torch.cuda.empty_cache()
    v, a = get_image_va_attributes(va_predictor, image_path)

    # extract content attributes
    torch.cuda.empty_cache()
    content_attr = get_image_content_attributes(image_path)

    return [
        product_id,
        color_attributes['warm_hue_ratio'],
        color_attributes['avg_saturation'],
        color_attributes['avg_brightness'],
        color_attributes['brightness_contrast'],
        composition_attributes['diagonal_dominance'],
        composition_attributes['symmetry'],
        composition_attributes['visual_balance'],
        composition_attributes['rule_of_thirds_left_top'],
        composition_attributes['rule_of_thirds_right_top'],
        composition_attributes['rule_of_thirds_left_bottom'],
        composition_attributes['rule_of_thirds_right_bottom'],
        composition_attributes['size_difference'],
        composition_attributes['color_difference'],
        composition_attributes['texture_difference'],
        v,
        a,
        content_attr['person'],
        content_attr['human_face'],
        content_attr['animal'],
        content_attr['text']
    ]


def main():
    image_dir = "/root/autodl-tmp/cover_images"
    image_paths = list(glob(os.path.join(image_dir, "*.png")))
    image_paths = sorted(image_paths)
    print("image numbers:", len(image_paths))

    img_attr_df = pd.DataFrame(columns=['product_id', 'warm_hue_ratio', 'avg_saturation', 'avg_brightness', 'brightness_contrast',
                                        'diagonal_dominance', 'symmetry', 'visual_balance', 'rule_of_thirds_left_top',
                                        'rule_of_thirds_right_top', 'rule_of_thirds_left_bottom', 'rule_of_thirds_right_bottom',
                                        'size_difference', 'color_difference', 'texture_difference', 'valence', 'arousal',
                                        'person', 'human_face', 'animal', 'text'])
    csv_path = '/root/autodl-tmp/img_attr_df.csv'

    if os.path.exists(csv_path):
        img_attr_df = pd.read_csv(csv_path)
        print("read csv file:", csv_path)

        analyzed_product_count = 0
    else:
        analyzed_product_count = 0

    for image_idx, image_path in enumerate(tqdm(image_paths)):
        if image_idx < analyzed_product_count:
            continue

        valence, arousal = get_image_va_attributes(va_predictor, image_path)
        # print(attributes)

        img_attr_df.loc[len(img_attr_df)] = [valence, arousal]

        img_attr_df.to_csv('/root/autodl-tmp/img_attr_df_multidataset.csv', index=False)

        # break

    img_attr_df.to_csv('/root/autodl-tmp/img_attr_df_multidataset.csv', index=False)


if __name__ == '__main__':
    print("Loading VA predictor...")
    global va_predictor
    va_predictor = VAPredictor(
        pretrained_model="vit_giant_patch14_clip_224_lora", 
        lora_reduction=8, 
        droprate=0.,
        model_path="output/va/EmoticVitLora/emotic_vit_lora_ep20/EMOTIC_NAPS_H_OASIS_Emotion6/more_trivial_augment_seed1_method2/model/model-best.pth.tar"
    )

    excel_file = "/root/autodl-tmp/img_attr_df.xlsx"
    data_df = pd.read_excel(excel_file, converters={'product_id': str})
    
    pbar = tqdm(data_df.itertuples(), total=len(data_df))
    for row in pbar:
        product_id = row.product_id
        
        img_path = join("/root/autodl-tmp/cover_images", product_id + ".png")
        
        va_out_dict = va_predictor(img_path)
        
        data_df.loc[row.Index, 'valence'] = va_out_dict['valence']
        data_df.loc[row.Index, 'arousal'] = va_out_dict['arousal']
        print(va_out_dict['valence'], va_out_dict['arousal'])
        
        pbar.set_description(f"Processing {product_id}")
    
    out_excel_file = "/root/autodl-tmp/img_attr_df_multidataset.xlsx"
    data_df.to_excel(out_excel_file, index=False)
        
    # image_path = '/root/autodl-tmp/Emotion6/images/anger/1.jpg'  # VA 3.6 5.4
    # print(va_predictor(image_path))