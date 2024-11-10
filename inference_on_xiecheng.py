from tqdm import tqdm
import pandas as pd

from os.path import join

from inferencer import VAPredictor


if __name__ == '__main__':
    print("Loading VA predictor...")
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