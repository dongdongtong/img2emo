from tqdm import tqdm
import pandas as pd

from os.path import join

from inferencer import MemPredictor


def predict(excel_file, model_path, dataset_name, out_excel_file):
    print(f"Loading {dataset_name} model from {model_path}")

    data_df = pd.read_excel(excel_file, converters={'product_id': str})
    
    lora_reduction = 8
    droprate = 0.
    mem_predictor = MemPredictor(
        lora_reduction=lora_reduction,
        droprate=droprate,
        model_path=model_path
    )
    
    pbar = tqdm(data_df.itertuples(), total=len(data_df))
    for row in pbar:
        product_id = row.product_id
        img_path = join("/data/dingsd/img2emo/img2attr/data/cover_images", product_id + ".png")
        
        memoriability = mem_predictor(img_path)
        print(f"Memoriabilities for {product_id}: {memoriability}")
        
        data_df.loc[row.Index, dataset_name + "_memoriability"] = memoriability.item()
        
        pbar.set_description(f"Processing {product_id}")
    
    data_df.to_excel(out_excel_file, index=False)


if __name__ == '__main__':
    print("Loading Image Memoriability predictor...")
    
    mem_model_names = ['LaMem', 'LNSIM', 'MemCat']
    mem_pretrained_model_paths = [
        # LaMem
        "output/clip_trivialaug_lamem/MemCLIPVitLora/clip_lamem_vit_lora_mse_srcc_ccc_ep20/LaMem/seed1_fold1/model/model-best.pth.tar",
        # LNSIM
        "output/clip_trivialaug_lnsim_cross_group/MemCLIPVitLora/clip_lnsim_vit_lora_mse_srcc_ccc_ep40/LNSIM/seed1/model/model-best.pth.tar",
        # MemCat
        "output/clip_trivialaug_memcat/MemCLIPVitLora/clip_memcat_vit_lora_mse_srcc_ccc_ep20/MemCat/seed1/model/model-best.pth.tar"
    ]
    
    excel_path = "/data/dingsd/img2emo/image_memoriability/datasets/final_data1.xlsx"
    out_excel_file = "/data/dingsd/img2emo/image_memoriability/datasets/final_data1_memoriability.xlsx"
    
    for model_name, model_path in zip(mem_model_names, mem_pretrained_model_paths):
        predict(excel_path, model_path, model_name, out_excel_file)
        
        excel_path = out_excel_file  # update the input file for the next model
        
    # image_path = '/root/autodl-tmp/Emotion6/images/anger/1.jpg'  # VA 3.6 5.4
    # print(va_predictor(image_path))