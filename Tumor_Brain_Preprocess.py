import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
TRAIN_DIR = "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Tumor-Brain-Dataset/Training"
TEST_DIR = "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Tumor-Brain-Dataset/Testing"
OUT_DIR = "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Tumor-Brain-Preprocess"

def create_dataset(IN_DIR, OUT_DIR, ouptut_csv):
    field_names = os.listdir(IN_DIR)
    file_sets = []
    label_sets = []
    for field_name in tqdm(field_names):
        
        image_folder = os.path.join(IN_DIR, field_name)
        
        file_names = os.listdir(image_folder)
        file_sets.extend(file_names)
        label_sets.extend([field_name] * len(file_names))
    
    data = {
        "C1": file_sets,
        "C2": label_sets
    }
    
    
    
    print(len(file_sets))
    print(len(label_sets))
    
    df = pd.DataFrame(data)
    
    label_mapping = {
        "glioma_tumor": 0,
        "meningioma_tumor": 1,
        "no_tumor": 2,
        "pituitary_tumor": 3
    }
    
    df["C2"] = df["C2"].map(label_mapping)
    
    df.to_csv(os.path.join(OUT_DIR, ouptut_csv), index=False, header=False)
    
    
def resize(Image_folder, size=(512,512)):
    for file_name in tqdm(os.listdir(Image_folder)):
        file_path = os.path.join(Image_folder, file_name)
        img = Image.open(file_path)
        resize_img = img.resize(size, Image.ANTIALIAS)
        resize_img.save(file_path)
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

create_dataset(TRAIN_DIR, OUT_DIR, TRAIN_CSV)
create_dataset(TEST_DIR, OUT_DIR, TEST_CSV)

# resize(os.path.join(OUT_DIR, "training"), (128, 128))
# resize(os.path.join(OUT_DIR, "testing"), (128,128))

