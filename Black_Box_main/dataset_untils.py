import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import argparse
import random
import csv


def extract_path_file(train_dir, 
                      test_dir, 
                      dataset, 
                      output_dir):
    
    """
    Args:
        train_dir: folder containing images, with kvasir is images folder, not mask
        test_dir: folder containing images
        dataset: ["kvasir", "brain_tumor"]
        ouptut_dir: output folder containing cvs map files
    """
    
    
    if not os.path.exists(os.path.join(output_dir, dataset)): # ex ./kvasir/
        os.makedirs(os.path.join(output_dir, dataset))
    
    if dataset == "brain_tumor":
        output_path = os.path.join(output_dir, "train.csv") # ex ./kvasir/train.csv
        with open(output_path, "w+") as f:
            for file_name in os.listdir(train_dir):
                file_path = os.path.join(train_dir, file_name) # ex ./{train_dir}/file_name.jpg
                f.write(file_path + "\n")
        
        output_path = os.path.join(output_dir, "test.csv") # ex ./kvasir/test.csv
        with open(output_path, "w+") as f:       
            for file_name in os.listdir(test_dir):
                file_path = os.path.join(test_dir, file_name)
                f.write(file_path + "\n")
                
    elif dataset == "kvasir":
        img_name_lists = os.listdir(train_dir)
        train_nums = int(0.8 * len(img_name_lists))
        train_indxs = random.sample(range(len(img_name_lists)), train_nums)
        test_indxs = set(range(len(img_name_lists))).difference(train_indxs)
        train_name_lists = [img_name_lists[i] for i in train_indxs]
        test_name_lists = [img_name_lists[i] for i in test_indxs]
        
        train_path_lists = [os.path.join(train_dir, file_name) for file_name in train_name_lists]
        test_path_lists = [os.path.join(train_dir, file_name) for file_name in test_name_lists]
    
        output_path = os.path.join(output_dir, "train.csv") # ex ./kvasir/train.csv
        with open(output_path, "w+") as f:
            for file_path in train_path_lists:
                f.write(file_path + "\n")
        
        output_path = os.path.join(output_dir, "test.csv") # ex ./kvasir/test.csv
        with open(output_path, "w+") as f:       
            for file_path in test_path_lists:
                f.write(file_path + "\n")
                
def replace_prefix(input_file, 
                   output_file,
                   old_prefix,
                   new_prefix):
    """
    Args:
        input_file: original csv file
        output_file: ouptut csv file
        old_prefix: The orginal str want to replace
        new_prefix: The output str want to replace
    """
    
    
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        lines = [row for row in reader]

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in lines:
            line[0] = line[0].replace(old_prefix, new_prefix)
            writer.writerow(line)

replace_prefix(input_file=r"D:\research\MAPR_src\Kvasir-SEG\split_set\train_.csv",
               output_file=r"D:\research\MAPR_src\train_kvasir.csv",
               old_prefix="/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG",
               new_prefix="option_folder")


# extract_path_file(train_dir=r"D:\research\MAPR_src\Tumor-Brain-Preprocess\training",
#                   test_dir=r"D:\research\MAPR_src\Tumor-Brain-Preprocess\testing",
#                   dataset="brain_tumor",
#                   output_dir=r"D:\research\MAPR_src")