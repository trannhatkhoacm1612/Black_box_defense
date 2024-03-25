import csv

def replace_prefix(input_file, output_file, old_prefix, new_prefix):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        lines = [row for row in reader]

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in lines:
            # Thay thế tiền tố của mỗi dòng
            line[0] = line[0].replace(old_prefix, new_prefix)
            writer.writerow(line)

# Thay thế tiền tố của mỗi dòng trong file "input.csv" và lưu vào file "output.csv"
replace_prefix("/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG/split_set/test.csv", "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG/split_set/test_.csv", "/kaggle/input/kvasir-dataset-for-classification-and-segmentation/kvasir-seg", "/mlcv1/WorkingSpace/Personal/baotg/Khoa")
replace_prefix("/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG/split_set/train.csv", "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG/split_set/train_.csv", "/kaggle/input/kvasir-dataset-for-classification-and-segmentation/kvasir-seg", "/mlcv1/WorkingSpace/Personal/baotg/Khoa")