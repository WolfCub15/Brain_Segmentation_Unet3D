import os
import torch
import numpy as np
import torch
import glob
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import gc
import re
import SimpleITK as sitk
import shutil
from tqdm import tqdm

from Parameters import *
from DiceLoss import * 
from DiceLoss import *
from ImageDataSet import *
from AverageMeter import *
from Resample import *
from DataAugmentation import *
from Padding import *
from RandomCrop import *
from AdaptiveHistogramEqualization import *

def CheckDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def CheckFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def FreeGpuCache():
    print("GPU Usage until emptying the cache")
    gpu_usage()                             
    
    gc.collect()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

def MakeDataset(data, out_path, data_type):
    for i, file_name in tqdm(enumerate(data), total=len(data)):
        file_types = glob.glob(file_name)
        output_directory = "{}/".format(out_path)
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for type in file_types:
            if data_type in type:
                output_path = "{}".format(output_directory)
                shutil.copy(type, output_path)

def CreateListFromPath(data_image_path, data_label_path):
    data_image_list = glob.glob(os.path.join(data_image_path, '*'))
    data_label_list = glob.glob(os.path.join(data_label_path, '*'))
    data_image_list.sort()
    data_label_list.sort()

    list_from_data = [{'image': os.path.join(data_image_path), 'label': os.path.join(data_label_path)} for data_image_path, data_label_path  in zip(data_image_list, data_label_list)]

    return list_from_data

def SortingByNumbers(input_value):
    numbers = re.compile(r'(\d + )') # Компилирует объект регулярного выражения для последующего использования.
    s = numbers.split(input_value) # Разделение по строкам, подходящих под шаблон.
    #print(parts)
    s[1::2] = map(int, s[1::2])
    return s

def ImagesListFromPath(images_path):
    images_list_from_path = [] 

    for dir_name, subdir_list, file_list in os.walk(images_path):
        for name in file_list:
            images_list_from_path.append(os.path.join(dir_name, name))

    images_list_from_path = sorted(images_list_from_path, key = SortingByNumbers)

    return images_list_from_path

def WriteImagesToFolder(type, images_list, labels_list, n, offset, images_path, labels_path):
    for i in range(n):
        image_i = images_list[offset + i]
        label_i = labels_list[offset + i]
        print(type, i, image_i, label_i)

        image = sitk.ReadImage(image_i)
        label = sitk.ReadImage(label_i)

        sitk.WriteImage(image, os.path.join(images_path, f"image{i:d}.nii"))
        sitk.WriteImage(label, os.path.join(labels_path, f"label{i:d}.nii"))

def CreateDataset(out_path, data_type):
    data_path = DATASET_PATH

    HGG_data = glob.glob(data_path + "HGG/*/*")
    LGG_data = glob.glob(data_path + "LGG/*/*")

    MakeDataset(HGG_data, out_path, data_type)
    MakeDataset(LGG_data, out_path, data_type)

def CreateDatasetFolters(images_path, labels_path, validation_number = 50, testing_number = 50):
    images_train_out_path = IMAGES_TRAIN_PATH
    images_validation_out_path = IMAGES_VALIDATION_PATH
    images_test_out_path = IMAGES_TEST_PATH
    labels_train_out_path = LABELS_TRARIN_PATH
    labels_validation_out_path = LABELS_VALIDATION_PATH
    labels_test_out_path = LABELS_TEST_PATH

    images_list = ImagesListFromPath(images_path)
    labels_list = ImagesListFromPath(labels_path)

    CheckFolder(images_train_out_path)
    CheckFolder(images_validation_out_path)
    CheckFolder(images_test_out_path)
    CheckFolder(labels_train_out_path)
    CheckFolder(labels_validation_out_path)
    CheckFolder(labels_test_out_path)

    WriteImagesToFolder(type = 'train', 
                        images_list = images_list, 
                        labels_list = labels_list, 
                        n = len(images_list) - int(validation_number + testing_number), 
                        offset = int(testing_number + validation_number), 
                        images_path = images_train_out_path, 
                        labels_path = labels_train_out_path)

    WriteImagesToFolder(type = 'validation', 
                        images_list = images_list, 
                        labels_list = labels_list, 
                        n = int(validation_number), 
                        offset = int(testing_number), 
                        images_path = images_validation_out_path, 
                        labels_path = labels_validation_out_path)

    WriteImagesToFolder(type = 'test', 
                        images_list = images_list, 
                        labels_list = labels_list, 
                        n = int(validation_number), 
                        offset = 0, 
                        images_path = images_test_out_path, 
                        labels_path = labels_test_out_path)

def ReadImage(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image

def HistogrammProcessing(path_1, path_2, result_path_1, result_path_2):
    data_list = CreateListFromPath(path_1, path_2)
    n = len(data_list)
    CheckFolder(result_path_1)
    CheckFolder(result_path_2)

    for i in range(n):
        data = data_list[i]
        image_path = data['image']
        label_path = data["label"]

        image = ReadImage(image_path)
        image = ImageNormalization(image)
        label = ReadImage(label_path)
        label = ImageNormalization(label)  
        sample = {'image': image, 'label': label}

        hist = AdaptiveHistogramEqualization()
        result = hist(sample)
        res_image = result['image']
        res_label = result['label']

        sitk.WriteImage(res_image, os.path.join(result_path_1, f"image{i:d}.nii"))
        sitk.WriteImage(res_label, os.path.join(result_path_2, f"label{i:d}.nii"))

    


if __name__ == "__main__":
    #CreateDataset(DATASET_LABELS_PATH, 'seg')
    #CreateDataset(DATASET_IMAGES_FLAIR_PATH, 'flair')
    #CreateDataset(DATASET_IMAGES_T1CE_PATH, 't1ce')

    #CreateDatasetFolters(DATASET_IMAGES_FLAIR_PATH, DATASET_LABELS_PATH)

    #HistogrammProcessing(IMAGES_TRAIN_PATH, LABELS_TRARIN_PATH, DATASET_HIST_IMAGES_TRAIN_PATH, DATASET_HIST_LABELS_TRARIN_PATH)
    HistogrammProcessing(IMAGES_VALIDATION_PATH, LABELS_VALIDATION_PATH, DATASET_HIST_IMAGES_VALIDATION_PATH, DATASET_HIST_LABELS_VALIDATION_PATH)
    HistogrammProcessing(IMAGES_TEST_PATH, LABELS_TEST_PATH, DATASET_HIST_IMAGES_TEST_PATH, DATASET_HIST_LABELS_TEST_PATH)
