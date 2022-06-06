import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.utils.data
from ImageProcessing import * 

class ImageDataset(Dataset):
    def __init__(self, 
                 data_list,
                 transforms = None,
                 train = False,
                 test = False,
                 segmentation_flag = True):

        self.data_list = data_list
        self.transforms = transforms
        self.train = train
        self.test = test
        self.bit = sitk.sitkFloat32
        self.segmentation_flag = segmentation_flag

    # the number of samples in dataset
    def __len__(self):
        return len(self.data_list)

    # function loads and returns a sample from the dataset at the given index
    def __getitem__(self, index):

        data = self.data_list[index]
        image_path = data["image"]
        label_path = data["label"]

        image = self.ReadImage(image_path)
        image = ImageNormalization(image)
        cast_image_filter = sitk.CastImageFilter()
        cast_image_filter.SetOutputPixelType(self.bit)
        image = cast_image_filter.Execute(image)

        if self.train:
            label = self.ReadImage(label_path)

            if self.segmentation_flag is False:
                label = ImageNormalization(label)  

            cast_image_filter.SetOutputPixelType(self.bit)
            label = cast_image_filter.Execute(label)

        elif self.test:
            label = self.ReadImage(label_path)

            if self.segmentation_flag is False:
                label = Normalization(label) 

            cast_image_filter.SetOutputPixelType(self.bit)
            label = cast_image_filter.Execute(label)

        else:
            label = sitk.Image(image.GetSize(), self.bit)
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

        sample = {'image': image, 'label': label}

        if self.transforms:  
            for transform in self.transforms:
                sample = transform(sample)

        image_array = sitk.GetArrayFromImage(sample['image'])
        label_array = sitk.GetArrayFromImage(sample['label'])

        if self.segmentation_flag is True:
            label_array = abs(np.around(label_array))

        image_array = np.transpose(image_array, (2, 1, 0))
        label_array = np.transpose(label_array, (2, 1, 0))

        image_array = image_array[np.newaxis, :, :, :]
        label_array = label_array[np.newaxis, :, :, :]

        image_tensor = torch.from_numpy(image_array)
        label_tensor = torch.from_numpy(label_array)

        return image_tensor, label_tensor

    def ReadImage(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    

    
