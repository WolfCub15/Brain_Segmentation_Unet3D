import os
import re
import SimpleITK as sitk

def NumericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def FilesList(Path):
    images_list = [] 
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key = NumericalSort)
    return images_list

def CheckFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def WriteImages(type, list_images, list_labels, n, offset, images_path, labels_path):
    for i in range(n):

        a = list_images[offset+i]
        b = list_labels[offset+i]

        print(type, i, a, b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image_directory = os.path.join(images_path, f"image{i:d}.nii")
        label_directory = os.path.join(labels_path, f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

if __name__ == "__main__":
    images_path = './Dataset/images_flair'
    labels_path = './Dataset/labels'
    validation_number = 50
    testing_number = 50

    images_train_out_path = './Folders/images/train'
    images_validation_out_path = './Folders/images/validation'
    images_test_out_path = './Folders/images/test'
    labels_train_out_path = './Folders/labels/train'
    labels_validation_out_path = './Folders/labels/validation'
    labels_test_out_path = './Folders/labels/test'  

    list_images = FilesList(images_path)
    list_labels = FilesList(labels_path)

    CheckFolder(images_train_out_path)
    CheckFolder(images_validation_out_path)
    CheckFolder(images_test_out_path)
    CheckFolder(labels_train_out_path)
    CheckFolder(labels_validation_out_path)
    CheckFolder(labels_test_out_path)

    WriteImages('train', list_images, list_labels, len(list_images) - int(validation_number + testing_number), int(testing_number + validation_number), images_train_out_path, labels_train_out_path)
    WriteImages('validation', list_images, list_labels, int(validation_number), int(testing_number), images_validation_out_path, labels_validation_out_path)
    WriteImages('test', list_images, list_labels, int(validation_number), 0, images_test_out_path, labels_test_out_path)
