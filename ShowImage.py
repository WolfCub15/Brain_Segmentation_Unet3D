import matplotlib.pyplot as plt 
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

#example_filename = "./dataset2019/archive/training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_flair.nii"
example_filename = './Result/result_3.nii'
img = nib.load(example_filename)
print(img)
 
width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()
 
num = 1
for i in range(0, height, 20):
    img_arr = img.dataobj[:, i, :]
    plt.subplot(5, 3, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
 
plt.show()

num = 1
for i in range(0, width, 20):
    img_arr = img.dataobj[i, :, :]
    plt.subplot(5, 3, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
 
plt.show()

test_image=nib.load(example_filename).get_data()
plt.imshow(test_image[75])
plt.show()

'''
import SimpleITK as sitk
from matplotlib import pyplot as plt

itk_img = sitk.ReadImage(example_filename)
img = sitk.GetArrayFromImage(itk_img)
print(img.shape) 
print(img[2][50][50][150]) 
plt.imshow(img[0, 85, :, :], cmap='gray')
plt.show()
'''