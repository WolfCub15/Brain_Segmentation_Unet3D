import nibabel as nib
import matplotlib.pyplot as plt
import gzip
import matplotlib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
from matplotlib import pylab as plt
import numpy as np
import cv2  
import SimpleITK as sitk
from sklearn import preprocessing
import glob
import os
import shutil
from tqdm import tqdm
from Folders import *
import imutils

def Show(image, title):
    plt.title(title)
    plt.imshow(image)
    plt.show()

'''
Адаптивное выравнивание гистограммы. 
Этот метод позволяет улучшить контраст без одновременного увеличения шума.

clipLimit: порог ограничения контрастности.

tileGridSize: делит входное изображение на M x N плиток, 
а затем применяет выравнивание гистограммы к каждой локальной плитке.
'''

def CLAHE(input_img, print_flag = False):
    clahe_img = input_img.copy()
    clane_gray_img = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0,	tileGridSize=(8, 8))
    clahe_img = clahe.apply(clane_gray_img)

    if print_flag == True:
        plt.title('CLAHE')
        plt.imshow(clahe_img, cmap='gray')
        plt.show()
    
    return clahe_img

'''
Извлечение мозга
'''
def BrainLocalization(input_img, print_flag = False):
    img = input_img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    otsu_thresh_val, img_res = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU)

    ret, markers = cv2.connectedComponents(img_res)

    # метка 0 - это фон, поэтому игнорируем ее и получаем области, которые занимают каждая компонента
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0] 

    largest_component = np.argmax(marker_area) + 1 

    # пиксели, которые соответствуют мозгу
    brain_mask = markers == largest_component 

    brain_out = img.copy()

    #остальные пиксели очищаем
    brain_out[brain_mask == False] = (0,0,0) 

    if print_flag == True:
        Show(brain_out, 'Brain localization')

    return brain_out

def CropImage(input_img, INPUT_IMAGE_SIZE, print_flag = False):
    img = input_img.copy()
    img = cv2.resize(img, dsize = INPUT_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations = 2)
    thresh = cv2.dilate(thresh, None, iterations = 2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key = cv2.contourArea)

    extreme_left = tuple(c[c[:, :, 0].argmin()][0])
    extreme_right = tuple(c[c[:, :, 0].argmax()][0])
    extreme_top = tuple(c[c[:, :, 1].argmin()][0])
    extreme_bot = tuple(c[c[:, :, 1].argmax()][0])

    img_draw_contours = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    img_pnt = cv2.circle(img_draw_contours.copy(), extreme_left, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extreme_right, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extreme_top, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extreme_bot, 8, (255, 255, 0), -1)

    crop_image = img[extreme_top[1] : extreme_bot[1] , extreme_left[0] : extreme_right[0]].copy()

    if print_flag == True:
        plt.figure(figsize=(10,2))
        plt.subplot(141)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Original image')

        plt.subplot(142)
        plt.imshow(img_draw_contours)
        plt.xticks([])
        plt.yticks([])
        plt.title('The biggest contour')

        plt.subplot(143)
        plt.imshow(img_pnt)
        plt.xticks([])
        plt.yticks([])
        plt.title('The extreme points')

        plt.subplot(144)
        plt.imshow(crop_image)
        plt.xticks([])
        plt.yticks([])
        plt.title('Crop image')

        plt.show()

    return crop_image

def ShowImage3D(img, type):
    OrthoSlicer3D(img.dataobj).show()
    
    num = 1
    for i in range(0, type, 20):
        img_arr = img.dataobj[:, i, :]
        plt.title('Show Image 3D')
        plt.imshow(img_arr, cmap='gray')
        num += 1

def ShowImageSlice(img, slice):
    test_image = nib.load(img).get_data()

    Show(test_image[slice], 'Show Image Slice')

def Image2Array(example_filename, x = 0, y = 0, z = 0, print_flag = False):
    itk_img = sitk.ReadImage(example_filename)
    img = sitk.GetArrayFromImage(itk_img)

    if print_flag == True:
        plt.title('Show Image Slice array')
        plt.imshow(img[x, :, :], cmap='gray')
        plt.show()

    return img

def ImageNormalize(img_array, K, print_flag = False):
    normalize_img = preprocessing.normalize(img_array[K])
    qwerty = normalize_img[:,:] * 1000

    if print_flag == True:
        plt.title('Show Image Normalization')
        plt.imshow(qwerty[:, :], cmap='gray')
        plt.show()

    return qwerty

def Nii2Cv2(example_filename, K, print_flag = False):
    image_array = Image2Array(example_filename, K)
    image_normalize = ImageNormalize(image_array, K)

    array = image_normalize.copy()
    cv2.imwrite('img.jpg', array)

    image = cv2.imread("img.jpg")
    if print_flag == True:
        plt.title('Show Image cv2')
        plt.imshow(image[ :, :], cmap='gray')
        plt.show()

    return image

def PrintHist(source_img, hist_img):
    plt.figure(figsize=(10,2))
    ax1 = plt.subplot(1,2,1)
    plt.title("Source histogram")
    plt.hist(source_img.ravel(), bins=range(255))

    ax2 = plt.subplot(1,2,2, sharey = ax1)
    plt.title("CLAHE histogram")
    plt.hist(hist_img.ravel(), bins=range(255))

    plt.show()

def Preprocessing(example_filename, K, print_hist = False):
    image = Nii2Cv2(example_filename, K)
    INPUT_IMAGE_SIZE = (image.shape[0],image.shape[1])

    brain_local = image.copy()
    brain_local = BrainLocalization(brain_local)

    crop_img = CropImage(brain_local, INPUT_IMAGE_SIZE)

    source_img = crop_img.copy()
    hist_img = crop_img.copy()
    hist_img = CLAHE(hist_img)

    if print_hist == True:
        PrintHist(source_img, hist_img)

    return hist_img


def main():
    '''
    images_path = "./Dataset/images_flair"
    out_path = "./Dataset/images_flair_process"

    list_images = ListFiles(images_path)

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for i in range(len(list_images)):
        example_filename = list_images[i]
        img = nib.load(example_filename)

        IMAGE_SIZE = img.shape
        print(IMAGE_SIZE)
    '''

    
    example_filename = "./Dataset/images_flair/BraTS19_2013_1_1_flair.nii"
    img = nib.load(example_filename)
    #print(img)
    width, height, queue = img.dataobj.shape

    slice = 150
    K = 90

    #ShowImage3D(img, width)
    #ShowImageSlice(example_filename, slice)

    preprocessing_result = Preprocessing(example_filename, K)
    plt.title('Preprecessing')
    plt.imshow(preprocessing_result, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()