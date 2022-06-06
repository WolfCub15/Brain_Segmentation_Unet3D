import SimpleITK as sitk
import matplotlib.pyplot as plt 
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

from Parameters import *
from Utils import *

# ориентация изображения в определенном порядке осей 
# Выравнивание гистограммы изменяет контрастность изображения.
# AdaptiveHistogramEqualizationImageFilter может создавать гистограмму с адаптивным выравниванием или версию нерезкой маски (вычитание локального среднего).
# Вместо того, чтобы применять строгое выравнивание гистограммы в окне около пикселя, этот фильтр предписывает функцию отображения (степенной закон), 
# управляемую параметрами альфа и бета. 
# Параметр альфа определяет, насколько фильтр действует как классический метод выравнивания гистограммы (альфа=0) и насколько фильтр действует как нерезкая маска (альфа=1). 
# Параметр бета определяет, насколько фильтр действует как нерезкая маска (бета = 0) и насколько фильтр действует как сквозной (бета = 1, с альфа = 1). 
# Окно параметров управляет размером региона, по которому рассчитывается локальная статистика. Размер окна контролируется SetRadius. Радиус по умолчанию равен 5 во всех направлениях. 
# Изменяя альфа, бета и окно, можно получить множество фильтров выравнивания и нерезкого маскирования. Граничное условие игнорирует часть окрестности за пределами изображения и перевешивает допустимую часть окрестности.


# RescaleIntensityImageFilter применяет попиксельное линейное преобразование к значениям интенсивности пикселей входного изображения. 
# Линейное преобразование определяется пользователем в терминах минимального и максимального значений, которые должно иметь выходное изображение.

class AdaptiveHistogramEqualization(object):
   
    def __init__(self):
        self.name = 'AdaptiveHistogramEqualization'

    def __call__(self, sample):

        # адаптивное выравнивание гистограммы по степенному закону
        adaptive_hist_eq_img_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
        adaptive_hist_eq_img_filter.SetAlpha(0.7)
        adaptive_hist_eq_img_filter.SetBeta(0.8) 

        # выполняем фильтр на входном изображении
        image = adaptive_hist_eq_img_filter.Execute(sample['image'])  

        # применяем линейное преобразование к уровням интенсивности входного изображения
        resacle_intensity_img_filter = sitk.RescaleIntensityImageFilter()
        resacle_intensity_img_filter.SetOutputMaximum(255)
        resacle_intensity_img_filter.SetOutputMinimum(0)

        # выполняем фильтр на входном изображении
        image = resacle_intensity_img_filter.Execute(image)  
        label = sample['label']
        

        return {'image': image, 'label': label}


if __name__ == '__main__':

    data_list = CreateListFromPath(IMAGES_TRAIN_PATH, LABELS_TRARIN_PATH)
    data = data_list[1]
    image_path = data['image']
    label_path = data["label"]

    image = ReadImage(image_path)
    image = ImageNormalization(image)
    label = ReadImage(label_path)
    label = ImageNormalization(label)  

    sample = {'image': image, 'label': label}

    writer = sitk.ImageFileWriter()
    input_path = "./Result/hist_input.nii"
    writer.SetFileName(input_path)
    writer.Execute(image)

    hist = AdaptiveHistogramEqualization()

    result = hist(sample)
    res_image = result['image']

    writer = sitk.ImageFileWriter()
    result_path = "./Result/hist.nii"
    writer.SetFileName(result_path)
    writer.Execute(res_image)

    ###########################################
    example_filename = input_path
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
    
    #######################################
    example_filename = result_path
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