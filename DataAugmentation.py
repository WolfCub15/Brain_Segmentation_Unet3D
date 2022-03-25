import SimpleITK as sitk
import numpy as np
from ImageProcessing import * 
'''
Увеличение данных — это подход для увеличения набора данных для обучения. 
Решаемая проблема заключается в том, что исходный набор данных 
недостаточно репрезентативен для общей совокупности изображений. 
Увеличение данных в анализе данных — это методы, используемые для увеличения объема данных 
путем добавления слегка измененных копий уже существующих данных
или вновь созданных синтетических данных из существующих данных. 
Он действует как регуляризатор и помогает уменьшить переоснащение при обучении модели машинного обучения.
'''
class DataAugmentation(object):
    def __init__(self):
        self.name = 'DataAugmentation'

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        random_augmentation = np.random.choice(7)

        if random_augmentation == 0: 
            return {'image': image, 'label': label}

        if random_augmentation == 1: 
            noise_image_filter = sitk.AdditiveGaussianNoiseImageFilter()
            noise_image_filter.SetMean(np.random.uniform(0, 1))
            noise_image_filter.SetStandardDeviation(np.random.uniform(0, 2))

            image = noise_image_filter.Execute(image)
            if Segmentation is False:
                label = noise_image_filter.Execute(label)

            return {'image': image, 'label': label}

        if random_augmentation == 2: 
            image = AdjustingBrightness(image, -15, 15)
            if Segmentation is False:
                label = AdjustingBrightness(label, -15, 15)

            return {'image': image, 'label': label}

        if random_augmentation == 3:  
            image = ImageContrast(image)
            if Segmentation is False:
                label = ImageContrast(label) 

            return {'image': image, 'label': label}

        if random_augmentation == 4:
            image = ImadJust(image)

            return {'image': image, 'label': label}

        if random_augmentation == 5: 
            image = Flipping(image, 0)
            label = Flipping(label, 0)

            return {'image': image, 'label': label}

        if random_augmentation == 6: 
            image = Flipping(image, 1)
            label = Flipping(label, 1)

            return {'image': image, 'label': label}
