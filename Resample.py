from ImageDataSet import *

# Передискретизируйте объем в образце до заданного размера вокселя

class Resample(object):
   
    def __init__(self, new_resolution, flag):
        self.name = 'Resample'
        self.new_resolution = new_resolution
        self.flag = flag

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        new_resolution = self.new_resolution
        flag = self.flag

        if flag is True:
            image = ResampleSitkImage(image, spacing = new_resolution, interpolator = 'linear')
            label = ResampleSitkImage(label, spacing = new_resolution, interpolator = 'linear')

            return {'image': image, 'label': label}

        if flag is False:
            return {'image': image, 'label': label}
            

