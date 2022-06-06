import SimpleITK as sitk
import numpy as np

Segmentation = True

def GetArray(image):
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, (2, 1, 0))
    image_spacing = image.GetSpacing()
    image_direction = image.GetDirection()
    image_origin = image.GetOrigin()

    return image_array, image_spacing, image_direction, image_origin

def GetImage(image_array, image_spacing, image_direction, image_origin):
    image = np.transpose(image_array, (2, 1, 0))
    image = sitk.GetImageFromArray(image)
    image.SetDirection(image_direction)
    image.SetOrigin(image_origin)
    image.SetSpacing(image_spacing)

    return image

def ReadFile(file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(file)
    image = reader.Execute()
    image = ImageNormalization(image)

    return image

def AdjustingBrightness(image, low, high):
    image_array, image_spacing, image_direction, image_origin = GetArray(image)

    add = np.random.randint(low, high)
    image_array = image_array + add
    image_array[image_array >= 255] = 255
    image_array[image_array <= 0] = 0

    brightness_image = GetImage(image_array, image_spacing, image_direction, image_origin)

    return brightness_image

def ImageContrast(image):
    image_array, image_spacing, image_direction, image_origin = GetArray(image)

    N = image_array.shape[0] * image_array.shape[1] * image_array.shape[2]
    sum = np.sum(image_array)
    average_brightness = sum / N
    delta = image_array - average_brightness
    correction = np.random.randint(-15, 15)
    delta_correct = delta * abs(correction) / 100

    tmp = image_array

    if correction >= 0:
        tmp = tmp + delta_correct
    else:
        tmp = tmp - delta_correct

    tmp[tmp >= 255] = 255
    tmp[tmp <= 0] = 0

    contrast_image = GetImage(tmp, image_spacing, image_direction, image_origin)

    return contrast_image

# imadjust - увеличение контраста изображений путем изменения диапазона интенсивностей исходного изображения.
def ImadJust(image):
    image_array, image_spacing, image_direction, image_origin = GetArray(image)

    numerator = image_array - image_array.min()
    denominator = image_array.max() - image_array.min()
    gamma = np.random.uniform(1, 2)
    image_array = ((numerator / denominator) ** gamma) * 255

    imadjust_image = GetImage(image_array, image_spacing, image_direction, image_origin)

    return imadjust_image

def Flipping(image, axes):
    image_array, image_spacing, image_direction, image_origin = GetArray(image)

    if axes == 0:
        image_array = np.fliplr(image_array)
    else:
        image_array = np.flipud(image_array)

    flipping_image = GetImage(image_array, image_spacing, image_direction, image_origin)

    return flipping_image


def ImageNormalization(image):
    normalize_image_filter = sitk.NormalizeImageFilter()
    resacle_intensity_filter = sitk.RescaleIntensityImageFilter()
    resacle_intensity_filter.SetOutputMaximum(255)
    resacle_intensity_filter.SetOutputMinimum(0)

    image = normalize_image_filter.Execute(image) 
    image = resacle_intensity_filter.Execute(image) 

    return image

def Normalization(image, x):
    MAX = 255
    MIN = 0
    image_array = sitk.GetArrayFromImage(image)
    upper = 200 + x
    lower = -200 + x
    image_array[image_array > upper] = upper
    image_array[image_array < lower] = lower

    image_result = sitk.GetImageFromArray(image_array)
    image_result.SetDirection(image.GetDirection())
    image_result.SetOrigin(image.GetOrigin())
    image_result.SetSpacing(image.GetSpacing())

    normalize_image_filter = sitk.NormalizeImageFilter()
    resacle_intensity_filter = sitk.RescaleIntensityImageFilter()
    resacle_intensity_filter.SetOutputMaximum(MAX)
    resacle_intensity_filter.SetOutputMinimum(MIN)

    image_result = normalize_image_filter.Execute(image_result)  
    image_result = resacle_intensity_filter.Execute(image_result) 

    return image_result

def InvertIntensityImage(sample):
    invert_intensity_img_filter = sitk.InvertIntensityImageFilter()
    image = invert_intensity_img_filter.Execute(sample['image'], 255)
    label = sample['label']

    return {'image': image, 'label': label}

def Resize(image, new_size, interpolator):
    dimension = image.GetDimension()
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    pixel_id_value = image.GetPixelIDValue()

    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(i - 1) * j if i * j > k else k for i, j, k in zip(size, spacing, reference_physical_size)]

    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [j / (i - 1) for i, j in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, pixel_id_value)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(direction)
    transform.SetTranslation(np.array(origin) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(size) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
  
    return sitk.Resample(image1 = image, 
                         referenceImage = reference_image, 
                         transform = centered_transform, 
                         interpolator = interpolator, 
                         defaultPixelValue = 0.0)

def ResampleSitkImage(sitk_image, spacing = None, interpolator = None, fill_value = 0):
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: 
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int) 
    new_size = [int(s) for s in new_size] 

    resample_image_filter = sitk.ResampleImageFilter()
    resample_image_filter.SetReferenceImage(sitk_image)
    resample_image_filter.SetSize(new_size)
    resample_image_filter.SetTransform(sitk.Transform())
    resample_image_filter.SetInterpolator(sitk_interpolator)
    resample_image_filter.SetOutputOrigin(orig_origin)
    resample_image_filter.SetOutputSpacing(new_spacing)
    resample_image_filter.SetOutputDirection(orig_direction)
    resample_image_filter.SetDefaultPixelValue(orig_pixelid)
    resampled_sitk_image = resample_image_filter.Execute(sitk_image)

    return resampled_sitk_image
