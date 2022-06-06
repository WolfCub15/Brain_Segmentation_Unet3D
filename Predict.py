from tqdm import tqdm
import datetime
import math
from Unet import UNet
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


from Resample import *
from ImageDataSet import *
from Utils import *
from Padding import *
from Parameters import *
from DiceLoss import *

def BatchPrepare(image, ijk_patch_indices):
    image_batches = []
    #print("ijk indeces: {}".format(ijk_patch_indices))

    for batch in ijk_patch_indices:
        #print("batch: {}".format(batch))
        image_batch = []
        for patch in batch:
            #print("patch: {}".format(patch))

            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        # image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)
    return image_batches

def ReadFile(file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(file)
    image = reader.Execute()

    image = ImageNormalization(image)

    return image

def InferenceSingleImage( write_image, 
                          model, 
                          image_path, 
                          label_path, 
                          result_path, 
                          resample, 
                          resolution, 
                          patch_size_x, 
                          patch_size_y, 
                          patch_size_z, 
                          stride_inplane, 
                          stride_layer, 
                          batch_size, 
                          segmentation):
                
    transforms1 = [ Resample(resolution, resample) ]

    out_size = (patch_size_x, patch_size_y, patch_size_z)
    transforms2 = [ Padding(out_size) ]

    image = ReadFile(image_path)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    image = castImageFilter.Execute(image)

    # создать пустую метку в паре с преобразованным изображением
    label_tfm = sitk.Image(image.GetSize(), sitk.sitkFloat32)
    label_tfm.SetOrigin(image.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image.GetSpacing())

    sample = {'image': image, 'label': label_tfm}

    for transform in transforms1:
        sample = transform(sample)

    # отслеживание заполнения
    image_array = sitk.GetArrayFromImage(sample['image'])
    pad_x = patch_size_x - (patch_size_x - image_array.shape[2])
    pad_y = patch_size_y - (patch_size_y - image_array.shape[1])
    pad_z = patch_size_z - (patch_size_z - image_array.shape[0])

    image_pre_pad = sample['image']

    for transform in transforms2:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']

    # конвертация в numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)
    label_np = np.asarray(label_np, np.float32)
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    if segmentation is True:
        label_np = np.around(label_np)

    # заполнение изображения, если размер z еще не четный
    if (image_np.shape[2] % 2) == 0:
        Pad = False
    else:
        image_np = np.pad(image_np, ((0,0), (0,0), (0, 1)), 'edge')
        label_np = np.pad(label_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        Pad = True

    # будет использоваться весовая матрица для усреднения перекрывающейся области
    weight_np = np.zeros(label_np.shape)

    # подготавливаем индексы пакетов изображений
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                # для последнего патча
                if istart + patch_size_x > image_np.shape[0]:
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                # для последнего патча
                if jstart + patch_size_y > image_np.shape[1]: 
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                # для последнего патча
                if kstart + patch_size_z > image_np.shape[2]: 
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = BatchPrepare(image_np, ijk_patch_indices)
    
    for i in tqdm(range(len(batches))):
        batch = batches[i]
    
        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
        batch = Variable(batch.cpu())
        
        pred = model(batch)
        pred = pred.squeeze().data.cpu().numpy()

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[:, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))

    # устранить перекрывающуюся область, используя взвешенное значение
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

    if segmentation is True:
        label_np = abs(np.around(label_np))

    # removed the 1 pad on z
    if Pad is True:
        label_np = label_np[:, :, 0:(label_np.shape[2]-1)]

    # removed all the padding
    label_np = label_np[:pad_x, :pad_y, :pad_z]

    # convert back to sitk space
    label = GetImage(label_np, image_pre_pad.GetSpacing(), image_pre_pad.GetDirection(), image_pre_pad.GetOrigin())

    # сохранить метку
    writer = sitk.ImageFileWriter()

    if resample is True:
        print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
        # label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='bspline')   # keep this commented
        if segmentation is True:
            label = Resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkLinear)
            label_array = np.around(sitk.GetArrayFromImage(label))
            label = sitk.GetImageFromArray(label_array)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())
        else:
            label = Resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkBSpline)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())
    else:
        label = label

    reader = sitk.ImageFileReader()
    reader.SetFileName(label_path)
    true_label = reader.Execute()
    true_label = sitk.GetArrayFromImage(true_label)
    true_label[true_label > 0] = 1

    predicted = sitk.GetArrayFromImage(label)

    dice = DiceCoeff(predicted, true_label)

    writer.SetFileName(result_path)

    if write_image is True:
        writer.Execute(label)
        print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path))

    return label, dice


if __name__ == "__main__":
    FreeGpuCache()
    
    torch.cuda.set_device(0)
    net = UNet(residual='pool').cpu()

    net.load_state_dict(torch.load(WEIGHTS_PATH), strict = False)
    net.eval()

    test_list = CreateListFromPath(IMAGES_VALIDATION_PATH, LABELS_VALIDATION_PATH)

    print("Size: ", len(test_list))
    result_label = []
    dices = []
    result_dice_path = os.path.join(HISTORY_PATH, "ResultDice")
    CheckDir(result_dice_path)

    for i in range(len(test_list)):
        data = test_list[i]
        image = data['image']
        label = data['label']
        result_path = "./Result/predict_result_{}.nii".format(i)


        label, dice = InferenceSingleImage(write_image = WRITE_IMAGE_FLAG, 
                                                        model = net, 
                                                        image_path = image, 
                                                        label_path = label, 
                                                        result_path = result_path, 
                                                        resample = TEST_RESAMPLE_FLAG, 
                                                        resolution = TEST_NEW_RESOLUTION,
                                                        patch_size_x = TEST_PATCH_SIZE[0],
                                                        patch_size_y = TEST_PATCH_SIZE[1],
                                                        patch_size_z = TEST_PATCH_SIZE[2], 
                                                        stride_inplane = STRIDE_INPLANE, 
                                                        stride_layer = STRIDE_LAYER,
                                                        batch_size = TEST_BATCH_SIZE, 
                                                        segmentation = SEGMENTATION_FLAG)

        print("Dice = ", dice)

        dice_string = "{} )  Dice: {:.8f}".format(i, dice)

        print(dice_string)
        open(os.path.join(result_dice_path, RESULT_DICE_FILE), 'a').write(dice_string + '\n')


