import time
import os
import torch
from zmq import device
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from DiceLoss import *
from GaussianFilter import Processing
from Utils import *
from Parameters import *
from ImageDataSet import *
from Unet import UNet
from AverageMeter import *
from Resample import *
from DataAugmentation import *
from Padding import *
from RandomCrop import *
from AdaptiveHistogramEqualization import *

def TestEpoch(net, loader, loss_function, scheduler):
    net.eval()
    valid_loss = AverageMeter()
    valid_dice = AverageMeter()

    with torch.no_grad():
      for i_batch, (data, label) in enumerate(loader):
          if torch.cuda.is_available():
            data = Variable(data.cuda())
            label = Variable(label.cuda())
          else:
            data = Variable(data.cpu())
            label = Variable(label.cpu())

          label[label > 0] = 1


          out = net(data)

          #out_tmp = out 
          #out_tmp = out_tmp.squeeze().data.cpu().numpy()
          #label_tmp = label.squeeze().cpu().numpy()

          #loss = loss_function(out, label)
          #valid_loss.update(loss.item())

          out = out.squeeze().data.cpu().numpy()
          out[np.nonzero(out < 0.5)] = 0.0
          out[np.nonzero(out >= 0.5)] = 1.0
          label = label.squeeze().cpu().numpy()
          dice = DiceCoeff(out, label)
          valid_dice.update(dice)
          #print("Test: {}  |  Dice: {:.4f}  |  Loss: {:.4f}".format(i_batch, valid_dice.val, valid_loss.val))
          #print("Test: {}  |  Dice: {:.4f} ".format(i_batch, valid_dice.val))

          information_string = ("Test: {}  |  Dice: {:.4f} ".format(i_batch, valid_dice.val))
          print(information_string)
          open(os.path.join(OUTPUT_PATH, OUTPUT_FILE), 'a').write(information_string + '\n')

      scheduler.step(valid_dice.avg)

    return valid_dice.avg, valid_loss.avg


def TrainEpoch(net, data_loader, optimizer, loss_function):
    net.train()
    train_loss = AverageMeter()
    train_dice = AverageMeter()

    for i_batch, (data, label) in enumerate(data_loader):
        #переменная является оберткой для тензора
        if torch.cuda.is_available():
            data = Variable(data.cuda())
            label = Variable(label.cuda())
        else:
            data = Variable(data.cpu())
            label = Variable(label.cpu())

        label[label > 0] = 1

        # передаем данные в сеть
        out = net(data) 

        #print(label)

        out_tmp = out 
        out_tmp = out_tmp.squeeze().data.cpu().numpy()
        label_tmp = label.squeeze().cpu().numpy()
        out_tmp[np.nonzero(out_tmp < 0.5)] = 0.0
        out_tmp[np.nonzero(out_tmp >= 0.5)] = 1.0
        dice = DiceCoeff(out_tmp, label_tmp)
        train_dice.update(dice)

        # оцениваем функцию затрат
        loss = loss_function(out, label)
        train_loss.update(loss.item())

        # обратное распространение
        loss.backward()
        optimizer.step()
        # устанавливаем градиенты в 0 
        optimizer.zero_grad() 

        #print("Train Batch: {}  |  Loss: {:.4f}  |  Training Dice: {:.4f}".format(i_batch, train_loss.val, train_dice.val))
        information_string = ("Train Batch: {}  |  Loss: {:.4f}  |  Training Dice: {:.4f}".format(i_batch, train_loss.val, train_dice.val))
        print(information_string)
        open(os.path.join(OUTPUT_PATH, OUTPUT_FILE), 'a').write(information_string + '\n')

    return train_dice.avg, train_loss.avg


def TrainModel():
    history_path = HISTORY_PATH
    Checkpoint_path = os.path.join(HISTORY_PATH, "Checkpoint")
    log_path = os.path.join(HISTORY_PATH, "Log")
    result_path = os.path.join(HISTORY_PATH, "Result")

    metric = 0

    min_pixel = int(MIN_PIXEL * ((PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]) / 100))

    CheckDir(history_path)
    CheckDir(log_path)
    CheckDir(Checkpoint_path)
    CheckDir(result_path)

    train_list = CreateListFromPath(IMAGES_TRAIN_PATH, LABELS_TRARIN_PATH)
    validation_list = CreateListFromPath(IMAGES_VALIDATION_PATH, LABELS_VALIDATION_PATH)
    test_list = CreateListFromPath(IMAGES_TEST_PATH, LABELS_TEST_PATH)

    #дополняем список данных для обучения
    for i in range(INCREASE_FACTOR_DATA):
        train_list.extend(train_list)
        validation_list.extend(validation_list)
        test_list.extend(test_list)

    print("Train: {}  |  Validation: {}  |  Test: {}".format(len(train_list), len(validation_list), len(test_list)))

    #train_list = Processing(train_list)
    #validation_list = Processing(validation_list)

    train_transforms = [
        Resample(NEW_RESOLUTION, RESAMPLE_FLAG),
        DataAugmentation(),
        Padding((PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2])),
        RandomCrop((PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2]), DROP_RATIO, min_pixel),
    ]

    test_transforms = [
        Resample(NEW_RESOLUTION, RESAMPLE_FLAG),
        Padding((PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2])),
        RandomCrop((PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2]), DROP_RATIO, min_pixel),
    ]
    # определить набор данных и загрузчик
    train_dataset = ImageDataset(train_list, transforms = train_transforms, train = True)
    validation_dataset = ImageDataset(validation_list, transforms = test_transforms, test = True)
    test_dataset = ImageDataset(test_list, transforms = test_transforms, test = True)

    '''hist = AdaptiveHistogramEqualization()
    #den = Denoising()

    for i in range(len(train_dataset)):
        train_dataset[i] = hist(train_dataset[i])

    for i in range(len(train_dataset)):
        validation_dataset[i] = hist(validation_dataset[i])

    for i in range(len(train_dataset)):
        test_dataset[i] = hist(test_dataset[i])
    '''
    print("Train set : {}  |  Validation set : {}  |  Test set : {}".format(len(train_dataset), len(validation_dataset), len(test_dataset)))

    # передаем в сеть с определенным размером пакета
    train_data_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)  
    validation_data_loader = DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_data_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    print("Train data: {}  |  Validation data: {}  |  Test data: {}".format(len(train_data_loader), len(validation_data_loader), len(test_data_loader)))

    if torch.cuda.is_available():
      torch.cuda.set_device(0)
      net = UNet(residual='pool').cuda()
    else:
      net = UNet(residual='pool').cpu()


    # загружаем веса сети
    if WEIGHTS_FLAG is True:
        net.load_state_dict(torch.load(LOAD_WEIGHTS_PATH))

    # реализует стохастический градиентный спуск 
    #optimizer = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: ((1 - (epoch / 1000) ** 0.9)) / 10)
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    # определяем функцию потерь
    loss_function = DiceLoss() 
    best_dice_value = 0.

    # определяем номер эпохи
    for epoch in range(EPOCH_INIT, EPOCH_N): 
        #lr = LEARNING_RATE * (0.5 ** (epoch // 4))
        #lr = (epoch / 1000) ** 0.99
        #for param_group in optimizer.param_groups:
        #    param_group["lr"] = lr

        #torch.set_grad_enabled(True)
        
        time_epoch_start = time.time()

        dice_train, train_loss  = TrainEpoch(net, train_data_loader, optimizer, loss_function)                       
        dice_validation, validation_loss = TestEpoch(net, validation_data_loader, loss_function, scheduler)
        #dice_test = TestEpoch(net, test_data_loader)

        time_epoch_end = time.time()
        time_epoch = (time_epoch_end - time_epoch_start) / 60

        #metrics = dice_validation
        #scheduler.step(metric)
        #lr = optimizer.param_groups[0]['lr']

        #information_string = "Epoch: {}  |  Dice Loss: {:.4f}  |  Time(min): {:.4f}  |  Validation Dice: {:.4f}  |  Testing Dice: {:.4f}".format(
         #   epoch, dice_loss, time_epoch, dice_validation, dice_test
        #)

        #information_string = "Epoch: {}  |  Time(min): {:.4f}  |  Train Dice: {:.4f}  |  Dice Loss: {:.4f}  |  Validation Dice: {:.4f}  | Validation Loss: {:.4f}".format(
        #   epoch, time_epoch, dice_train, train_loss, dice_validation, validation_loss
        #)

        information_string = "Epoch: {}  |  Time(min): {:.4f}  |  Train Dice: {:.4f}  |  Dice Loss: {:.4f}  |  Validation Dice: {:.4f}".format(
           epoch, time_epoch, dice_train, train_loss, dice_validation
        )

        print(information_string)
        open(os.path.join(log_path, LOG_FILE), 'a').write(information_string + '\n')

        torch.save(net.state_dict(), os.path.join(Checkpoint_path, "Network_{}.pth.gz".format(epoch)))

        if dice_validation > best_dice_value:
            best_dice_value = dice_validation
            torch.save(net.state_dict(), os.path.join(Checkpoint_path, "Best_Dice.pth.gz"))


def CheckAccuracyModel():
    torch.cuda.set_device(0)
    net = UNet(residual='pool').cuda()

    net.load_state_dict(torch.load(LOAD_BEST_DICE_PATH))

    train_list = CreateListFromPath(IMAGES_TRAIN_PATH)
    validation_list = CreateListFromPath(IMAGES_VALIDATION_PATH)
    test_list = CreateListFromPath(IMAGES_TEST_PATH)


if __name__ == '__main__':
    FreeGpuCache()    
    torch.cuda.memory_summary(device = None, abbreviated = False)    
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    print("Using {} device".format(device))

    TrainModel()
