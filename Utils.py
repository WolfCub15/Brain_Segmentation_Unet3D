import os
import torch
import numpy as np
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

from DiceLoss import * 
from Predict import *
from DiceLoss import *
from Utils import *
from NiftiDataset import *
from AverageMeter import *
from Resample import *
from DataAugmentation import *
from Padding import *
from RandomCrop import *

def CheckDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def FreeGpuCache():
    print("GPU Usage")
    gpu_usage()                             
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

if __name__ == "__main__":
    import numpy as np
    yt = np.random.random(size=(2, 1, 3, 3, 3))
    #print(yt)
    yt = torch.from_numpy(yt)
    yp = np.zeros(shape=(2, 1, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp)
    #print(yp)
    dl = BinaryDiceLoss()
    print(dl(yp, yt).item())