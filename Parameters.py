DATASET_PATH = "./dataset2019/archive/training/"
DATASET_LABELS_PATH = "./Dataset/labels"
DATASET_IMAGES_FLAIR_PATH = "./Dataset/images_flair"
DATASET_IMAGES_T1CE_PATH = "./Dataset/images_t1ce"
DATASET_IMAGES_T1_PATH = "./Dataset/images_t1"
DATASET_IMAGES_T2_PATH = "./Dataset/images_t2"
IMAGES_TRAIN_PATH = './Folders/images/train'
IMAGES_VALIDATION_PATH = './Folders/images/validation'
IMAGES_TEST_PATH = './Folders/images/test'
LABELS_TARIN_PATH = './Folders/labels/train'
LABELS_VALIDATION_PATH = './Folders/labels/validation'
LABELS_TEST_PATH = './Folders/labels/test'
HISTORY_PATH = './History'
LOG_PATH = './History/Log/'
LOAD_BEST_DICE_PATH = './History/Checkpoint/Best_Dice.pth.gz'
LOG_FILE = 'train_log.txt'

TEST_IMAGE = './Folders/images/test/image3.nii'
TEST_LABEL = './Folders/labels/test/label3.nii'
RESULT_PATH = './Result/result_3.nii'
RESULT_FILE_PATH_NAME = './Result/result_'
WEIGHTS_PATH = './History/Checkpoint/Best_Dice.pth.gz'

WEIGHTS_FLAG = True
RESAMPLE_FLAG = True
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 1e-4
INCREASE_FACTOR_DATA = 3
NEW_RESOLUTION = (0.6, 0.6, 2.2)
PATCH_SIZE = [16, 16, 16]
DROP_RATIO = 0 
MIN_PIXEL = 0.1
BATCH_SIZE = 4
EPOCH_INIT = 45
EPOCH_COUNT = 100
EPOCH_N = EPOCH_INIT + EPOCH_COUNT
STRIDE_INPLANE = 64
STRIDE_LAYER = 16 
LOAD_WEIGHTS_PATH = "./History/Checkpoint/Network_{}.pth.gz".format(EPOCH_INIT-1)


TEST_RESAMPLE_FLAG = False
TEST_NEW_RESOLUTION = (1, 1, 2)
TEST_PATCH_SIZE = [64, 64, 16]
TEST_BATCH_SIZE = 1
SEGMENTATION_FLAG = True
PATCH_SIZE_FOR_CHECK = [256, 256, 256]
BATCH_SIZE_FOR_CHECK = 1
