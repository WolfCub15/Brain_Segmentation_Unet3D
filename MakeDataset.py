import glob
import os
import shutil
from tqdm import tqdm

def MakeDataset(data, out_path, data_type):
    for i, file_name in tqdm(enumerate(data), total=len(data)):
        file_types = glob.glob(file_name)
        output_dir = "{}/".format(out_path)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for type in file_types:
            if data_type in type:
                output_path = "{}".format(output_dir)
                shutil.copy(type, output_path)


def main():
    data_path = "./dataset2019/archive/training/"

    #out_path = "./Dataset/labels"
    #out_path = "./Dataset/images_flair"
    #out_path = "./Dataset/images_t1"
    #out_path = "./Dataset/images_t2"
    out_path = "./Dataset/images_t1ce"

    #data_type = 'seg'
    #data_type = 'flair'
    #data_type = 't1'
    #data_type = 't2'
    data_type = 't1ce'

    HGG_data = glob.glob(data_path + "HGG/*/*")
    LGG_data = glob.glob(data_path + "LGG/*/*")

    MakeDataset(HGG_data, out_path, data_type)
    MakeDataset(LGG_data, out_path, data_type)


if __name__ == "__main__":
    main()