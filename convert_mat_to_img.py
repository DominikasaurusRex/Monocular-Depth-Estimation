import os
import numpy as math_library
import h5py 
from PIL import Image as image_library
#import random


def convert_nyu_dataset_into_images_and_csv(path):
    print("loading dataset: %s" % (path))
    nyu_dataset_file  = h5py.File(path)


    trainings_material = []
    #create the images and depth maps
    #also add all the paths to trainingsmaterial
    for i, (image_original, image_depth) in enumerate(zip(nyu_dataset_file['images'], nyu_dataset_file['depths'])):
        image_original_transpose = image_original.transpose(2, 1, 0)
        image_depth_transpose = image_depth.transpose(1, 0)
        image_depth_transpose = (image_depth_transpose/math_library.max(image_depth_transpose))*255.0
        image_original_pil = image_library.fromarray(math_library.uint8(image_original_transpose))
        image_depth_pil = image_library.fromarray(math_library.uint8(image_depth_transpose))
        image_original_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (i))
        image_original_pil.save(image_original_name)
        image_depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (i))
        image_depth_pil.save(image_depth_name)

        trainings_material.append((image_original_name, image_depth_name))

    #random.shuffle(trainings_material) #we should probably keep the same first trainingsmaterials
    
    write_csv_file(trainings_material)

def write_csv_file(trainingsmaterial):
    with open('train.csv', 'w') as output:
        for (image_original_path, image_depth_path) in trainingsmaterial:
            output.write("%s,%s" % (image_original_path, image_depth_path))
            output.write("\n")



if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_dataset_path = 'data/nyu_depth_v2_labeled.mat'
    convert_nyu_dataset_into_images_and_csv(nyu_dataset_path)