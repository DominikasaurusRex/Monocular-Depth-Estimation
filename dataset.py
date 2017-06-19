import tensorflow as tensorflow
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

GRAYSCALE = False
NUMBER_OF_THREADS = 4
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_trainingbatches_from_csv(self, csv_file_path):
        filename_queue = tensorflow.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tensorflow.TextLineReader()
        
        _, serialized_example = reader.read(filename_queue)
        filename_original, filename_depth = tensorflow.decode_csv(serialized_example, [["path"], ["annotation"]])
        
        # original Images
        image_original_file = tensorflow.read_file(filename_original)
        if GRAYSCALE:
            image_original = tensorflow.image.decode_jpeg(image_original_file, channels=1)  # in gray
        else:
            image_original = tensorflow.image.decode_jpeg(image_original_file, channels=3)  # in rgb
        image_original = tensorflow.cast(image_original, tensorflow.float32)    
           
        # Depth Maps
        image_depth_file = tensorflow.read_file(filename_depth)
        image_depth = tensorflow.image.decode_png(image_depth_file, channels=1)
        image_depth = tensorflow.cast(image_depth, tensorflow.float32)
        # image_depth = tensorflow.cast(image_depth, tensorflow.int64)
        image_depth = tensorflow.div(image_depth, [255.0])  # create 0-1 instead of 0-255
        
        # resize
        image_original = tensorflow.image.resize_images(image_original, (IMAGE_HEIGHT, IMAGE_WIDTH))
        image_depth = tensorflow.image.resize_images(image_depth, (TARGET_HEIGHT, TARGET_WIDTH))
        depth_map_signum = tensorflow.sign(image_depth)
        
        # generate batch
        original_images, depth_maps, depth_maps_signums = tensorflow.train.batch(
            [image_original, image_depth, depth_map_signum],
            batch_size=self.batch_size,
            num_threads=NUMBER_OF_THREADS,
            capacity=50 + 3 * self.batch_size,
        )
        return original_images, depth_maps, depth_maps_signums
    

def output_predict_into_images(predictions, originals, groundtruths, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (original, prediction, groundtruth) in enumerate(zip(originals, predictions, groundtruths)):
        # original      
        if GRAYSCALE:    
            original = original.transpose(2, 0, 1)
            original_pil = Image.fromarray(np.uint8(original[0]), mode="L")
        else:
            original_pil = Image.fromarray(np.uint8(original))
        original_name = "%s/%05d_org.png" % (output_dir, i)
        original_pil.save(original_name)
        
        # ground truth
        groundtruth = groundtruth.transpose(2, 0, 1)
        groundtruth_transposed = groundtruth * 255.0
        groundtruth_pil = Image.fromarray(np.uint8(groundtruth_transposed[0]), mode="L")
        groundtruth_name = "%s/%05d_ground.png" % (output_dir, i)
        groundtruth_pil.save(groundtruth_name)
        
        # prediction        
        prediction_transposed = prediction.transpose(2, 0, 1)
        # print(prediction)
        # print("after Transpose", prediction_transposed)
        if np.max(prediction_transposed) == 0:
            print(" !!!ERROR!!!: Maximum Depth is 0. Black Picture")
        prediction_transposed = (prediction_transposed / np.max(prediction_transposed)) * 255.0
        prediction_pil = Image.fromarray(np.uint8(prediction_transposed[0]), mode="L")
        prediction_name = "%s/%05d.png" % (output_dir, i)
        prediction_pil.save(prediction_name)
