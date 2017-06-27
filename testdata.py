import tensorflow as tensorflow
from tensorflow.python.platform import gfile as directory_handler
import numpy as math_library
from PIL import Image as image_library

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74
        
def load_test_image(file_path):
        
        # original Images
        image_original_file = tensorflow.read_file(file_path)
        image_original = tensorflow.image.decode_jpeg(image_original_file, channels=3)  # in rgb
        image_original = tensorflow.cast(image_original, tensorflow.float32)    
        
        # resize
        image_original = tensorflow.image.resize_images(image_original, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        return image_original
    
    
def output_test_predictions_into_images(predict_test_image, test_image, output_dir):
    print("output test into %s" % output_dir)
    create_output_directory(output_dir)
    
    # original      
    original_pil = image_library.fromarray(math_library.uint8(test_image))
    original_name = "%s/testPicture.png" % (output_dir)
    original_pil.save(original_name)
            
    # prediction 
    prediction_transposed = predict_test_image[0].transpose(2, 0, 1)
    prediction_transposed = (prediction_transposed / math_library.max(prediction_transposed)) * 255.0
    prediction_pil = image_library.fromarray(math_library.uint8(prediction_transposed[0]), mode="L")
    prediction_name = "%s/prediction.png" % (output_dir)
    prediction_pil.save(prediction_name)


def create_output_directory(output_dir):
    if not directory_handler.Exists(output_dir):
        directory_handler.MakeDirs(output_dir)
