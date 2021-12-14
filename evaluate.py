import tensorflow as tf
import glob
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)

args = parser.parse_args()

# Receive data directory paths
input_dir = args.input_dir

# Take input_dir as a factor and bring up the test dataset
IMG_HEIGHT = 320
IMG_WIDTH = 320
# Set the input data path
base_path = os.path.abspath(input_dir)
# Use ImageDataGenerator that makes it easy to retrieve data from a set folder.
test_image_generator = ImageDataGenerator(rescale=1./255)
# The target size is set according to the model input, and since it is one-hot encoding, set it to 'categorical'.
test_data_gen = test_image_generator.flow_from_directory(batch_size=1,
                                                        directory=base_path,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='categorical')

model_paths = glob.glob(os.path.join("saved_models", "*.h5"))
with open("result.txt", "a") as f:
    f.write(f"model_path\taccuracy\n")
    cnt = 0
    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        result = model.evaluate(test_data_gen, return_dict=True)
        accu = result["accuracy"]
        print(f"{model_path}\t{accu}")
        f.write(f"{model_path}\t{accu}\n")
        if cnt >= 9:
            break
        cnt += 1
