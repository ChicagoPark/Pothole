import tensorflow as tf
import glob
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)

args = parser.parse_args()

# 데이터 디렉터리 경로를 받아오는 코드
input_dir = args.input_dir

# input_dir 을 인자로 받아 Test Dataset 불러오기
IMG_HEIGHT = 320
IMG_WIDTH = 320
# 입력받은 데이터 경로 설정 (자세한 사항 - 보고서의 이미지 로딩과 동일)
base_path = os.path.abspath(input_dir)
# 설정된 폴더에서 데이터를 손 쉽게 불러올 수 있는 ImageDataGenerator 를 활용
test_image_generator = ImageDataGenerator(rescale=1./255)
# 모델 입력에 맞게 타겟 사이즈를 설정 하고, 웟핫잇코딩이므로 categorical 로 설정
test_data_gen = test_image_generator.flow_from_directory(batch_size=1,
                                                        directory=base_path,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='categorical')

model_paths = glob.glob(os.path.join("saved_models", "*.h5"))
with open("result.txt", "a") as f:
    f.write(f"model_path\taccuracy\n")
    #cnt = 0
    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        # test_img 와 test_label 은 직접 불러올 것
        result = model.evaluate(test_data_gen, return_dict=True)
        accu = result["accuracy"]
        print(f"{model_path}\t{accu}")
        f.write(f"{model_path}\t{accu}\n")
        if cnt >= 9:
            break
        cnt += 1
