# chicago_work

---

transfer learning : https://www.youtube.com/watch?v=WJZoywOG1cs&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=11

dataugmentation (spliting data) : https://blog.naver.com/siniphia/222118275636

early stopping : https://youtu.be/2UHCjhyNLKw

---


## [0] 실습 유용한 명령어

---

## [절차1] : 데이터 불러오기

```python
base_path = os.path.abspath("실차 데이터")
```
## [절차2] : 학습데이터 검증데이터 분리 및 Data Augmentation 변수 할당


#### `SEED 를 활용한 데이터 분리 : seed 를 활용하여 train set 과 val set 의 augmentation 적용을 분리할 수 있는 것이다.`
```python
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)    

train_generator = datagen_train.flow_from_directory(
    data_root,
    seed=42,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    shuffle=True,
    subset='training')

val_generator = datagen_val.flow_from_directory(
    data_root,
    seed=42,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    shuffle=True,
    subset='validation')
```
## [절차3] : 모델 가져오기



## [절차4] : 모델 훈련 시키기


## [절차5] : 결과 시각화하기


## [절차6] : Test 진행하기

`시각화`

```python
# [1] 모델 시각화
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
# [2] 모델 요약
model.summary()
```
<!--
https://codetorial.net/tensorflow/visualize_model.html
-->
---

---

`기본 모델링`
```python

```
---

# 고찰

딥러닝 모델 훈련 시 바로 전체 데이터를 넣으면 오랜시간 기다린 후에 에러가 발생하여 효율이 매우 떨어져서

```python
x = tf.random.normal(shape=(5, 320, 320, 3))
y = tf.constant([0, 1, 2, 3, 4])
```

---

제출파일 : json_parsing.ipynb

---
