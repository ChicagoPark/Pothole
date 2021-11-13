# chicago_work

---

transfer learning : https://www.youtube.com/watch?v=WJZoywOG1cs&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=11

dataugmentation (spliting data) : https://blog.naver.com/siniphia/222118275636

early stopping : https://youtu.be/2UHCjhyNLKw

---


## [0] 실습 유용한 명령어

---

`코랩에 바로 구글드라이브 경로로만 데이터 가져오기`
```bash
https://drive.google.com/file/d/1X1J2RwBu9KhRpw8GqS-s1ZeI6YF0VZTO/view
! gdown --id 1X1J2RwBu9KhRpw8GqS-s1ZeI6YF0VZTO
```
<!-- 코랩 드라이브 가져오는 것 알
https://www.youtube.com/watch?v=Mq8-WdcnzVo
-->

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
## [절차3] : 모델 가져오고, 최적의 하이퍼파라미터 찾아내기

`모델 빌드하기`
```python
def build_model(hp):
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None)

    # 해당 층의 매개변수는 그대로 사용하기 위해서 freezing 을 한다.
    base_model.trainable = False
    inputs = keras.Input(shape=(320, 320, 3))
    x = base_model(inputs, training=False)
    #x = keras.layers.Dropout(0.2)(x)
    x=keras.layers.GlobalAveragePooling2D()(x)
    
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 5)):
        x=keras.layers.Dense(# Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "elu"]))(x)
    if hp.Boolean("dropout"):
        x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(3, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), # 교차엔트로피로 구성
              metrics=['accuracy'])
    
    return model
```

`keras_tuner 적용가능한지 확인하기`
```python
import keras_tuner as kt

build_model(kt.HyperParameters())
```

`파라미터 찾아나서기`
```python
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)
```
`설정한 것 확인하기`
```python
tuner.search_space_summary()
```

`찾아 나선다`
```python
tuner.search(train_data_gen,validation_data=val_data_gen, epochs=2)
```
`최종결과 도출`
```python
tuner.results_summary()
```



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
