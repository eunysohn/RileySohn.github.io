layout: post
title: "Storage"
date: 2021-10-26 15:20:00 -0000
categories: Infra


Loss function : measures how good the current 'guess' is

Dense: a layer of connected neurons

Convergence: the process of getting very close to the correct answer

Optimizer: generates a new and improved guess



Google Colab: hosted Jupiter notebook service,

​                          for free

​                          stored in google drive

[Google colab]: https://colab.research.google.com

https://research.google.com/seedbank

https://fresearch.google.com/notebook/welcome

[Google Colab faq]: https://research.google.com/colaboratory/faq.html



## Week2. Introduction to Computer Vision

Fashion MNIST

[Fashion MNIST]: https://github.com/zalandoresearch/fashion-mnist

- data loading

```
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()
```

-  neural network train 시 data 나눠서 일부는 train 용, 일부는 validation 용으로 사용

​        : to test a network with previously unseen data

60,000 images for train

 10,000 images for test

- labeling은 숫자로
   -> bias 예방, 다른 언어 사용자 이해

[bias와 관련 기술]: https://ai.google/responsibilities/responsible-ai-practices/



```python
# important : 1st, 3rd layers
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
    // 28 x 28 turns flatten to simple linear layer
	keras.layers.Dense(128, activation=tf.nn.relu),
    // middle layer: hidden layer, 128 neurons
	keras.layers.Dense(10, activation=tf.nn.softmax)
    // 10 neurons because 10 classes
])
```

cf)

[andreu in youtube, neural network overview]: https://youtu.be/fXOsFF95ifk

matplotlib

normalize data

.34 -> .66 acuracy

- normalizing

neural nw 트레이닝 할 때는 모든 value가 0과 1 사이 값인 게 좋음

 : 0 ~255 사이 값으로 이루어진 set 0 ~ 1로 변경 -> normalizing,

   list도 loop없이 아래 처럼 나눌 수 있음

```python
training_images = training_images / 255.0 
```

   : normalize 안시키면 loss가 굉장히 커짐

- model design

```python
model = tf.keras.model.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                   # 128 -> 1024로 증가시켰더니 loss율 낮아짐
                                   # training takes longer, but is more accurate, but not always 'more is better'
                                   # 128 위에 512 layer 하나 추가했더니 128->1024와 같은 loss율
                                   tf.keras.lyaers.Dense(10, activation=tf.nn.softmax)
                                   # last layer's neuron number = number of classes classifying for
                                 ]) 
```

   . Sequential: That defines a SEQUENCE of layers in the neural network

   . Flatten: Flatten takes the square and turns it into a 1 dimensional set

   . Dense: Adds a layer of neurons.

​                  Each layer of neurons need an activation function to tell them what to do. There's lots of options. 

​      .Relu: If X>0 return X, else return 0. it only passes values 0 or greater to the next layer in the network

​      .Softmax: takes a set of values, and effectively picks the biggest one. 

  <개선 point들>

1. neuron 수 증가
2. Dense layer 추가
3. epochs 수 증가
   - Try 15 epochs - you'll probably get a model with a much better loss than the one with 5
   - Try 30 epochs - you might see the loss value stops decreasing, and sometimes increases.
   - side effect called 'overfitting'

- compile model

```python
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
```

 at the end of training, there is the accuracy value. 0.9098 means your neural network is about 91% accurate in classifying the training data.

- test model

```python
model.evaluate(test_images, test_labels)
```

- model predict

```python
classifications = model.predict(test_images)
print(classifications[0])
```

  -> [3.3885094e-06 1.9757925e-07 2.1538317e-07 3.5735017e-08 3.1178701e-07 1.1117136e-03 4.0823693e-07 4.4879936e-02 8.4565276e-05 9.5391917e-01]

  it's the probability that this item is each of the 10 classes



- overfitting 전에 원하는 값 나오면 training 멈추기

​      : callbacks

```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.6): 
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
      
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                   ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```

 cf) callbacks example [notebook](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_2_callbacks.ipynb)

​      MNIST http://yann.lecun.com/exdb/mnist/



## Week3. Enhancing Vision with Convolutional Neural Networks

- DL에서 convolution : they narrow down the content of the image to focus on specific, distinct, details
                                     filter 사용과 비슷

- image마다 불필요한 공간 있음

- filter 적용해서 emphasize (edge detection). highlighted 된 부분만 학습
- Add some layers to do convolution before you have the dense layers, and then the information going to the dense layers is more focussed, and possibly more accurate.
- pooling : 이미지 압축    ex) 16pixel -> 4pixel
- Layers
  - Conv2D layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
  - MaxPooling2D layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
  - convolutional Layer 적용 예시

```python
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# 1st convolution expects a single tensor containing everything
# 60,000 28x28x1 items in a list -> single 4D list (60,000x28x28x1)
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# define model
model = tf.keras.models.Sequential([
  # specifying 1st convolution, 
  # 64개의 3x3 filters(convolutions) generate-32배수로 적어지면 accuracy 떨어짐, 3x3 grid size의 convolution. activation은 relu - 0보다 적은 값 제거, input_shape은 이전과 동일(28x28), 1: tallying using a single byte for color depth (grayscale이라 1 byte만 필요)
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
  # compress image. convolution으로 하이라이트된 content는 보존
  # pool layer, pick max value 2x2 pool -> every 4 pixels, the biggest value will survive
  tf.keras.layers.MaxPooling2D(2,2),
  # another 2 layers, so nw can learn another set of convolutions on top of the existing one and then again, pool to reduce the size
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # 1/4, 1/4 한 값, flatten 전에 값이 많이 smaller 됨
  # 위 convolution 진행한 값 flatten
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train
model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```

[convolution 관련 좋은 자료]: https://bit.ly/2UGa7uH	"응교수님 강의 시리즈 about 2d, 3d 이미지 처리. 겁나 김"

model.summary 결과값 (convolution 작업한 데이터)

Model: "sequential_1" 

_________________________________________________________________

 Layer (type)                 Output Shape              Param #    ================================================================= 

conv2d (Conv2D)              (None, 26, 26, 64)        640        

_________________________________________________________________

 max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0          

_________________________________________________________________

 conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928      

_________________________________________________________________

 max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0          

_________________________________________________________________

 flatten_1 (Flatten)          (None, 1600)              0          

_________________________________________________________________

dense_2 (Dense)              (None, 128)               204928     

_________________________________________________________________

dense_3 (Dense)              (None, 10)                1290       ================================================================= 

Total params: 243,786 Trainable params: 243,786 Non-trainable params: 0



1) convolution 수 변경하면 -> 값이 작아지면 accuracy 떨어짐
   Conv2D(이 값, (3,3), activation='relu', input_shape=(28,28,1))
2) convolution을 줄이면? -> accuracy 더 좋아지는데..
   convolution 2번, 32convolution 일 때 & epoch 5회 : 92% accuracy
   flatten되는 값이 훨씬 많긴 함 (1회: 5408)
   convolution 1번, 32convolution일 때 & epoch 5회: 93.98% accuracy & test acc: 98.58%
3) convolution 추가하면? 
   3번 했더니 accuracy 더 떨어짐. 걸리는 시간은 다 비슷하게 낮음. accuray : 88%
4) Remove all Convolutions but the first. What impact do you think this will have? 
    빨라짐. 
5) callback 적용해보기



quiz)

1) what is a convolution? a technique to isolate features in images

2) what is pooling? a technique to reduce the information in an image while maintaining

3) How do convolutions improve image recognition? they isolate features in images

4) applying convolutions on top of our deep neural network will make training:
   It depends on many factors. It might make your training faster or slower, and a poorly desinged convolutional layer may even be less efficient than a plain DNN.

5) after passing a 3x3 filter over a 28x28 image, how big will the output be?
    : 26 x 26

6) after max pooling a 26x26 image with a 2x2 filter, how bit will the output be?

   : 13x13







## Week4. Using Real-world Images

convolutions을 사용하여 efficient, accurate를 높일 수 있으나, 28x28, grayscale, subject is centered만 가능

real-world는? 더 크고 무겁고 중심에 물체가 있지 않은 그림을 control. horses & humans 분류 (ai가 generate한 image 사용)

실제 self-driving car에서 convolutional neural networks 사용

ex) 

[africa에서 Cassava 병에 걸린 crop 찾는]: https://www.youtube.com/watch?v=NlpS-DhayQA

- Understanding ImageGenerator

 : real-world 이미지는 subject가 여러 개 있거나 여러 위치에 있고 사이즈도 다양함. 그런 이미지를 다루기 위해서는 직접 이미지를 구하고 labeling해야 함. 이미지 가져다 디렉토리에 넣고 코드로 처리하면, tensorflow ImageGenerator 사용해서 training용 이미지와 test용 이미지 generate 가능. 각 디렉토리에 label과 함께 generate해 줌

```python
from tensorflow.keras.preprocessing.image
import ImageDataGenerator
# instantiating ImageGenerator. pass rescale to normalize the data
train_datagen = ImageDataGenerator(rescale=1./255)
# call the flow from directory
# always point the directory contains sub-directories. the names of the sub-directories will be the labels for your images
train_generator = train_datagen.flow_from_directory(
  train_dir,
  # images might come in all shapes and sizes, bu for training a neural nw, the input data all has to be the same size, so the images will need to be resized to make them consistent. with this code, the images are resized as they're loaded.
  target_size=(300, 300), 
  # calculating batch size는 이번 강의 범위 아님
  batch_size=128,
  # picks between two different things(horses, humans)
  class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  target_size=(300, 300),
  batch_size=32,
  class_mode='binary'
)
```



- Defining a ConvNet to use complex images
  model for classifying humans or horses

```python
model = tf.keras.models.Sequential([
  # 3 sets of convolution pooling layers: higher complexity. 3: color images (red, green, blue - 24-bit color pattern)
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
  # 298x298 x 16
  tf.keras.layers.MaxPooling2D(2, 2),
  # 149x149 x 16
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  # 147x147 x 32
  tf.keras.layers.MaxPooling2D(2, 2),
  # 73x73 x 32
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  # 71x71 x 64
  tf.keras.layers.MaxPooling2D(2, 2),
  # 35x35 x 64 (78,400 data)
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  # sigmoid: binary classification
  # 1 neuron for 2 classes bc. different activation. softmax 사용한다면 2개 neuron으로 해야	
  tf.keras.layers.Dense(1, activation='sigmoid')
])
```



- Training the ConvNet with fit_generator
  : model train 때 model.fit 아니라 model.fit_generator 사용

```python
from tensorflow.keras.optimizers import RMSprop

# compile은 loss function과 optimizer 정의
# 10 classes 일 때는 sparse_categorical_crossentropy, binary choices에서는 binary_crossentropy
# adam optimizer도 사용가능. RMSprop은 performance 관련해서 learning rate 조정 가능
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# dataset이 아니라 generator 사용하므로 model.fit_generator 사용
history = model.fit_generator(
  # 미리 setup했던. streams the images from the training directory
  train_generator,
  # 1,024 images with 128 batches so 8
  steps_per_epoch=8,
  epochs=15,
  validation_data=validation_generator,
  # 256 images with 32 batches (batch 수는 위 generator 만들 때 사용했떤 값)
  validation_steps=8,
  # how much to display while training is going on
  verbose=2
)
```



 만든 model 사용해서 predict

```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
  
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
```

 



- Walking through developing a ConvNet

[Loss Functions]: https://gombru.github.io/2018/05/23/cross_entropy_loss/	"Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names"
[RMSprop]: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf	"Neural Networks for Machine Learning"
[Binary Classification]: https://www.youtube.com/watch?v=eqEc66RFY0I&amp;t=6s





```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255. normalize
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory (
  '/tmp/horse-or-human/',				# the source directory for training images with 2 sub-dirs
  target_size = (300, 300),			# All images will be resized to 300x300
  batch_size = 128,
  class_mode='binary'						# since we use binary_crossentropy, we need binary labels
)

# train model
history = model.fit_generator(
  train_generator,
  steps_per_epoch=8,
  epochs=15,
  verbose=1
)

# test model
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = file.upload()

for fn in uploaded.keys()
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
    
    
# visualizing Intermidate images
```

[ML Crash Course]: https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture	"loss metrics"



- Adding automatic validation to test accuracy

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
  '/tmp/horse-or-human/',
  target_size=(300, 300),
  batch_size=128,
  class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
  '/tmp/validation-horse-or-human/',
  target_size=(300, 300),
  batch_size=32,
  class_mode='binary'
)

history = model.fit_generator(
  train_generator,
  steps_per_epoch=8,
  epochs=15,
  verbose=1,
  validation_data = validation_generator,
  validation_steps=8
)
```



- Exploring the impact of compressing images

300x300 -> 150x150으로 바꿔서 할 경우: training에 시간 훨씬 적게 걸리고, convolution layer도 더 적게 사용하지만

​                                                                  real image를 잘못 classifying 함. Using this smaller set is much cheaper to train, but then errors like this woman with her back turned and her legs obscured by the dress will happen, because we don't have that data in the training set. That's a nice hint about how to edit your dataset for the best effect in training.



<quiz>

1. using image generator, how do you label images?
    : It's based on the directory the image is contained in
2. what method on the Image Generator is used to normalize the image?
    : rescale
3. how did we specify the training size for the images?
    : the target_size parameter on the training generator
4. when we specify the input_shape to be (300, 300, 3), what does that mean?
    : Every Image will be 300x300 pixels, with 3 bytes to define color
5. If your training data is close to 1.000 accuracy, but your validation isn't, what's the risk here?
    : You're overfitting on your training data
6. Convolutional Neural Networks are better for classifying images like horses and humans because
    : in these images, the features may be in different parts of the frame, there's a wide variety of horses, 
    there's a wide variety of humans
7. After reducing the size of the images, the training results were different. Why?
    : we removed some convolutions to handle the smaller images



<links>

[Horses or Humans Convnet]: https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb
[Horses or Humans with Validation]: https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb
[Horses or Humans with Compacting of Images]: https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb

















