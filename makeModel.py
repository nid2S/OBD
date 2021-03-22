import numpy as np
import tensorflow as tf
import Camera
from glob import glob
from sklearn.model_selection import train_test_split

root = './dataset/images/*.jpg'
classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# get images list
imageList = glob(root)
if len(imageList) == 0:
    raise FileNotFoundError('image not exist.')

# get images max size
imageSize = Camera.getMaxSize(imageList)

# make image set
all_image = np.ndarray(shape=(len(imageList), imageSize[0], imageSize[1]), dtype=np.float32)
# all_image = np.ndarray(shape=(len(imageList), imageSize[0], imageSize[1], 3), dtype=np.float32)

all_label = np.ndarray(shape=(len(imageList),), dtype=np.int32)

for i in range(len(imageList)):
    all_image[i] = Camera.pilImread(imageList[i])
    # all_image[i] = Camera.cv2Imread(imageList[i])

    for j in classes:
        if j in imageList[i][len(root)-4:-4]:
            all_label[i] = j
            break

        if j == classes[-1]:
            raise ValueError("image don't match with classes")

# split to train and test set
train_img, test_img, train_label, test_label = train_test_split(all_image, all_label, train_size=0.9)

# make model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(len(imageList), imageSize[0], imageSize[1])),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    matrix='accuracy'
)

# train model
model.fit(train_img, train_label, epochs=300)

# test model
loss, accuracy = model.evaluate(test_img, test_label)
print(accuracy)