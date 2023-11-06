import tensorflow as tf
from keras.layers import RandomFlip, RandomZoom, RandomRotation
from keras.optimizers import Lion

from model import CNeXt

image_size = (256, 256)
batch_size = 256

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    image_size=image_size,
    batch_size=batch_size,
)
class_names = dataset.class_names
num_classes = len(class_names)

data_augmentation = tf.keras.Sequential([
        RandomZoom(0.2),
        RandomRotation(0.2),
    ], name='data_augmentation')

model = tf.keras.Sequential([
    data_augmentation,
    CNeXt(num_classes=num_classes),
])
model.build(input_shape=(batch_size, *image_size, 3))

model.compile(optimizer=Lion(learning_rate=0.001), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(dataset, epochs=50)
print('Saving model...')
model.layers[-1].save_weights('image-model-weights.h5')
print('Done.')
