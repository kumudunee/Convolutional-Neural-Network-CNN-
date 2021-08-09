import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()
print(len(x_train))
print(len(x_test))

print(x_train[0])

print(x_train[0].shape)
plt.matshow(x_train[0])
plt.show()

plt.matshow(x_train[2])
plt.show()

print(y_train[2])

print(y_train[:5])

print(x_train.shape)

x_train = x_train / 255
x_test = x_test / 255

print(x_train[0])

print(x_train.reshape(len(x_train), 28*28))

x_train_flattened = x_train.reshape(len(x_train),28*28)
print(x_train_flattened.shape)

x_test_flattened = x_test.reshape(len(x_test),28*28)
print(x_test_flattened.shape)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
])

model.complile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_flattened, y_train, epochs=5)

model.fit(x_test, x_test_flattened, y_test)

y_predicted = model.predict(x_test_flattened)
print(y_predicted[1])

np.argmax(y_predicted[1])

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

cm = tf.math_confusion_matrix(lbels=y_test,predictions=y_predicted_labels)
print(cm)
