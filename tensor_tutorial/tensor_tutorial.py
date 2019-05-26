from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Getting training and test data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Class names of the labels
    # labels are 0-9
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # train_images.shape # (num of images, img_dim.x, img_dim.y)
    # test_images.shape # (num of images, img_dim.x, img_dim.y)

    # making pixel values fall between 0.0 and 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # verifying the data is in the correct format
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    model = keras.Sequential([
        # transforms 2D array to 1D
        keras.layers.Flatten(input_shape=(28, 28)), 
        # first neural layer of 128 nodes
        keras.layers.Dense(128, activation=tf.nn.relu),
        # 10-node softmax layer that returns an array of 10 probability scores that sum to 1
        # i.e the probability that an images belongs to 1-10 labels
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Settings added during compile
    # Loss function: Measures how accurate the model is during training
    # Optimizer: How the model updated based on the data it sees and loss function
    # Metrics: Used to monitor the training and testing steps

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model and 'fit' it to the training data
    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy: {}'.format(test_acc))

if __name__ == "__main__":
    main()
