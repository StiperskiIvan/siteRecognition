import os

import numpy as np
import keras
import tensorflow as tf
from keras.utils import np_utils
from sklearn import preprocessing

from prepare_dataset import preprocess_image


symbols_list = ['amazon', 'bbc', 'cnn', 'ebay', 'github', 'google', 'njuskalo', 'spiegel', 'theGuardian', 'youtube']


def calculate_confusion_matrix(eval_path, model):
    test_image = []
    test_label = []
    X_test = []

    for symbols_dir in os.listdir(eval_path):
        if symbols_dir.split()[0] in symbols_list:
            for image in os.listdir(str(eval_path) + '/' + symbols_dir):
                test_label.append(symbols_dir.split()[0])
                test_image.append(str(eval_path) + '/' + symbols_dir + '/' + image)

    for path in test_image:
        img = preprocess_image(path)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img /= 255.
        X_test.append(img)

    # X_test = np.array(X_test)
    # X_test = X_test.astype('float32')
    # X_test /= 255
    label_encoder = preprocessing.LabelEncoder()
    y_test_temp = label_encoder.fit_transform(test_label)
    y_test = keras.utils.np_utils.to_categorical(y_test_temp, 10)
    predictions = []
    labels = y_test_temp
    for item in X_test:
        prediction = model.predict(item)
        predictions.append(np.argmax(prediction))

    matrix = tf.math.confusion_matrix(labels, predictions)
    matrix = matrix.numpy()
    normalised_matrix = []

    for label in matrix:
        max_items = sum(label)
        label = (label / max_items)
        normalised_matrix.append(label)
    print(f"confusion matrix: \n {matrix} \n")
    np.set_printoptions(formatter={'all': lambda x: " {:.2f} ".format(x)})
    print(f"Accuracy per class is: \n {symbols_list} \n {np.diagonal(normalised_matrix)}")

