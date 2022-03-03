import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import os
import cv2
from sklearn import preprocessing


symbols_list = ['amazon', 'bbc', 'cnn', 'ebay', 'github', 'google', 'njuskalo', 'spiegel', 'theGuardian', 'youtube']

resize_size = (250, 250)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)
    img = np.array(img)
    return img


def check_random_image(train_label, train_image):
    random_number = random.randint(0, len(train_label))
    image = cv2.imread(train_image[random_number])
    plt.imshow(image)
    plt.title("Label: " + train_label[random_number])
    plt.show()


def prepare_data(training_path, eval_path):
    # Loading the folder paths of all testing and training images
    # separate into training and testing

    train_image = []
    train_label = []

    for symbols_dir in os.listdir(training_path):
        if symbols_dir.split()[0] in symbols_list:
            for image in os.listdir(str(training_path) + '/' + symbols_dir):
                train_label.append(symbols_dir.split()[0])
                train_image.append(str(training_path) + '/' + symbols_dir + '/' + image)

    test_image = []
    test_label = []

    for symbols_dir in os.listdir(eval_path):
        if symbols_dir.split()[0] in symbols_list:
            for image in os.listdir(str(eval_path) + '/' + symbols_dir):
                test_label.append(symbols_dir.split()[0])
                test_image.append(str(eval_path) + '/' + symbols_dir + '/' + image)

    print("Length of train_image : ", len(train_image), " , length of labels list : ", len(train_label))
    print("Length of test_image : ", len(test_image), " , length of labels list : ", len(test_label))

    # Verifying the data
    # Let's see that we have 10 unique labels for both test and train

    unique_test = list(set(test_label))
    unique_train = list(set(train_label))
    print("Length of test unique labels: ", len(unique_test), " : ", unique_test)
    print("Length of train unique labels: ", len(unique_train), " : ", unique_train)

    # Loading the images and label and checking correctness
    # check_random_image(train_label, train_image)
    # Creating train test and validation set
    test = np.array(cv2.imread(train_image[20]))
    print(test.shape)

    # Creating the X_train and X_test

    X_train = []
    X_test = []

    # loading the images from the path
    for path in train_image:
        img = preprocess_image(path)
        X_train.append(img)

    for path in test_image:
        img = preprocess_image(path)
        X_test.append(img)

    # creating numpy array from the images
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # normalizing the data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    # Creating the y_train and y_test

    # label encoding the 10 symbols
    label_encoder = preprocessing.LabelEncoder()
    y_train_temp = label_encoder.fit_transform(train_label)
    y_test_temp = label_encoder.fit_transform(test_label)

    print("y_train_temp shape: ", y_train_temp.shape)
    print("y_test_temp shape: ", y_test_temp.shape)

    # creating matrix labels list
    y_train = keras.utils.np_utils.to_categorical(y_train_temp, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test_temp, 10)

    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)
    return X_train, y_train, X_test, y_test




