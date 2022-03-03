import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt
from prepare_dataset import prepare_data
from model import train_model_1, train_model_2, train_model_3, save_model_as_json, load_model_as_json
from preproces_image import prepare_image_for_network
from tools import calculate_confusion_matrix

symbols_list = ['amazon', 'bbc', 'cnn', 'ebay', 'github', 'google', 'njuskalo', 'spiegel', 'theGuardian', 'youtube']
list_of_targets = ["https://www.amazon.com/",
                   "https://www.bbc.com/",
                   "https://cnn.com",
                   "https://www.ebay.com/",
                   "https://github.com/",
                   "https://www.google.com/",
                   "https://www.njuskalo.hr/",
                   "https://www.spiegel.de/",
                   "https://www.theguardian.com/",
                   "https://www.youtube.com/"]

training_path = "data/training"
testing_path = "data/test"
image_path = "test_images/amazon1.jpg"
model_path = "models_json/model.json"


def _parse_args():
    parser = argparse.ArgumentParser(description="script to predict the website from the image")
    parser.add_argument(
        "--input-path",
        help="path to image that wants to be tested",
        default=image_path,
        type=str
    )
    return parser.parse_args()


def metrics(testing_path, model):
    calculate_confusion_matrix(testing_path, model)


def test_model(image_path, model):
    result = predict_image(image_path, model)
    img = cv2.imread(image_path)
    plt.imshow(img)
    plt.title(result)
    plt.show()


def predict_image(image_path, model):
    image = prepare_image_for_network(image_path)
    prediction = model.predict(image)
    result = np.argmax(prediction)
    final_label = list_of_targets[result]
    return final_label


def train_model():
    X_train, y_train, X_test, y_test = prepare_data(training_path, testing_path)
    model = train_model_2(X_train, y_train, X_test, y_test)
    save_model_as_json(model)
    if model:
        print('Model successfully trained')
    else:
        print("Error creating model")
    return model


def load_model():
    model = load_model_as_json(model_path)
    return model


def run_model():
    model = load_model()
    if model:
        print("model successfully loaded")
    else:
        print("error loading the model")
    img_path = args.input_path
    result = predict_image(img_path, model)
    return result


if __name__ == '__main__':
    args = _parse_args()
    result = run_model()
    print(f"Click on the link to get to your site {result} ,if this is not the site you were looking for we apologize,"
          f"try to take another screenshot, the model is not perfect :)")
    # model = train_model()
    # test_model(image_path, model)
