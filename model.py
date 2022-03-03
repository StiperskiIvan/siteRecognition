import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
from keras.models import model_from_json


def load_model_as_json(model_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models_json/modelw.h5")
    return model


def save_model_as_json(model):
    model_json = model.to_json()
    with open("models_json/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models_json/modelw.h5")
    print("Model weights saved to disk")


def define_model_1(x_train):
    # Creating Sequential model
    # using sequential model for training
    model = Sequential()
    # 1st layer and taking input in this of shape 250x250x3 ->  250 x 250 pixles and 3 channels
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    # maxpooling will take highest value from a filter of 2*2 shape
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # it will prevent overfitting by making it hard for the model to idenify the images
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    # last layer predicts 10 labels
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
        # metrics=['top_k_categorical_accuracy'],
    )

    model.summary()
    # Visualising the model
    # displaying the model
    keras.utils.vis_utils.plot_model(model, "model.png", show_shapes=True)
    return model


def define_model_2(x_train):
    # Define model 2.
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    model.summary()
    # Visualising the model
    # displaying the model
    keras.utils.vis_utils.plot_model(model, "model2.png", show_shapes=True)
    return model


def define_model_3(x_train):
    # Define model 3.
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    model.summary()
    # Visualising the model
    # displaying the model
    keras.utils.vis_utils.plot_model(model, "model3.png", show_shapes=True)
    return model


def train_model_1(X_train, y_train, X_test, y_test,  disp_mod_acc=False, disp_mod_loss=False, predict=False):
    # training the model
    model = define_model_1(X_train)
    history = model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=5,
        validation_split=0.2,
        shuffle=True
    )
    model.save('models/model.h5')
    # Visualising the outcome
    # displaying the model accuracy
    if disp_mod_acc:
        plt.plot(history.history['accuracy'], label='train', color="red")
        plt.plot(history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    # displaying the model loss
    if disp_mod_loss:
        plt.plot(history.history['loss'], label='train', color="red")
        plt.plot(history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    score = model.evaluate(X_train, y_train, verbose=0)
    print('Training accuarcy: {:0.2f}%'.format(score[1] * 100))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

    if predict:
        pred = model.predict(X_test)
        print(pred)
    return model


def train_model_2(X_train, y_train, X_test, y_test, disp_mod_acc=False, disp_mod_loss=False, predict=False):
    # training the model
    model = define_model_2(X_train)
    history = model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=25,
        validation_split=0.2,
        shuffle=True
    )
    model.save('models/model.h5')
    # Visualising the outcome
    # displaying the model accuracy
    if disp_mod_acc:
        plt.plot(history.history['accuracy'], label='train', color="red")
        plt.plot(history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    # displaying the model loss
    if disp_mod_loss:
        plt.plot(history.history['loss'], label='train', color="red")
        plt.plot(history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    score = model.evaluate(X_train, y_train, verbose=0)
    print('Training accuarcy: {:0.2f}%'.format(score[1] * 100))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

    if predict:
        pred = model.predict(X_test)
        print(pred)
    return model


def train_model_3(X_train, y_train, X_test, y_test, disp_mod_acc=False, disp_mod_loss=False, predict=False):
    # training the model
    model = define_model_3(X_train)
    history = model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=100,
        validation_split=0.2,
        shuffle=True
    )
    model.save('models/model3.h5')
    # Visualising the outcome
    # displaying the model accuracy
    if disp_mod_acc:
        plt.plot(history.history['accuracy'], label='train', color="red")
        plt.plot(history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    # displaying the model loss
    if disp_mod_loss:
        plt.plot(history.history['loss'], label='train', color="red")
        plt.plot(history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    score = model.evaluate(X_train, y_train, verbose=0)
    print('Training accuarcy: {:0.2f}%'.format(score[1] * 100))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

    if predict:
        pred = model.predict(X_test)
        print(pred)

    return model
