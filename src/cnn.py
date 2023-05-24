##### LOAD PACKAGES
# basic tools
import os
import pandas as pd
import numpy as np

# image preprocessing
import cv2

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import matplotlib.pyplot as plt

##### FUNCTIONS
# PLOTTING FUNCTION
# This function was provided as part of the Visual Analytics course
# in the Cultural Data Science elective at Aarhus University.
def plot_history(H, epochs):
    
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.plot(
        np.arange(0, epochs), 
        H.history["loss"], 
        label = "train_loss")
    plt.plot(
        np.arange(0, epochs), 
        H.history["val_loss"], 
        label = "val_loss", 
        linestyle = ":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(
        np.arange(0, epochs), 
        H.history["accuracy"], 
        label = "train_acc")
    plt.plot(
        np.arange(0, epochs), 
        H.history["val_accuracy"], 
        label = "val_acc", 
        linestyle = ":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plt.savefig(os.path.join(
        "out", 
        "plot.jpg"))

##### MAIN
def main():

    # LOAD DATA

    # define directories for images
    test_dir = os.path.join(
        "..",
        "..",
        "431824",
        "images",
        "test")
    train_dir = os.path.join(
        "..",
        "..",
        "431824",
        "images",
        "train")
    val_dir = os.path.join(
        "..",
        "..",
        "431824",
        "images",
        "val")

    # define overall path to metadata files
    path = os.path.join(
        "..",
        "..",
        "431824",
        "images",
        "metadata")

    # load metadata files
    test_data = pd.read_json(
        os.path.join(
            path,
            "test_data.json"), 
        lines = True)
    train_data = pd.read_json(
        os.path.join(
            path,
            "train_data.json"), 
        lines = True)
    val_data = pd.read_json(
        os.path.join(
            path,
            "val_data.json"), 
        lines = True)

    # CREATE FLOW DATAFRAME FOR MODEL TRAINING
    # Create supplementary path for making a full path out of the paths
    # contained in the metadata files.
    sup_path = os.path.join( 
        "..",
        "..", 
        "..", 
        "431824")

    # Pull 'path' column into variable.
    test_imgs = test_data["image_path"]
    train_imgs = train_data["image_path"]
    val_imgs = val_data["image_path"]

    # Pull 'label' column into variable.
    y_test = test_data["class_label"]
    y_train = train_data["class_label"]
    y_val = val_data["class_label"]

    # Create dataframes for flow.
    test = {
        'image_path': test_imgs,
        'label': y_test
    }
    train = {
        'image_path': train_imgs,
        'label': y_train
    }
    val = {
        'image_path': val_imgs,
        'label': y_val
    }
    test_df = pd.DataFrame(test)
    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)

    # Create full paths.
    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(
        sup_path, 
        x))
    train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(
        sup_path, 
        x))
    val_df['image_path'] = val_df['image_path'].apply(lambda x: os.path.join(
        sup_path, 
        x))

    # BINARIZE LABELS
    # Binarize labels (one-hot vectors).
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)

    labelNames = [
        'blouse', 
        'dhoti_pants', 
        'dupattas', 
        'gowns', 
        'kurta_men',
        'leggings_and_salwars', 
        'lehenga', 
        'mojaris_men', 
        'mojaris_women',
        'nehru_jackets', 
        'palazzos', 
        'petticoats', 
        'saree', 
        'sherwanis',
        'women_kurta'] # alphabetical order

    # CREATE MODEL
    # Load model without classifier layers.
    model = VGG16(
        include_top = False, 
        pooling = 'avg',
        input_shape = (224, 224, 3))

    # Mark loaded layers as not trainable.
    for layer in model.layers:
        layer.trainable = False

    # Add new classifier layers.
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(
        256, 
        activation = 'relu')(bn)
    class2 = Dense(
        128, 
        activation = 'relu')(class1)
    output = Dense(
        15, # there are 15 Indian fashion categories
        activation = 'softmax')(class2)

    # Define new model.
    model = Model(
        inputs = model.inputs, 
        outputs = output)

    # Configure learning rate.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.01,
        decay_steps = 10000,
        decay_rate = 0.9)
    sgd = SGD(learning_rate = lr_schedule)

    # Compile model.
    model.compile(
        optimizer = sgd,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

    # DEFINE FLOW
    # Define important variables.
    IMG_SHAPE = 224
    batch_size = 128

    # Define data generator. Images are normalized, and horizontal
    # flipping and rotation are added. They are also resized in order
    # to fit the model's size requirements.
    datagen = ImageDataGenerator(
        rescale = 1./255, 
        horizontal_flip = True, 
        rotation_range = 20,
        preprocessing_function = lambda x: tf.image.resize(x, (IMG_SHAPE, IMG_SHAPE)))

    # Image iterator for test split. 
    img_iter_test = datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = test_dir,
        x_col = "image_path",
        y_col = "label",
        target_size = (IMG_SHAPE, IMG_SHAPE),
        batch_size = batch_size,
        class_mode = "categorical")

    # Image iterator for train split. 
    # define image iterator to flow images from folder
    img_iter_train = datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = train_dir,
        x_col = "image_path",
        y_col = "label",
        target_size = (IMG_SHAPE, IMG_SHAPE),
        batch_size = batch_size,
        class_mode = "categorical")

    # Image iterator for validation split. 
    # define image iterator to flow images from folder
    img_iter_val = datagen.flow_from_dataframe(
        dataframe = val_df,
        directory = val_dir,
        x_col = "image_path",
        y_col = "label",
        target_size = (IMG_SHAPE, IMG_SHAPE),
        batch_size = batch_size,
        class_mode = "categorical")

    # RUN MODEL
    # Train model
    H = model.fit(
        img_iter_train,
        validation_data = img_iter_val,
        epochs = 10)

    # Plot.
    plot = plot_history(H, 10)

    # Make predictions on test dataset.
    predictions = model.predict(
        img_iter_test, 
        batch_size = batch_size)

    # Prepare classification report. 
    report = classification_report(
        y_test.argmax(axis = 1),
        predictions.argmax(axis = 1),
        target_names = labelNames)

    outpath = os.path.join(
            "out",
            "report.txt")

    with open(outpath, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()