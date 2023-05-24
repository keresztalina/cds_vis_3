# Assignment 3 - Using pretrained CNNs for image classification
This assignment is ***Part 3*** of the portfolio exam for ***Visual Analytics S23***. The exam consists of 4 assignments in total (3 class assignments and 1 self-assigned project).

## 3.1. Contribution
The initial assignment was created partially in collaboration with other students in the course, also making use of code provided as part of the course (for example the plotting function, which has only been edited to save the produced plot). The final code is my own. Several adjustments have been made since the initial hand-in. 

Here is the link to the GitHub repository containing the code for this assignment: ADD

## 3.2. Assignment description by Ross
*(**NB!** This description has been edited for brevity. Find the full instructions in ```README_rdkm.md```.)*

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report

## 3.3. Methods
The purpose of this script is to use a pretrained convolutional neural network (CNN) with additional classifier layers to classify previously unseen images of Indian fashion into 15 possible categories with the highest possible accuracy. 

First, the structure of the dataset is established within the script. The ```.json``` files containing metadata are loaded and preprocessed in order to allow images to be flowed from the directory into the model (the dataset is large, over 3GB - loading all images at once would at best be computationally inefficient, or at worst would crash the computer). In the resulting dataframes, one column contains the relative path to each image, and another column contains the associated label. The labels are also binarized.

Second, the model is loaded. The pretrained model VGG16 is loaded without its top layers (i.e. the original classification layers), and the existing layers are set to non-trainable, as we would like to be compuationally efficient and use the existing weights. Three classifier layers are added: two hidden layers (256 and 128 neurons) with ```relu``` activation functions (in order to avoid vanishing gradients), and an output layer (15 neurons to correspond to the 15 possible labels). The learning rate of the model is also configured to an expontially decaying learning rate, where the model initially moves fast down the loss curve, then slows down in order to avoid missing the minimum.

Third, the images are preprocessed and the data is augmented. The images are rescaled and resized to 224 * 224 pixels. The data is augmented through horizontal flipping and rotation in a range of 20 degrees. The images are then prepared to be flowed from the directory in batches based on the metadata dataframes.

Then, the model is fit to the data and validated on the validation split through 10 epochs. The model's performance is tested by making it predict the labels in the test split. Finally, a classification report is made.

## 3.4 Usage
### 3.4.1. Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.







