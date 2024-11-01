Fashion MNIST Image Classification

This project implements a Convolutional Neural Network (CNN) for classifying fashion items from the Fashion MNIST dataset using TensorFlow and Keras.

Functionality:

Loads and preprocesses the Fashion MNIST dataset.
Defines a CNN architecture with convolutional and pooling layers, followed by dense layers for classification.
Optionally implements data augmentation to artificially increase the training data and improve model robustness.
Trains the model on the training data and evaluates its performance on the test set.
Makes predictions on new, unseen images and compares them to the true labels.

Instructions:

Install Required Packages (Python):

Ensure you have TensorFlow and Keras installed using pip install tensorflow keras.
Run the Python Script (fashion_mnist.py):

Save the provided code as fashion_mnist.py.
Execute the script from your terminal using python fashion_mnist.py.

Run the R Script (fashion_mnist.R):

Install TensorFlow and Keras packages in R if not already installed:
Code snippet
if (!require(tensorflow)) install.packages("tensorflow")
if (!require(keras)) install.packages("keras")


Save the provided code as fashion_mnist.R.
Open the script in RStudio or your preferred R environment and run it line by line or as a whole script.

Output:

The script will print the accuracy of the model on the test set.
For the R version, it will also print the true label and predicted label for the first two test images.
