CIFAR-10 CNN Image Classifier

This project is a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model achieves high accuracy by incorporating data augmentation, batch normalization, and dropout layers to enhance performance.

Requirements

Python 3.7+

TensorFlow

NumPy

JAX

Matplotlib (optional for visualization)

You can install the required libraries using:

pip install tensorflow numpy jax matplotlib

Project Structure

CIFAR10-CNN-Image-Classifier/
│── model.py            # CNN Model Training Script
│── dataset.py          # CIFAR-10 Data Loading
│── README.md           # Project Documentation
│── requirements.txt    # Required Libraries
└── results/            # Stores Training Results & Logs

How to Run

Clone this repository:

git clone https://github.com/yourusername/CIFAR10-CNN-Image-Classifier.git

Navigate to the project folder:

cd CIFAR10-CNN-Image-Classifier

Install dependencies:

pip install -r requirements.txt

Run the model training script:

python model.py

Uploading the Project to GitHub

Follow these steps to upload your project to GitHub:

Initialize Git:

git init

Add files:

git add .

Commit changes:

git commit -m "Initial commit - CIFAR-10 CNN Image Classifier"

Create a new repository on GitHub and copy the repo URL.

Add remote repository:

git remote add origin https://github.com/yourusername/CIFAR10-CNN-Image-Classifier.git

Push code to GitHub:

git branch -M main
git push -u origin main

Results

After training, the model achieves a test accuracy of ~84%. The accuracy may improve further with hyperparameter tuning.
