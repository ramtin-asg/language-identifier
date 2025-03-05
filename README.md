# language-identifier
The code prepares a dataset, builds a basic neural network, and creates training samples for an NLP classification task based on bigram features. However, some errors and missing implementations make it incomplete for training.
This project implements a feedforward neural network to classify text based on bigrams (pairs of consecutive characters). It uses PyTorch for model training and evaluation.

Features
Text Preprocessing: Converts Unicode text to ASCII and extracts bigrams.
Neural Network: A simple feedforward model with one hidden layer.
Training & Loss Calculation: Uses CrossEntropyLoss and SGD optimizer.
Prediction: Classifies input text based on trained bigram features.
Requirements
Python, PyTorch, NLTK, Google Colab (for file uploads)
Usage
Upload text files for training.
Train the model using the provided script.
Use predict() to classify new input text.
