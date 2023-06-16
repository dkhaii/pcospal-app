# General
This repository contains a machine learning project that focuses on predicting the risk of Polycystic Ovary Syndrome (PCOS) 
using an Artificial Neural Network (ANN) algorithm. 
The ANN model is designed to perform binary classification, categorizing individuals into low or high PCOS risk categories.
The original dataset we used can be accessed [here](https://www.kaggle.com/code/karnikakapoor/pcos-diagnosis)

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Layers](#layers)
- [Callbacks](#callbacks)
- [Model Performance](#model-performance)
- [Inference](#inference)


## Introduction <a name="introduction"></a>
Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder among women of reproductive age. 
Early identification and risk assessment of PCOS can aid in early intervention and treatment. 
This project aims to build a machine learning model that can predict the risk of PCOS based on given input features.

## Model Architecture <a name="model-architecture"></a>
The model utilizes an Artificial Neural Network (ANN) for PCOS risk classification. ANN is a powerful machine learning algorithm inspired by the structure and functioning of the human brain. It consists of interconnected artificial neurons that mimic the behavior of biological neurons.

## Layers <a name="layers"></a>
The ANN model for PCOS risk classification comprises three layers:

1. **Keras Preprocessing Layer**: This layer is responsible for preparing and preprocessing the input data before feeding it into the neural network. It handles tasks such as data normalization, feature scaling, and data augmentation.

2. **Dense Layer**: The dense layer is the main building block of the neural network. It consists of multiple interconnected neurons, also known as nodes or units. Each neuron in the dense layer receives input from all the neurons in the previous layer and performs a weighted sum of the inputs, followed by an activation function to introduce non-linearity. The dense layer helps in learning complex patterns and relationships in the data.

3. **Dropout Layer**: The dropout layer is used to prevent overfitting, a phenomenon where the model performs well on the training data but fails to generalize to unseen data. It randomly drops a certain percentage of neurons during training, forcing the network to learn redundant representations and improving its generalization capabilities.

## Callbacks <a name="callbacks"></a>
Callbacks are used in this project to enhance the training process and prevent the model from training for too long or overfitting. 
The following callbacks are implemented:
- **Early Stopping**: Monitors a specified metric (e.g., validation loss) and stops training if there is no improvement beyond a certain number of epochs.
- **Model Checkpoint**: Saves the best model during training based on a specified metric (e.g., validation accuracy). This allows us to restore the best model weights for later use.

## Model Performance <a name="model-performance"></a>
The developed ANN model achieves an accuracy of approximately 90% in predicting PCOS risk. It should be noted that the performance of the model may vary depending on the quality and quantity of the input data, as well as the hyperparameter settings.

## Running Inference <a name="inference"></a>
To perform inference using the trained model, follow these steps:

1. Clone the repository:
```bash
$ git clone https://github.com/ardhikaptr11/pcos-detection.git
```
2. Extract the contents of `pcos-savedmodel.zip` file. This archive contains the saved model and necessary files for inference.
3. Open a terminal and navigate to the project directory.
4. Run the `inference.py` script:
```
python inference.py
```

The `inference.py` script serves as a simulation to obtain input data that will be fed into the model for prediction. You can modify the script to provide the required input data and observe the model's predictions.


