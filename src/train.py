import numpy as np
import pandas as pd
import time
from nnModel import *
from utility import *
from optimizers import *


if __name__ == "__main__":
    # Set seed
    np.random.seed(2)

    # Read data from csv file
    data = pd.read_csv('../data/train.csv')

    # Split into train and cross validation sets
    Y = data['label']
    X = data.drop(labels = ['label'], axis = 1)
    data_m = data.shape[0]
    n_x = X.shape[1]
    n_y = 10

    train_m = int(data_m * 0.9)
    crossVal_m = data_m - train_m
    trainY = Y[:train_m]
    trainX = X[:train_m]
    crossValY = Y[train_m:]
    crossValX = X[train_m:]

    # Convert into numpy ndarrays
    trainY = np.array(trainY).transpose().reshape(1, train_m)
    trainX = np.array(trainX).transpose().reshape(n_x, train_m)

    crossValY = np.array(crossValY).transpose().reshape(1, crossVal_m)
    crossValX = np.array(crossValX).transpose().reshape(n_x, crossVal_m)

    # Clear data
    del data

    # Hyper parameters
    n_h = [400, 400, 400]
    lambdaVal = 0.5
    dropout = 0
    alpha = 1E-4
    beta1 = 0.9
    beta2 = 0.999
    numEpochs = 10 
    batchSize = 84

    # Normalize data
    trainX = trainX / 255
    crossValX = crossValX / 255

    # Initialize Weights theta and b
    parameters = initializeParams([n_x] + n_h + [n_y])
    # parameters = loadParameters()

    # Train
    t0 = time.time()
    parameters = adam(trainX, trainY, parameters, lambda X, Y, parameters: nnCost(X, Y, parameters, lambdaVal, dropout), alpha = alpha, beta1 = beta1, beta2 = beta2, numEpochs = numEpochs, batchSize = batchSize)
    print("\nTime: " + str(time.time() - t0))

    # Accuracy on training
    predictions = predict(trainX, parameters)
    print("\nTraining accuracy: " + str(np.sum(trainY == predictions) / train_m) )

    # Save parameters    
    saveParameters(parameters)

    # Clear memory
    del trainX, trainY, predictions

    # Check accuracy on cross validation
    predictions = predict(crossValX, parameters)
    print("\nCross validation accuracy: " + str(np.sum(crossValY == predictions) / crossVal_m))
