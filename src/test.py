import numpy as np 
import pandas as pd
from nnModel import *
from utility import *


if __name__ == "__main__":
    # Read test data from csv
    testData = pd.read_csv('../data/test.csv')
    
    # Normalize data
    testData = testData / 255

    # Convert to np ndarray
    testData = np.array(testData).transpose()

    # Load model weights
    parameters = loadParameters()

    # Calculate predictions and set imageId
    predictions = np.transpose( predict(testData, parameters) )
    imageId = np.arange(predictions.size).reshape(predictions.size, 1) + 1

    # Create test submission csv and save
    testSubmission = np.hstack((imageId, predictions))
    testSubmission = pd.DataFrame(testSubmission, columns = ['ImageId', 'Label'])
    testSubmission.to_csv('../data/submission.csv', index = False)

    



