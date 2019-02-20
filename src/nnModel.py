import numpy as np
from activationFunctions import *


def initializeParams(layer_sizes: list) -> dict:
    Theta = []
    B = []

    for i in range (len(layer_sizes) - 1):
        Theta.append( np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1])) )
        B.append( np.zeros((layer_sizes[i + 1], 1)) )

    return {"Theta": np.array(Theta), "B": np.array(B)}

'''
    X is n_x x m matrix with samples along column
    Y is 1 x m matrix 
    Theta is numHiddenLayers x hiddenLayerSize x previousHiddenLayerSize
    b is numHiddenLayers x hiddenLayerSize x 1    
'''
def nnCost(X: np.ndarray, Y: np.ndarray, parameters: dict, lambdaVal: int = 0, dropout = 1):
    # Constants
    Theta = parameters["Theta"]
    B = parameters["B"]
    m = np.size(X, 1)
    n_h_size = Theta.shape[0] - 1
    n_y = Theta[-1].shape[0]

    # === Forward propagation ===========================================
    Z = []
    A = [X]
    D = []

    # Hidden layers
    for i in range (n_h_size):
        Di = np.random.rand(A[i].shape[0], A[i].shape[1]) > dropout
        A[i] = (A[i] * Di) / (1 - dropout)
        if (i > 0):
            D.append(Di)
        Z.append( Theta[i].dot(A[i]) + B[i] )
        A.append( activation(Z[i] ))

    # Output layer
    Dy = np.random.rand(A[-1].shape[0], A[-1].shape[1]) > dropout
    D.append(Dy)
    A[-1] = (A[-1] * Dy) / (1 - dropout)
    Zy = Theta[-1].dot(A[-1]) + B[-1]
    Ay = outputActivation(Zy)

    outputs =  np.arange(n_y).reshape([n_y,1])
    yMat = (np.repeat(outputs, m, 1) == Y) * 1
    loss = np.sum( -yMat * np.log(Ay) + (yMat - 1) * np.log(1 - Ay) )

    #L2
    regularization = 0
    for i in range (Theta.size):
        regularization += np.sum(Theta[i] ** 2)

    cost = (1 / m) * loss + (lambdaVal / (2 * m)) * regularization

    # === Back propagation====================================================
    dZ = []
    dTheta = []
    dB = []

    #Output layer
    dZTemp = Ay - yMat
    dZ.insert(0, dZTemp)
    dThetaTemp = (1 / m) * dZTemp.dot( np.transpose(A[-1] ) ) + (lambdaVal / m) * Theta[-1]
    dTheta.insert(0, dThetaTemp)
    dBTemp = (1 / m) * np.sum(dZTemp, axis = 1, keepdims = True)
    dB.insert(0, dBTemp)

    for i in range (n_h_size, 0, -1):
        dA = (np.transpose(Theta[i]).dot(dZ[0]) * D[i-1]) / (1 - dropout)
        dZTemp = dA * activationGradient( Z[i - 1] )
        dZ.insert(0, dZTemp)
        dThetaTemp = (1 / m) * dZTemp.dot( np.transpose(A[i - 1]) ) + (lambdaVal / m) * Theta[i -1]
        dTheta.insert(0, dThetaTemp)
        dBTemp = (1 / m) * np.sum(dZTemp, axis = 1, keepdims = True)
        dB.insert(0, dBTemp)

    return cost, {"dTheta": np.array(dTheta), "dB": np.array(dB)}



def predict(X: np.ndarray, parameters: dict) -> np.ndarray:
    Theta = parameters["Theta"]
    B = parameters["B"]
    n_h_size = Theta.shape[0] - 1
    m  = X.shape[1]

    Z = []
    A = [X]

    # Hidden layers
    for i in range (n_h_size):
        Z.append( Theta[i].dot(A[i]) + B[i] )
        A.append( activation(Z[i] ))

    # Output layer
    Zy = Theta[-1].dot(A[-1]) + B[-1]
    Ay = outputActivation(Zy)

    return Ay.argmax(axis = 0).reshape(1, m)

