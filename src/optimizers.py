import numpy as np

def gradientDescentStep(parameters: dict, costFunc, alpha: int = 0):
    cost, dParameters = costFunc(parameters)
    parameters["Theta"] -= alpha * dParameters["dTheta"]
    parameters["B"] -= alpha * dParameters["dB"]
    return cost, parameters


def gradientDescent(parameters: dict, costFunc, alpha: int = 0, numIterations: int = 100):
    for i in range(numIterations):
        cost, parameters = gradientDescentStep(parameters, costFunc, alpha)
        print("Epoch " + str(i) + "   |    Cost: " + str(cost)) 
    return parameters
    

def batchGradientDescent(X: np.ndarray, Y: np.ndarray, parameters: dict, costFunc, alpha: int = 0, numEpochs: int = 100, batchSize: int = 1):
    X_batches = np.array_split(X, X.shape[1] // batchSize, axis = 1)
    Y_batches = np.array_split(Y, Y.shape[1] // batchSize, axis = 1)
    for epoch in range (numEpochs):
        for i in range (len(X_batches)):
            cost, parameters = gradientDescentStep(parameters, lambda parameters: costFunc(X_batches[i], Y_batches[i], parameters), alpha)
            print("Epoch " + str(epoch) + "  Batch " + str(i) + "     |    Cost: " + str(cost))
    return parameters


def momentum(X: np.ndarray, Y: np.ndarray, parameters: dict, costFunc, alpha: int = 0, beta: int = 0.9, numEpochs: int = 100, batchSize: int = 1):
    VdTheta, VdB = 0, 0
    X_batches = np.array_split(X, X.shape[1] // batchSize, axis = 1)
    Y_batches = np.array_split(Y, Y.shape[1] // batchSize, axis = 1)
    for epoch in range (numEpochs):
        for i in range (len(X_batches)):
            cost, dParameters = costFunc(X_batches[i], Y_batches[i], parameters)
            VdTheta = beta * VdTheta + (1 - beta) * dParameters["dTheta"]
            VdB = beta * VdB + (1 - beta) * dParameters["dB"]
            parameters["Theta"] -= alpha * VdTheta
            parameters["B"] -= alpha * VdB
            print("Epoch " + str(epoch) + "  Batch " + str(i) + "     |    Cost: " + str(cost))
    return parameters


def RMSprop(X: np.ndarray, Y: np.ndarray, parameters: dict, costFunc, alpha: int = 0, beta: int = 0.999, numEpochs: int = 100, batchSize: int = 1):
    SdTheta, SdB = 0, 0
    Epsilon = 10 ** -8
    X_batches = np.array_split(X, X.shape[1] // batchSize, axis = 1)
    Y_batches = np.array_split(Y, Y.shape[1] // batchSize, axis = 1)
    for epoch in range (numEpochs):
        for i in range (len(X_batches)):
            cost, dParameters = costFunc(X_batches[i], Y_batches[i], parameters)
            SdTheta = beta * SdTheta + (1 - beta) * np.square(dParameters["dTheta"])
            SdB = beta * SdB + (1 - beta) * np.square(dParameters["dB"])
            parameters["Theta"] -= alpha * dParameters["dTheta"] / ( SdTheta ** 0.5 + Epsilon )
            parameters["B"] -= alpha * dParameters["dB"] / ( SdB ** 0.5 + Epsilon )
            print("Epoch " + str(epoch) + "  Batch " + str(i) + "     |    Cost: " + str(cost))
    return parameters


def adam(X: np.ndarray, Y: np.ndarray, parameters: dict, costFunc, alpha: int = 0, beta1: int = 0.9, beta2: int = 0.999, numEpochs: int = 100, batchSize: int = 1):
    VdTheta, VdB, SdTheta, SdB = 0, 0, 0, 0
    Epsilon = 10 ** -8
    t = 0
    X_batches = np.array_split(X, X.shape[1] // batchSize, axis = 1)
    Y_batches = np.array_split(Y, Y.shape[1] // batchSize, axis = 1)
    for epoch in range (1, numEpochs + 1):
        for i in range (len(X_batches)):
            t += 1
            cost, dParameters = costFunc(X_batches[i], Y_batches[i], parameters)

            # Calculate V and S
            VdTheta = beta1 * VdTheta + (1 - beta1) * dParameters["dTheta"]
            VdB = beta1 * VdB + (1 - beta1) * dParameters["dB"]
            SdTheta = beta2 * SdTheta + (1 - beta2) * np.square(dParameters["dTheta"])
            SdB = beta2 * SdB + (1 - beta2) * np.square(dParameters["dB"])

            # Corrected values
            VdThetaCorrected = VdTheta / (1 - beta1 ** t)
            VdBCorrected = VdB / (1 - beta1 ** t)
            SdThetaCorrected = SdTheta / (1 - beta2 ** t)
            SdBCorrected = SdB / (1 - beta2 ** t)

            parameters["Theta"] -= alpha * VdThetaCorrected / (SdThetaCorrected ** 0.5 + Epsilon)
            parameters["B"] -= alpha * VdBCorrected / (SdBCorrected ** 0.5 + Epsilon)
            print("Epoch " + str(epoch) + "  Batch " + str(i) + "     |    Cost: " + str(cost))
    return parameters