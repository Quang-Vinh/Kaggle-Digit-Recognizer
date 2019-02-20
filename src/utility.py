import numpy as np
import os

def loadParameters() -> dict:
    Theta = []
    B = []
    i = 0

    while(True):
        try:
            Theta.append( np.genfromtxt("../parameters/Theta" + str(i) + ".csv", delimiter = ",") )
            b = np.genfromtxt("../parameters/B" + str(i) + ".csv", delimiter = ",")
            B.append( b.reshape(b.size, 1) )
        except Exception as e:
            break
        i += 1

    return {"Theta": np.array(Theta), "B": np.array(B)}


def saveParameters(parameters: dict):
    deleteParameters()
    
    Theta = parameters["Theta"]
    B = parameters["B"]

    for i in range (Theta.size):
        np.savetxt("../parameters/Theta" + str(i) + ".csv", Theta[i], delimiter = ",")
        np.savetxt("../parameters/B" + str(i) + ".csv", B[i], delimiter = ",")


def deleteParameters():
    folder = '../parameters'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)