import numpy as np

def regressionMatrix(inputSignal: np.ndarray, outputSignal: np.ndarray, lenYparams: int, lenUparams: int) -> np.ndarray:

    """
    Function for creating a "phi" vector matrix
    Function parameters:
        - lenYparams - number of parameters from AR
        - lenUparams - number of parametr from FIR
        - inputSignal - input signal u[k]
        - outputSignal - output signal y[k]

    Requires:
        - Numpy

    Notes:
    Each row corresponds to time step k and contains: [y[k-1], y[k-2], ..., y[k-lenYparams], u[k-1], u[k-2], ..., u[k-lenUparams]]
    If any of the required past values are unavailable (k < delay), they are replaced with zeros.
    The first lenYparams columns contain delayed output values y[k-i]
    The next lenUparams columns contain delayed input values u[k-i]
    The delays are ordered from smallest (most recent) to largest (oldest)
    """
    
    if len(inputSignal) != len(outputSignal):
        print("Input and output signals must have equal length")
        return None

    lenOut = len(outputSignal)
    outputSignal = np.pad(outputSignal, (lenYparams, 0), 'constant', constant_values=(0))
    inputSignal = np.pad(inputSignal, (lenUparams, 0), 'constant', constant_values=(0))
    
    y = np.zeros(lenYparams)
    u = np.zeros(lenUparams)
    phiMatrix = np.zeros((lenOut, lenUparams + lenYparams))
    
    for k in range(0, lenOut):
        y = outputSignal[k:k+lenYparams][::-1]
        u = inputSignal[k:k+lenUparams][::-1]
            
        row = np.concatenate((y, u))
        phiMatrix[k, :] = row
    return phiMatrix

def modifGramSchmidt(V: np.ndarray) -> np.ndarray:
    '''
    Matrix orthogonalisation using Modified Gramm Schmidt Method (MGS)
    '''
    
    (n, k) = np.shape(V)
    U = np.zeros((n, k))
    U = V
    for i in range(0, k):
        U[:, i] = U[:, i]/np.linalg.norm(U[:, i])
        for j in range(i+1, k):
            U[:, j] = U[:, j] - U[:, i] * np.dot(U[:, j], U[:, i]) / np.dot(U[:, i], U[:, i]) 
                  
    return U