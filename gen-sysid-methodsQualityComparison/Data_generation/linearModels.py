import numpy as np

def FIRmodel(params: np.ndarray, varianceSigma: float, signal: np.ndarray, 
             noise: np.ndarray = None) -> np.ndarray:
    """
    FIR (finite impulse response) model with optional external noise.
    
    Input:
        - params: values of model parameters
        - varianceSigma: value of noise variance (used only if noise is None)
        - signal: input signal
        - noise: optional external noise array (same length as signal)

    Output:
        - output: filtered signal with added noise
    """
    amountParams = len(params)
    lenSignal = len(signal)

    toeplitzMatrix = np.zeros((lenSignal, amountParams))
    for i in range(lenSignal):
        toeplitzMatrix[i, :min(i + 1, amountParams)] = signal[max(0, i - amountParams + 1):i + 1][::-1]

    output = toeplitzMatrix @ params

    if noise is None:
        noise = np.random.normal(0, np.sqrt(varianceSigma), lenSignal)
    else:
        if len(noise) != lenSignal:
            raise ValueError(f"Noise length must be {lenSignal}, got {len(noise)}")

    output += noise

    return output



def ARmodel(params: np.ndarray, noiseVar: float, seq_length: int, initVals: np.ndarray = None,
            noise: np.ndarray = None) -> np.ndarray:
    """
    Function for generating an AutoRegressive (AR) model.

    Parameters:
        params     - coefficients of the AR model
        noiseVar   - variance of the Gaussian noise
        seq_length - number of output samples to generate
        initVals   - initial conditions for the AR model
        noise      - optional pre-defined noise vector

    """

    params = np.asarray(params)
    order = len(params)

    output = np.zeros(seq_length + order)

    if initVals is None:
        initVals = np.zeros(order)
    else:
        initVals = np.asarray(initVals)

    output[:order] = initVals[::-1]

    if noise is None:
        noise = np.random.normal(0.0, np.sqrt(noiseVar), seq_length)

    for t in range(order, seq_length + order):
        output[t] = np.dot(params, output[t - order:t][::-1]) + noise[t - order]

    return output[order:]

def ARXmodel(paramsOut: np.ndarray, paramsIn: np.ndarray, inputSignal: np.ndarray,
             noiseVar: float, noise: np.ndarray = None) -> np.ndarray:
    """
    Function for calculating ARX (autoregressive with exogenous input) model output.
    
    Parameters:
        - paramsOut : AR parameters (for past outputs)
        - paramsIn : FIR parameters (for inputs)
        - inputSignal : input signal u[t]
        - noiseVar : noise variance (used only if noise is None)
        - noise : optional external noise vector, shape = (len(inputSignal),)
    
    Returns:
        - Simulated output signal (np.array)
    """
    numOut = len(paramsOut)
    numIn = len(paramsIn)
    outLength = len(inputSignal)

    if noise is None:
        noise = np.random.normal(0, np.sqrt(noiseVar), outLength)
    else:
        if len(noise) != outLength:
            raise ValueError(f"Noise length must be {outLength}, got {len(noise)}")

    inputSignal = np.pad(inputSignal, (numIn - 1, 0), 'constant', constant_values=(0))
    output = np.zeros(outLength + numOut - 1)

    output[numOut - 1] = np.dot(paramsIn[0], inputSignal[numIn - 1]) + noise[0]
    for t in range(0, outLength - 1):
        ARPart = np.dot(paramsOut[::-1], output[t:t + numOut])
        FIRPart = np.dot(paramsIn[::-1], inputSignal[t + 1:t + numIn + 1])
        output[t + numOut] = FIRPart + ARPart + noise[t + 1]

    return output[numOut - 1:]
