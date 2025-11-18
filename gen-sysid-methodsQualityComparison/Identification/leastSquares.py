import numpy as np

def ordinaryLeastSq(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares method computed by vector multiplication
    Parameters:
        phi - matrix of regressors
        y - vector of measured outputs
    Returns:
        theta - vector of estimated model parameters
    """
    
    RN = phi.T @ phi
    rN = phi.T @ y
    
    theta = np.linalg.solve(RN, rN)
    return theta

def sumOrdinLeastSq(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares using sum to compute regression matrix and
    regression vector. It calculates least squares
    estimator for a linear regression model in form:
          y = phi * theta + e
    where:
    phi - matrix of regressors
    theta - vector of parameters 
    y - vector of measured outputs
    e - vector of noise
    
    Function returns theta - vector of estimated model parameters.
    """
    
    if phi.ndim != 2:
        raise ValueError("phi must be a 2D array (matrix).")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array (vector).")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi and y must have the same number of rows.")
    
    n, m = phi.shape
    RN = np.zeros((m, m))
    rN = np.zeros(m)

    for i in range(n):
        RN += np.outer(phi[i, :], phi[i, :])  
        rN += phi[i, :] * y[i]             
        
    if np.linalg.matrix_rank(RN) < m:
        raise ValueError("Matrix RN is singular or rank-deficient.")

    theta = np.linalg.solve(RN, rN)
    
    return theta

import numpy as np

def regularizedLeastSquares(phi: np.array, signal: np.array, D: np.array, lambd: float):
    """
    signal - vector of signal (dimension N x 1)
    phi - regressors matrix (dimension N x k)
    D - regularization matrix (dimension k x k)
    lambd - regularization constant

    theta - vector of estimated parameters (dimension k x 1)

    When in argument input would be a horizontal vector, theta would be the same
    """
    
    Rn = phi.T @ phi
    rn = phi.T @ signal
    Pn = Rn + lambd * D

    theta = np.linalg.solve(Pn, rn)

    return theta    

def iterativeWeightingLeastSquares(phi: np.ndarray, y: np.ndarray, uM: float, iterNum: int, tol = 1e-6) -> np.ndarray:
    '''
    Computes parameter estimates using:
    - Least Squares Method with Iterative Weighting (IWLS)

    Parameters:
    - phi: Regression matrix
    - y: Output vector
    - uM: Regularization (weighting) factor
    - iterNum: Maximum number of iterations
    - tol: Convergence tolerance

    Returns:
    - theta: Estimated model parameters
    '''

    lmt = 1e-4

    RN = phi.T @ phi
    rN = phi.T @ y

    theta = ordinaryLeastSq(phi, y)

    for _ in range(iterNum):
        thetaPrev = theta.copy()
        D = np.diag(1.0 / (np.abs(theta) + lmt))
        theta = np.linalg.solve((RN + uM * D), rN)

        if np.linalg.norm(theta - thetaPrev) < tol:
            break

    return theta

def forwardOrthogonalLeastSquares(phi: np.ndarray, y: np.ndarray, nTerms: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    
    """
    FOLS (Forward Orthogonal Least Squares)
    
    Args:
        phi : ndarray, shape (nSamples, nFeatures) — regression matrix
        y : ndarray, shape (nSamples,) — output vector (observations)
        nTerms : int — number of parameters (features) to select

    Returns:
        selectedIdx : list — indices of the selected columns (features)
        coefficients : ndarray, shape (nFeatures,) — model coefficients (zeros for non-selected features)
        yPred : ndarray, shape (nSamples,) — predicted output
    """
    
    nSamples, nFeatures = phi.shape
    selectedIdx = []
    remainingIdx = list(range(nFeatures))
    theta = np.zeros(nFeatures)
    yPred = []

    if nTerms>nFeatures:
        nTerms = nFeatures
        theta = ordinaryLeastSq(phi, y)
        yPred = phi @ theta
        return list(range(nFeatures)), theta, yPred
    
    for step in range(nTerms):
        possibleErr = []
        possibleIndices = []

        for idx in remainingIdx:
            tempIdx = selectedIdx + [idx]
            tempPhi = phi[:, tempIdx]
            tempTheta = ordinaryLeastSq(tempPhi, y)

            tempY = tempPhi @ tempTheta
            error = np.mean((y- tempY)**2)

            possibleErr.append(error)
            possibleIndices.append(idx)

        bestIdxPos = np.argmin(possibleErr)
        bestIdx = possibleIndices[bestIdxPos]

        selectedIdx.append(bestIdx)
        remainingIdx.remove(bestIdx)

    
    theta[selectedIdx] = ordinaryLeastSq(phi[:, selectedIdx], y)

    yPred = phi @ theta

    return selectedIdx, theta, yPred

def orthogonalMatchingPursuit(collectionOfRegressors: np.ndarray, signalMeasurement: np.ndarray, k: int) -> np.ndarray:

    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse approximation.

    This function estimates a sparse representation x of a given measurement 
    signal y = D * x, where:
        - y (signalMeasurement) is the measurement vector,
        - D (collectionOfRegressors) is a matrix whose columns are potential regressors,
        - x is a sparse vector (with at most k nonzero elements) such that
          the reconstruction error is minimized.

    The OMP algorithm iteratively selects the dictionary atoms (columns of D)
    that best correlate with the current residual, then solves a least squares 
    problem over the selected atoms to update the sparse coefficient vector.

    Parameters:
    signalMeasurement : np.ndarray
        1D array representing the measurement vector y.
    dictionary : np.ndarray
        2D array (matrix) where each column is a possible feature (atom).
    k : int
        Maximum number of nonzero elements to select in the sparse solution (sparsity level).

    Returns:
    x : np.ndarray
        1D array representing the estimated sparse coefficient vector such that
        signalMeasurement ≈ dictionary @ x, with at most k nonzero entries.
    """

    residual = signalMeasurement.copy()
    selectedRegressors = []

    for _ in range(k):

        dotProduct = collectionOfRegressors.T @ residual
        bestdotProduct = np.argmax(np.abs(dotProduct))

        if bestdotProduct in selectedRegressors: 
            break
    
        selectedRegressors.append(bestdotProduct) 

        selectedRegressorsCollection = collectionOfRegressors[:, selectedRegressors]
      
        xS = ordinaryLeastSq(selectedRegressorsCollection, signalMeasurement)

        residual = signalMeasurement - selectedRegressorsCollection @ xS
        
        error = (residual**2).sum()**0.5

        if error < 1e-6:
            break

    x = np.zeros(collectionOfRegressors.shape[1])   
    x[selectedRegressors] = xS.copy()  

    return x             

