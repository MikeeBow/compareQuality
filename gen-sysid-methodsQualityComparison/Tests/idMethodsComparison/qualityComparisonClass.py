import numpy as np
from Data_generation.linearModels import FIRmodel
from Tools.matrixTools import regressionMatrix
from Identification.leastSquares import ordinaryLeastSq
from Identification.leastSquares import regularizedLeastSquares
from Identification.leastSquares import iterativeWeightingLeastSquares
from Identification.leastSquares import forwardOrthogonalLeastSquares
from Identification.leastSquares import orthogonalMatchingPursuit

'''
TODO:
- implement more methods for estimation in test
- adapt to be useful for second test
- optimise code

- add execution time measurments
- add AR and ARX to generate output signal
- add possibility to estimate more or less parameters than in original model
- spróbować skrócić output = concatenate w init model 
'''

class qualityComp:
    '''
    Class for quality comparison of different model parameters
    estimation methods
    '''
    def __init__(self, modParams: np.ndarray, desiredSNR: int,
                 noiseRealizations: int = 100):
        self.modParams = modParams
        self.desiredSNR = desiredSNR
        self.noiseRealisations = noiseRealizations
        
        self.inSignal = None
        self.output = None
        self.estimatedParams = None
        self.meanSqErr = 0
    
    def genInSignalAR(self, inParams: np.ndarray, u0: np.ndarray, 
                    inputVar: float, dataNum: int) -> np.ndarray:
        '''
        Generates autoregressive input signal in form of:
        u(t) = a1*u(t-1) + a2*u(t-2) + ... + an*u(t-n) + v(t)
        where:
            - v(t) is Gaussian noise with variance inputVar
            - a1, a2, ..., an are model parameters in inParams
            - u0 is initial conditions array
        '''
        if (len(inParams) != len(u0)):
            raise ValueError("Wrong number of initial conditions")
        
        order = len(inParams)
        inNoise = np.random.normal(0.0, np.sqrt(inputVar), dataNum)
        
        self.inSignal = np.concatenate([u0, np.zeros(dataNum)])
        for t in range(0, dataNum):
            self.inSignal[t+order] = np.dot(inParams[::-1], self.inSignal[t:t + order])\
                + inNoise[t]
                
        self.inSignal = self.inSignal[order:]
    
    def genInSignalSin():
        '''
        
        '''
        pass
        
    def initModel(self):
        '''
        Calculates noise variance for desired SNR and generates 
        output signal (FIR model) based on model paramerters - modParams
        '''
        y = FIRmodel(self.modParams, 0, self.inSignal)
        
        outVar = np.var(y)
        noiseVar = outVar/(10**(self.desiredSNR/10))
        
        self.output = y + np.random.normal(0, noiseVar, (self.noiseRealisations, len(y)))
        self.output = np.concatenate([np.zeros((len(self.output), 1)), self.output[:, :-1]], axis=1)
        return self.output
    
    def modelRequiredSignals(self):

        #LSMR
        self.D = np.eye(len(self.modParams))
        self.lambd = 1

        #LSMIW
        self.uM  = 1
        self.iterNum = len(self.modParams)
        self.tol = 1e-4

        #FOLS
        self.nTerms = 5

        #OMP
        self.k = 1

        return
    def estimation(self):
        '''
        Estimates model parameters based on input and output signals for:
        - Ordinary Least Squares Method (OLS)
        - Orthogonal Matching Pursuit (OMP)
        - Least Squares with Iterative Weighting (IWLS)
        - Regurarized Least Squares (RLS)
        - Fast Orthogonal Least Squares (FOLS)

        '''
        self.estimatedParams = np.zeros((5, self.noiseRealisations, len(self.modParams)))
        for n in range(0, self.noiseRealisations):
            phi = regressionMatrix(self.inSignal, self.output[n], 0, len(self.modParams))

            self.estimatedParams[0][n] = ordinaryLeastSq(phi, self.output[n])       
            self.estimatedParams[1][n] = regularizedLeastSquares(phi, self.output[n], self.D, self.lambd)
            self.estimatedParams[2][n] = iterativeWeightingLeastSquares(phi, self.output[n], self.uM, self.iterNum, self.tol)
            _, theta_fols, _ = forwardOrthogonalLeastSquares(phi, self.output[n], self.nTerms)
            self.estimatedParams[3][n] = theta_fols
            self.estimatedParams[4][n] = orthogonalMatchingPursuit(phi, self.output[n], self.k)

    def qualityTest(self):
        '''
        Uses mean square error (MSE) to measure quality of given method 
        '''
        for n in range(0, self.noiseRealisations):
            self.meanSqErr += (self.estimatedParams[0, n]-self.modParams).T \
                                @ (self.estimatedParams[0, n]-self.modParams)
               
        self.meanSqErr *= (1/self.noiseRealisations)          
                
    def getResults(self):
        '''
        Prints and plots test results
        '''
        pass
    
    
    