"""
Comparison test for 2 ordinary least squares realisations.
Compares estimated parameters from both methods and checks if they are similar: 
difference between them lower than 10^(-6). 
Compares time of execution for different number of parameters and distinct data lengths.
ordinaryLeastSq -> using vector multiplication:
	theta =  RN^(-1) * rN

sumOrdinLeastSq -> using sum to compute RN and rN matrices:
		RN = sum(Phi[k] * Phi[k].T)
		rN = sum(Y[k] * Phi[k] )
		k = 0, ..., N
   
where:
	RN = phi.T * phi <- regression matrix
	rN = phi.T * y <- vector of regression
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from Data_generation.linearModels import FIRmodel, ARmodel, ARXmodel
from Identification.leastSquares import ordinaryLeastSq, sumOrdinLeastSq
from Tools.matrixTools import regressionMatrix

testNum = 0

def estimateParams(u: np.ndarray, y: np.ndarray, lenYparams: np.float32,
                   lenUparams: np.float32, printResult: bool = True
				   ) -> np.float32:
	global testNum
	
	print(f"\nTest {testNum+1}:")	
	phi = regressionMatrix(u, y, lenYparams, lenUparams)
 
	Tstart = time.time()
	thetaVect = ordinaryLeastSq(phi, y)
	Tend = time.time()
	vectTime = Tend - Tstart

	Tstart = time.time()
	thetaSum = sumOrdinLeastSq(phi, y)
	Tend = time.time()
	sumTime = Tend - Tstart

	if(thetaVect-thetaSum < 1e-6).all():
		print(f"passed")
		if(printResult == True):
			print(thetaVect)
	else:
		(print(f"failed"))
		print(thetaVect)
		print(thetaSum)

	testNum += 1
	return vectTime, sumTime

vectTimeCurrent = 0
sumTimeCurrent = 0

print("\nFIR model tests:")

# Test if methods return parameters similar to the given ones
# FIR model test 1
u = np.concatenate([np.array([1]), np.zeros(99)])
y = FIRmodel(np.array([1, -2.2, 0.1]), 0, u)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 0, 3)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# FIR model test 2
u = np.concatenate([np.array([1]), np.zeros(99)])
y = FIRmodel(np.array([1, -2.2, 0.1, 0.6, 1.1, 0.9, -1]), 0, u)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 0, 7)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# FIR model test 3
u = np.concatenate([np.array([1]), np.zeros(99)])
y = FIRmodel(np.array([1, -2.2, 0.1, 0.6, 1.1,
                        0.2, 1.3, -2, -0.2, -0.6,
                        1.1, 0.3, -0.4, 0.9, 0.2]),
              			0, u)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 0, 15)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")


print("\nExecution time for different number of parameters:")

testNum = 0
size = 512
steps = 9

paramNum = np.round(np.geomspace(2, size, steps)).astype(int)
print(paramNum)
vectTime = np.zeros(steps)
sumTime = np.zeros(steps)
params = np.random.uniform(-2, 2, size)

u = np.ones(2000)
for i in paramNum:
	y = FIRmodel(params[:i], 0, u)
	y = np.concatenate([np.array([0]), y[:-1]])

	vectTime[testNum-1], sumTime[testNum-1] = estimateParams(u, y, 0, i, False)
	print(f"Vector OLS time: {vectTime[testNum-1]} s,\nSum OLS time:{sumTime[testNum-1]} s")
 
# Plot result
print(paramNum)
fig = plt.subplots()
plt.plot(paramNum, vectTime, 'b.', label="Vector OLS")
plt.plot(paramNum, sumTime, 'r.', label="Sum OLS")
plt.xscale('log', base=2)
plt.xticks(paramNum, paramNum)
plt.xlabel("Number of parameters")
plt.ylabel("Time [s]")
plt.title("Time of both realisations for different number of parameters")
plt.legend(loc="upper left")
plt.grid()
plt.figure(1)

print("\nExecution time for different number of data:")

testNum = 0

points = np.arange(500, 10500, 500)
steps = points.shape[0]
vectTime = np.zeros(steps)
sumTime = np.zeros(steps)

params = np.random.uniform(-2, 2, 50)
input = np.ones(5000)

for i in points:
	u = input[:i]
	y = FIRmodel(params, 0, u)
	y = np.concatenate([np.array([0]), y[:-1]])

	vectTime[testNum-1], sumTime[testNum-1] = estimateParams(u, y, 0, 50, False)
	print(f"Vector OLS time: {vectTime[testNum-1]} s,\nSum OLS time:{sumTime[testNum-1]} s")

# Plot result
fig = plt.subplots()
plt.plot(points, vectTime, 'bx', label="Vector OLS")
plt.plot(points, sumTime, 'rx', label="Sum OLS")
plt.xlabel("Number of data points")
plt.ylabel("Time [s]")
plt.title("Time of both realisations for different number of data points")
plt.legend(loc="upper left")
plt.grid()
plt.figure(2)
plt.show()

testNum = 0

# Different noise variance
# FIR model test 1
u = np.ones(500)
y = FIRmodel(np.array([-0.2, 1.9, -2, -0.1, 1.4, -0.4, 2]), 0.1, u)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 0, 7)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# FIR model test 2
u = np.ones(500)
y = FIRmodel(np.array([-0.2, 1.9, -2, -0.1, 1.4, -0.4, 2]), 0.5, u)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 0, 7)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# FIR model test 3
u = np.ones(500)
y = FIRmodel(np.array([-0.2, 1.9, -2, -0.1, 1.4, -0.4, 2]), 1, u)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 0, 7)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")


testNum = 0
print("\nAR model tests:")

# AR model test 1
y = ARmodel(np.array([0.5, -0.5, 1]), 0, 100, np.array([1, 0, 0]))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# AR model test 2
y = ARmodel(np.array([0.5, -0.5, 1]), 0, 1000, np.array([1, 0, 0]))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# AR model test 3
y = ARmodel(np.array([0.5, -0.5, 1]), 0, 10000, np.array([1, 0, 0]))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# AR model test 4
y = ARmodel(np.array([1.2]), 0, 1000, np.ones(1))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 1, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# AR model test 5
y = ARmodel(np.array([0.5, -0.5, 0.4]), 0, 1000, np.ones(3))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# AR model test 6
y = ARmodel(np.array([0.25, -0.1, 0.1, 0.5, -0.7]), 0, 1000, np.ones(5))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 5, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# AR model test 7
y = ARmodel(np.array([0.5, -0.5]), 0.15, 10000, np.ones(2))
y = np.concatenate([np.array([0]), y[:-1]])
u = np.zeros(y.shape)

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 2, 0)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")


testNum = 0
print("\nARX model tests:")

# ARX model test 1
u = np.ones(100)
y = ARXmodel([-1, 0.6, 0.9], [-0.9, -0.5], u, 0)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 2)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# ARX model test 2
u = np.ones(1000)
y = ARXmodel([-1, 0.6, 0.9], [-0.9, -0.5], u, 0)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 2)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# ARX model test 3
u = np.ones(7)
y = np.array([0, 0.5, 1.75, 3, 3, 3.1, 2])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 2)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# ARX model test 4
u = np.concatenate([np.ones(1), np.zeros(19)])
y = ARXmodel([0.2, -1.2], [-1, 1.5, -0.7, 0.75], u, 0)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 2, 4)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# ARX model test 5
u = np.concatenate([np.array([1]), np.zeros(49)])
y = ARXmodel([0.2, -1.2, 0.65], [-1, 1.5, 0.7, -0.75, -1, 0.65, -1.2, 0.8], u, 0)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 3, 8)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")

# ARX model test 6
u = np.concatenate([np.array([1]), np.random.uniform(-2, 2, 50)])
y = ARXmodel([0.2, -1.2, 0.65, -1, 1.3, -0.2], [0.1, -1, 0.2, -0.75, 2], u, 0)
y = np.concatenate([np.array([0]), y[:-1]])

vectTimeCurrent, sumTimeCurrent = estimateParams(u, y, 6, 5)
print(f"Vector OLS time: {vectTimeCurrent} s,\nSum OLS time:{sumTimeCurrent} s")




