import numpy as np
import matplotlib.pyplot as plt
import Data_generation.linearModels as lm  # ARmodel, ARXmodel, FIRmodel
from Identification.leastSquares import ordinaryLeastSq as LSE
import Tools.matrixTools as rm

# Display parameters
widthin = 12
hight = 4

def test_fir():
# Test parameters
    paramsDict = {
        4:  np.array([10, 5, 2, 0.1]),
        10: np.arange(9, -1, -1),
        20: np.array([40, 30, 20, 15, 10, 7.5, 5, 2.5, 2, 1.75, 1.5, 1.25, 1, 0.5, 0.1, 0, 0, 0, 0, 0])
    }
    varianceSigmaList = [0.1, 0.3, 0.5, 1, 34]
    lenghtSignalList = [70, 100, 200]

    for numberParams, parameters in paramsDict.items():
        for noiceVariance in varianceSigmaList:
            for N in lenghtSignalList:
                noiseFIR = np.random.normal(0, np.sqrt(0.01), N)
                inputFIR = np.random.rand(N)
                outputSignal = lm.FIRmodel(parameters, noiceVariance, inputFIR,noise = noiseFIR)
                phi = rm.regressionMatrix(inputFIR, outputSignal, 0, numberParams)
                theta = LSE(phi, outputSignal)
                outputLSE = lm.FIRmodel(theta, noiceVariance, inputFIR, noise = noiseFIR)

                fig = plt.figure(figsize=(widthin, hight))
                fig.canvas.manager.set_window_title(f"FIR varianceSigma: {noiceVariance}, N: {N}, params: {numberParams}")
                plt.subplot(1, 2, 1)
                plt.title("Output FIR")
                plt.plot(outputSignal, label="True output", color='blue')
                plt.plot(outputLSE, label="Estimated output", color='orange')
                plt.xticks(np.arange(0, N+1, 10))
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 2, 2)
                plt.title("True parameters vs Estimated parameters")
                plt.scatter(np.arange(len(parameters)), parameters, color="blue", label="True parameters")
                plt.scatter(np.arange(len(theta)), theta, color="orange", label="Estimated parameters")
                plt.xticks(np.arange(0, len(parameters), 1))
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.show(block=True)
                plt.close()

def test_arx():
    #Test parameters
    parameters = np.array([0.3, 0.5])
    paramsFIR = np.array([10, 5, 2, 0.1])
    lenghtSignalList = [70, 100, 200]

    for N in lenghtSignalList:
        noiseARX = np.random.normal(0, np.sqrt(0.01), N)
        inputARX = np.random.rand(N)
        outputARX = lm.ARXmodel(parameters, paramsFIR, inputARX, 0.01,noise = noiseARX)
        phi = rm.regressionMatrix(inputARX, outputARX, len(parameters), len(paramsFIR))
        theta = LSE(phi, outputARX)
        outputARXlse = lm.ARXmodel(theta[:len(parameters)], theta[len(parameters):], inputARX, 0.01,noise = noiseARX)

        fig = plt.figure(figsize=(widthin, hight))
        fig.canvas.manager.set_window_title(f"ARX test: N={N}")
        plt.subplot(1, 2, 1)
        plt.title("Output ARX")
        plt.plot(outputARX, label="True output", color="blue")
        plt.plot(outputARXlse, label="Estimated output", color="orange")
        plt.xticks(np.arange(0, len(outputARX)+1, 10))
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title("True parameters vs Estimated parameters")
        combined = np.concatenate((parameters, paramsFIR))
        plt.stem(combined, linefmt="blue", markerfmt='b', basefmt=" ", label=" True Parameters")
        plt.stem(theta, linefmt="orange", markerfmt='o', basefmt=" ", label="Estimated parameters")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show(block=True)
        plt.close()

def test_ar():
    #Test parameters
    parameters = np.array([0.3, 0.5])
    lenghtSignalList = [70, 100, 200]

    for N in lenghtSignalList:
        noiseAR = np.random.normal(0, np.sqrt(0.01), N)
        inputAR = np.zeros(N)
        outputAR = lm.ARmodel(parameters, 0.01, N,noise=noiseAR)
        phi = rm.regressionMatrix(inputAR, outputAR, len(parameters), 0)
        theta = LSE(phi, outputAR)
        outputARlse = lm.ARmodel(theta, 0.01, N,noise=noiseAR)

        fig = plt.figure(figsize=(widthin, hight))
        fig.canvas.manager.set_window_title(f"AR test: N={N}")
        plt.subplot(1, 2, 1)
        plt.title("Output AR")
        plt.plot(outputAR, label="True output", color="blue")
        plt.plot(outputARlse, label="Estimated output", color="orange")
        plt.xticks(np.arange(0, len(outputAR)+1, 10))
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title("True parameters vs Estimated parameters")
        plt.stem(parameters, linefmt='blue', markerfmt='b', basefmt=" ", label="True parameters")
        plt.stem(theta, linefmt='orange', markerfmt='o', basefmt=" ", label="Estimated parameters")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show(block=True)
        plt.close()

# ========== MAIN LOOP ==========
while True:
    print("\nWybierz test do uruchomienia:")
    print("  f - test FIR")
    print("  x - test ARX")
    print("  a - test AR")
    print("  q - zakończ program")
    user_choice = input("Twój wybór: ").lower()

    if user_choice == 'f':
        test_fir()
    elif user_choice == 'x':
        test_arx()
    elif user_choice == 'a':
        test_ar()
    elif user_choice == 'q':
        print("Zakończono program.")
        break
    else:
        print("Nieprawidłowy wybór. Spróbuj ponownie.")
