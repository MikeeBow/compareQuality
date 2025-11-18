import numpy as np
from Tools.matrixTools import regressionMatrix

def test_regressionMatrix_basic():
    u = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])

    phi = regressionMatrix(u, y, lenYparams=1, lenUparams=1)

    expected = np.array([
        [0, 0],
        [5, 1],
        [4, 2],
        [3, 3],
        [2, 4],
    ])

    if np.array_equal(phi, expected):
        print("Basic test passed.")
    else:
        print("Basic test failed.")
        print("Expected:\n", expected)
        print("Got:\n", phi)


def test_regressionMatrix_longer_params():
    u = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])

    phi = regressionMatrix(u, y, lenYparams=2, lenUparams=2)

    expected = np.array([
        [0, 0, 0, 0],
        [5, 0, 1, 0],
        [4, 5, 2, 1],
        [3, 4, 3, 2],
        [2, 3, 4, 3],
    ])

    if np.array_equal(phi, expected):
        print("Longer params test passed.")
    else:
        print("Longer params test failed.")
        print("Expected:\n", expected)
        print("Got:\n", phi)


def test_regressionMatrix_zero_input():
    u = np.zeros(5)
    y = np.zeros(5)

    phi = regressionMatrix(u, y, lenYparams=2, lenUparams=2)

    expected = np.zeros((5, 4))

    if np.array_equal(phi, expected):
        print("Zero input test passed.")
    else:
        print("Zero input test failed.")
        print("Expected:\n", expected)
        print("Got:\n", phi)


def test_regressionMatrix_different_signals():
    u = np.array([1, 0, 1, 0, 1])
    y = np.array([0, 1, 0, 1, 0])

    phi = regressionMatrix(u, y, lenYparams=1, lenUparams=2)

    expected = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ])

    if np.array_equal(phi, expected):
        print("Different signals test passed.")
    else:
        print("Different signals test failed.")
        print("Expected:\n", expected)
        print("Got:\n", phi)

def test_regressionMatrix_mismatch_length():
    u = np.array([1, 2, 3])
    y = np.array([1, 2])

    phi = regressionMatrix(u, y, lenYparams=1, lenUparams=1)

    if phi is None:
        print("Mismatch length test passed.")
    else:
        print("Mismatch length test failed.")
        print("Expected: None")
        print("Got:\n", phi)


def test_regressionMatrix_FIR():
    u = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 0, 0, 0, 0])

    phi = regressionMatrix(u, y, lenYparams=0, lenUparams=3)

    expected = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 1, 0],
        [3, 2, 1],
        [4, 3, 2],
    ])

    if np.array_equal(phi, expected):
        print("FIR input test passed.")
    else:
        print("FIR input test failed.")
        print("Expected:\n", expected)
        print("Got:\n", phi)


        
def test_regressionMatrix_AR():
    u = np.array([0, 0, 0, 0, 0])
    y = np.array([1, 2, 3, 4, 5])

    phi = regressionMatrix(u, y, lenYparams=2, lenUparams=0)

    expected = np.array([
        [0, 0],
        [1, 0],
        [2, 1],
        [3, 2],
        [4, 3],
    ])

    if np.array_equal(phi, expected):
        print("AR input test passed.")
    else:
        print("AR input test failed.")
        print("Expected:\n", expected)
        print("Got:\n", phi)

test_regressionMatrix_basic()
test_regressionMatrix_longer_params()
test_regressionMatrix_zero_input()
test_regressionMatrix_different_signals()
test_regressionMatrix_mismatch_length()
test_regressionMatrix_FIR()
test_regressionMatrix_AR()