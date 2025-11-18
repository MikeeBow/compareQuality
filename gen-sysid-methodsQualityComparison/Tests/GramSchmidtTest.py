import numpy as np
from Tools.matrixTools import modifGramSchmidt

'''
Test for Modified Gram Schmidt orthogonalisation method (MGS)
In order to check if matrix is truly orthogonal we use the property:
    A.T*A = I
where: A.T is transposition of matrix A
'''

count = 1

def checkLinIndepend(matrix):
    """
    Check if the given matrix has linearly independent columns.
    """
    rank = np.linalg.matrix_rank(matrix)
    if rank == matrix.shape[1]:
        return True
    else:
        return False
    
def checkOrthogonality(matrix):
    """
    Check if the given matrix is orthogonal.
    """
    global count
    if(checkLinIndepend(matrix) == False):
        print(f'Matrix {count} consists of linearly dependent columns,\n',
              'cannot be orthonormalised\n')
        count += 1
    else:
        shape = np.shape(matrix)
        isOrthogonal = np.allclose(np.dot(np.transpose(matrix), matrix), np.identity(shape[1]))
        print(f'Matrix {count} is orthogonal: {isOrthogonal}\n')
        count += 1
    
# 2x2 Matrix
V = np.array([[2., 2.], [1., 3.]])
shape = np.shape(V)
V = modifGramSchmidt(V)

print(V)
checkOrthogonality(V)

# 3x3 Matrix
V = np.array([[2, 3, 4],[5, 1, 2.5], [4, 1.25, 6]])
shape = np.shape(V)
V = modifGramSchmidt(V)

print(V)
checkOrthogonality(V)

# Random generated matrix for defined size
size = 100
V = np.random.rand(size, size)
V = modifGramSchmidt(V)

checkOrthogonality(V)

# Random generated rectangular matrix 10x4
V = np.random.rand(4, 6)
V = modifGramSchmidt(V)

checkOrthogonality(V)

# Random generated rectangular matrix 10x4
V = np.random.rand(10, 4)
V = modifGramSchmidt(V)

checkOrthogonality(V)