import numpy as np


def isEigenVec(A, v):
    if np.all(v == 0):
        return False
    l = A.dot(v)
    idx_nonzero = (v != 0).argmax()
    eigenval = l[idx_nonzero] / v[idx_nonzero]
    v_c = eigenval * v
    return np.all(v_c == l)

if __name__ == '__main__':
    A = np.array([[8, 3], [2, 7]])
    v = np.ones(2)

    i = 0
    N = 5000
    while not isEigenVec(A, v):
        v = A.dot(v)
        v = v / np.linalg.norm(v)
        i += 1
        if i % 5 == 0:
            print('{}: {}'.format(i, v))

        if i == N:
            print('Over limit passes')
            break

    if i != N:
        print('Found eigvector(pass {}): {}'.format(i, v)) 
        np.linalg.eig
    
