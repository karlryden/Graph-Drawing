import numpy as np
import matplotlib.pyplot as plt

class graph:
    def __init__(self, A):
        self.A = A
        self.N = len(A)

    def eig(self):
        return np.linalg.eig(self.A)[1]

    def laplacian(self):
        D = []
        for u in self.A:
            delta = 0
            for v in u:
                if v != 0:
                    delta += 1

            D.append([delta])

        return np.array(D) - self.A

    def cart(self):
        ret = []
        [lambdas, eigs] = np.linalg.eig(self.laplacian())
        [l2, l1] = sorted(lambdas)[:2]
        i = np.where(lambdas == l1)
        j = np.where(lambdas == l2)
        x1 = eigs[i]
        x2 = eigs[j]
        for k in range(self.N):
            ret.append(np.array([x1[0][k], x2[0][k]]))

        return ret

    def plot(self):
        X = [pos[0] for pos in self.cart()]
        Y = [pos[1] for pos in self.cart()]

        plt.scatter(X, Y, linewidth=8, color='k')
        for i in range(self.N):
            for j in range(i, self.N):
                if self.A[i][j] != 0:
                    plt.plot([X[i], X[j]], [Y[i], Y[j]], color='k')

        plt.show()

if __name__ == '__main__':
    import random
    G = graph(np.array([[0, 1, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 1, 1, 0]]))

    #print(np.linalg.eig(A)[1])
    print(G.laplacian())
    #G.plot()

    G1 = graph(np.array([[0, 1, 0, 0, 0],
                         [1, 0, 1, 1, 0],
                         [0, 1, 0, 0, 1],
                         [0, 1, 0, 0, 1],
                         [0, 0, 1, 1, 0]]))

    #G1.plot()

    N = 25
    A = np.random.rand(N, N)
    A = np.round(0.75 * A)
    G2 = graph(A)
    #G2.plot()

    B = np.diagflat((N-1)*[1], -1) + np.diagflat((N-1)*[1], 1)
    B[0][N-1] = 1
    B[N-1][0] = 1
    G3 = graph(B)
    G3.plot()
