import math, random, queue
import numpy as np
import matplotlib.pyplot as plt

class graph:
    def __init__(self, A=np.array([[0]])):
        self.delta = 1000   # Constant emphasizing stress due to graph-theoretical distance between nodes
        self.A = np.zeros((1, 1))
        self.N = 1
        self.X = np.random.rand(self.N, 2)
        self.B = np.zeros((1, 1))

        n = len(A)
        if n > 1:           # Adds more nodes if adjacency matrix is given to constructor
            for i in range(1, n):
                self.add([j for j in range(i) if A[i][j]])

    # Neighbors of node i
    def nbrs(self, i):
        return [j for j in range(self.N) if self.A[i][j] != 0]

    # Euclidean distance from node i to node j
    def dist(self, i, j):
        u = self.X[i]
        v = self.X[j]

        return np.linalg.norm(v - u)

    # Breadth first search from node i to node j
    def bfs(self, i, j):
        q = queue.Queue()
        q.put((i, 0))
        visited = set()
        while not q.empty():
            now, steps = q.get()
            if now == j:
                return steps

            visited.add(now)

            for nbr in self.nbrs(now):
                if nbr not in visited:
                    q.put((nbr, steps+1))

    # Weight function proportional to graph-theoretical distance between node i and node j
    def w(self, i, j):
        return self.delta*1/(self.B[i][j]**2)

    # Add node to graph
    def add(self, nbrs):
        # Extend X
        self.X = np.concatenate((self.X, np.array([[random.random(), random.random()]])), axis=0)

        # Extend A, B
        x = np.zeros((1, self.N))
        self.A = np.concatenate((self.A, x), axis=0)
        self.B = np.concatenate((self.B, x), axis=0)

        y = np.zeros((self.N+1, 1))
        self.A = np.concatenate((self.A, y), axis=1)
        self.B = np.concatenate((self.B, y), axis=1)

        # Update A
        for nbr in nbrs:
            self.A[self.N][nbr] = 1
            self.A[nbr][self.N] = 1

        # Update B
        for j in range(self.N):
            b = self.bfs(self.N, j)
            self.B[self.N][j] = b
            self.B[j][self.N] = b

        self.N += 1
        
        # Remake V, V^-1
        self.V = np.zeros((self.N, self.N))
        E = np.eye(self.N)
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                [ei, ej] = [E[:,k][np.newaxis].T for k in [i, j]]
                self.V += self.w(i, j)*np.dot((ei - ej), (ei - ej).T)

        self.V_inv = np.linalg.pinv(self.V)

    # Stress function
    def sigma(self):
        ret = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                ret += self.w(i, j)*(self.dist(i, j) - self.B[i][j])**2

        return ret

    # B(Z)Z
    def F(self):
        ret = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    ret[i][j] = -self.w(i, j) * self.B[i][j] / self.dist(i, j)

        for i in range(self.N):
            ret[i][i] = -np.sum(ret[i,:])

        return np.dot(ret, self.X)

    # Minimizes stress using majorizer
    def majorize(self, eps=5):
        # Update X using majorizer function from Cauchy-Schwarz inequality
        X_star = np.dot(self.V_inv, self.F())
        self.X = X_star

    # Repeatedly majorizes stress until local minimum is reached
    def SMACOF(self, eps=5):
        while True:
            alpha = self.sigma()
            self.majorize()
            beta = self.sigma()
            
            # Break if stress reduction is smaller than threshold
            if alpha - beta < eps:
                break

    # Plots graph
    def plot(self):
        X = self.X[:,0]
        Y = self.X[:,1]
        
        plt.scatter(X, Y, linewidth=8, color='k')
        for i in range(self.N):
            plt.annotate(f"  {i}", (X[i], Y[i]), fontsize=12)
            for j in range(i, self.N):
                if self.A[i][j] != 0:
                    plt.plot([self.X[i][0], self.X[j][0]], [self.X[i][1], self.X[j][1]], color='k')


if __name__ == '__main__':

    G = graph()
    G.add([0])    # 1
    G.add([1])    # 2
    G.add([2])    # 3
    G.add([2])    # 4
    G.add([3, 4]) # 5
    G.add([5])
    G.add([6])
    G.add([5, 6])
    G.add([7, 2])
    G.add([4])
    G.add([0, 10])    

    '''
    A = np.array([[0, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 0]])

    G = graph(A)
    '''
    print(G.A)
    '''
    G.SMACOF()
    G.plot()
    plt.show()
    '''

    # Animates majorization process
    import time
    eps = 1
    while True:        
        plt.clf()
        alpha = G.sigma()
        t0 = time.time()
        G.majorize()
        beta = G.sigma()
        t1 = time.time()
        print(f"Majorizing time: {t1-t0}")
        print(f"stress: {beta}")
        G.plot()

        # Break if stress reduction is smaller than threshold
        if alpha - beta < eps:
            break
        
        plt.pause(0.01)

    plt.show()
