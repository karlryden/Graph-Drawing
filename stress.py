import math, random, queue
import numpy as np
import matplotlib.pyplot as plt

k = 25
l = 5
q = 8
m = 1
k_e = 8
g = 10

class graph:
    def __init__(self, A=0):
        if A == 0:
            self.A = np.array([[0]])
            self.N = 1
            self.D = np.array([[0]])
            self.B = np.array([[0]])
            self.X = np.array([np.array([random.random(), random.random()])])

    def nbrs(self, i):
        return [j for j in range(self.N) if self.A[i][j] != 0]

    def dist(self, i, j):
        u = self.X[i]
        v = self.X[j]

        return np.linalg.norm(v - u)

    def bfs(self, i, j):
        q = queue.Queue()
        q.put((i, 0))
        ret = 0
        visited = set()
        while not q.empty():
            now, steps = q.get()
            if now == j:
                return steps

            visited.add(now)

            for nbr in self.nbrs(now):
                if nbr not in visited:
                    q.put((nbr, steps+1))


    def add(self, *nbrs):
        self.N += 1
        i = self.N-1
        x = np.zeros((1, i))
        self.A = np.concatenate((self.A, x), axis=0)
        self.D = np.concatenate((self.D, x), axis=0)
        self.B = np.concatenate((self.B, x), axis=0)
        y = np.zeros((self.N, 1))
        self.A = np.concatenate((self.A, y), axis=1)
        self.D = np.concatenate((self.D, y), axis=1)
        self.B = np.concatenate((self.B, y), axis=1)
        self.X = np.concatenate((self.X, np.array([[random.random(), random.random()]])), axis=0)

        for nbr in nbrs:
            self.A[i][nbr] = 1
            self.A[nbr][i] = 1

        for j in range(i):
            d = self.dist(i, j)
            self.D[i][j] = d
            self.D[j][i] = d

            b = self.bfs(i, j)
            self.B[i][j] = b
            self.B[j][i] = b

    def sigma(self):
        ret = 0
        for i in range(self.N - 2):
            for j in range(i + 1, self.N - 1):
                ret += (self.D[i][j] - self.B[i][j])**2

        return ret

    def tau(self, Z):
        def B(Z):
            ret = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    b = -self.B[i][j] / self.D[i][j]
                    ret[i][j] = b
                    ret[j][i] = b

            return ret

        return self.sigma() + 2 * np.trace(np.dot(self.X.T, np.dot(B(self.X), self.X))) - 2 * np.trace(np.dot(self.X.T, np.dot(B(Z), Z)))

    def plot(self):
        X = [p[0] for p in self.P]
        Y = [p[1] for p in self.P]
        
        plt.scatter(X, Y, linewidth=8, color='k')
        for i in range(self.N):
            plt.annotate(f"  {i}", (X[i], Y[i]), fontsize=12)
            for j in range(i, self.N):
                if self.M[i][j] != 0:
                    plt.plot([self.P[i][0], self.P[j][0]], [self.P[i][1], self.P[j][1]], color='k')


if __name__ == '__main__':
    import time
    t0 = time.time()
    G = graph()
    print(G.A)
    print(G.D)
    print(G.B)
    t1 = time.time()
    print(f"init-time: {t1-t0}")
    G.add(0)    # 1
    G.add(1)    # 2
    G.add(2)    # 3
    G.add(2)    # 4
    G.add(3, 4) # 5
    print(G.A)
    print(G.D)
    print(G.B)
    t2 = time.time()
    print(f"add-time: {t2-t1}")
    '''
    print(G.sigma())
    t3 = time.time()
    print(f"sigma-time: {t3-t2}")
    print(G.tau(np.eye(G.N)))
    t4 = time.time()
    print(f"tau-time: {t4-t3}")
    '''
    eps = 0.1
    for _ in range(50):
        Z = G.X



