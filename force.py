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
    def __init__(self, A):
        self.A = A
        self.N = len(A)
        self.M = np.zeros((self.N, self.N))
        self.D = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in self.A[i]:
                self.M[i][j] = 1

                if j > i:
                    d = self.bfs(i, j)
                    self.D[i][j] = d
                    self.D[j][i] = d

        positions = []
        velocities = []
        for _ in range(self.N):
            positions.append(np.array([random.random(), random.random()]))
            velocities.append(np.array([0.0, 0.0]))

        self.P = np.array(positions)
        self.V = np.array(velocities)

    def nbrs(self, i):
        return self.A[i]

    def d(self, i, j):
        u = self.P[i]
        v = self.P[j]

        return np.linalg.norm(u - v)

    def bfs(self, i, j):
        q = queue.Queue()
        q.put(i)
        ret = 0
        visited = set()
        while not q.empty():
            n = q.get()
            if n == j:
                return ret

            visited.add(n)
            ret += 1

            for nbr in self.nbrs(n):
                if nbr not in visited:
                    q.put(nbr)

    def arg(self, i, j):
        u = self.P[i]
        v = self.P[j]
        w = v - u

        return math.atan2(w[1], w[0])

    def hooke(self, i, j):
        return -k * (l - self.d(i, j))
        #return -k * (l - math.log(self.d(i, j)))
        #return -k * (self.bfs(i, j) - self.d(i, j))

    def coulomb(self, i, j):
        return k_e * q**2 / (self.d(i, j)**2)

    def res(self, i):
        ret = np.array([0.0, 0.0])
        for j in range(self.N):
            if i != j:
                theta = self.arg(i, j)
                C = self.coulomb(i, j)
                S = 0
                if j in self.nbrs(i):
                    S = self.hooke(i, j)

                ret += (S - C) * np.array([math.cos(theta), math.sin(theta)])
                #alpha = math.atan2(self.P[i][1], self.P[i][0])
                #ret -= g * np.array([math.cos(alpha), math.sin(alpha)])
        print(f"F{i}: {ret}")

        return ret

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
    G = graph([[1],
               [0, 2],
               [1, 3, 4],
               [2, 5],
               [2, 5],
               [3, 4]])

    print(G.D)
    '''
    t = 0.01
    
    for _ in range(1000):
        #print(G.P)
        plt.clf()
        G.plot()
        A = np.array([0.0, 0.0])
        for i in range(G.N - 1):
            a = G.res(i) * 1/m
            A += a
            G.V[i] += t * a
            G.P[i] += t * G.V[i]

        if np.linalg.norm(A) < 1:
            break

        plt.pause(0.01)
    plt.show()
    '''
