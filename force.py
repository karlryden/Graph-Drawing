import math, random, queue
import numpy as np
import matplotlib.pyplot as plt

k = 8
l = 1
q = 2
m = 1
k_e = 0.1

class graph:
    def __init__(self, A):
        self.A = A
        self.N = len(A)
        self.M = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in self.A[i]:
                self.M[i][j] = 1

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
        q.offer(i)
        ret = 0
        visited = set()
        while not q.empty():
            n = q.pop()
            if n == j:
                return ret

            visited.add(n)
            ret += 1

            for nbr in self.nbrs(n):
                if nbr not in visited:
                    q.offer(nbr)

    def arg(self, i, j):
        u = self.P[i]
        v = self.P[j]
        w = u - v

        return math.atan2(w[1], w[0])

    def hooke(self, i, j):
        return -k * (-l + self.d(i, j))
        #return -k * (l - math.log(self.d(i, j)))

    def coulomb(self, i, j):
        return k_e * q**2 / (self.d(i, j)**2)

    def res(self, i):
        ret = np.array([0.0, 0.0])
        for j in self.nbrs(i):
            theta = self.arg(i, j)
            S = self.hooke(i, j)
            C = self.coulomb(i, j)
            print(f"S: {S}, C: {C}")
            ret += (S - C) * np.array([math.cos(theta), math.sin(theta)])

        return ret

    def plot(self):
        X = [p[0] for p in self.P]
        Y = [p[1] for p in self.P]

        plt.scatter(X, Y, linewidth=8, color='k')
        for i in range(self.N):
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

    t = 0.01
    for _ in range(1000):
        #print(G.P)
        G.plot()
        for i in range(G.N):
            a = G.res(i) * 1/m
            G.V[i] += t * a
            G.P[i] += t * G.V[i]

        if np.linalg.norm(a) < 0.25:
            break

        plt.pause(0.01)
        plt.clf()
    plt.show()
