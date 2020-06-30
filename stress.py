import math, random, queue
import numpy as np
import matplotlib.pyplot as plt

def one(n, k):
    ret = np.zeros((n, 1))
    ret[k][0] = 1
    return ret

class graph:
    def __init__(self, A=np.array([[0]])):
        self.delta = 1000

        if len(A) == 1:
            self.N = 1
            self.X = np.array([np.array([random.random(), random.random()])])
            self.B = np.array([[0]])
            self.V = np.array([[0]])

        else:
            self.A = A
            self.N = len(A)
            self.X = np.random.rand(self.N, 2)
            self.B = np.zeros((self.N, self.N))
            for i in range(self.N - 1):
                for j in range(i + 1, self.N):
                    b = self.bfs(i, j)
                    self.B[i][j] = b
                    self.B[j][i] = b

            self.V = np.zeros((self.N, self.N))
            for i in range(self.N - 1):
                for j in range(i + 1, self.N):
                    self.V += self.w(i, j)*np.dot((one(self.N, i) - one(self.N, j)), (one(self.N, i) - one(self.N, j)).T)

    def nbrs(self, i):
        return [j for j in range(self.N) if self.A[i][j] != 0]

    def dist(self, i, j):
        u = self.X[i]
        v = self.X[j]

        return np.linalg.norm(v - u)

    def w(self, i, j):
        return self.delta*1/(self.B[i][j])**2

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

    def add(self, *nbrs):
        self.N += 1
        i = self.N - 1
        x = np.zeros((1, i))
        
        # Extend X
        self.X = np.concatenate((self.X, np.array([[random.random(), random.random()]])), axis=0)

        # Extend A, B
        self.A = np.concatenate((self.A, x), axis=0)
        self.B = np.concatenate((self.B, x), axis=0)

        y = np.zeros((self.N, 1))
        self.A = np.concatenate((self.A, y), axis=1)
        self.B = np.concatenate((self.B, y), axis=1)
        
        for nbr in nbrs:
            self.A[i][nbr] = 1
            self.A[nbr][i] = 1

        for j in range(i):
            b = self.bfs(i, j)
            self.B[i][j] = b
            self.B[j][i] = b

        #Extend V
        self.V = np.zeros((self.N, self.N))
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                    self.V += self.w(i, j)*np.dot((one(self.N, i) - one(self.N, j)), (one(self.N, i) - one(self.N, j)).T)

    def sigma(self):
        ret = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                ret += self.w(i, j)*(self.dist(i, j) - self.B[i][j])**2

        return ret

    def F(self):
        ret = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    ret[i][j] = -self.w(i, j) * self.B[i][j] / self.dist(i, j)

        for i in range(self.N):
            ret[i][i] = -np.sum(ret[i,:])

        return np.dot(ret, self.X)

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
    import time
    t0 = time.time()
    '''
    G = graph()
    G.add(0)    # 1
    G.add(1)    # 2
    G.add(2)    # 3
    G.add(2)    # 4
    G.add(3, 4) # 5
    '''

    n = 25
    rho = 0.66
    rnd = np.vectorize(round)
    G = graph(rnd(rho * np.random.rand(n, n)))
    print(G.A)

    eps = 0.1
    for _ in range(100):
        plt.clf()
        alpha = G.sigma()
        
        X_star = np.dot(np.linalg.pinv(G.V), G.F())
        G.X = X_star
        
        beta = G.sigma()
        
        print(alpha - beta)
        G.plot()


        if alpha - beta < eps:
            break
    
        plt.pause(0.01)

plt.show()