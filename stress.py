import math, random, queue
import numpy as np
import matplotlib.pyplot as plt

def one(n, k):
    ret = np.zeros((n, 1))
    ret[k][0] = 1
    return ret

class graph:
    def __init__(self, A=np.array([[0]])):
        self.delta = 1000   # Constant emphasizing stress due to graph-theoretical distance between nodes
        self.A = A

        # Default constructor
        if len(A) == 1:
            self.N = 1
            self.X = np.array([np.array([random.random(), random.random()])])
            self.B = np.array([[0]])
            self.V = np.array([[0]])

        # Constructor using given adjacency matrix
        else:
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
        return self.delta*1/(self.B[i][j])**2

    # Add node to graph
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

        # Remake V
        self.V = np.zeros((self.N, self.N))
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                self.V += self.w(i, j)*np.dot((one(self.N, i) - one(self.N, j)), (one(self.N, i) - one(self.N, j)).T)

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
        X_star = np.dot(np.linalg.pinv(self.V), self.F())
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
            #plt.annotate(f"  {i}", (X[i], Y[i]), fontsize=12)
            for j in range(i, self.N):
                if self.A[i][j] != 0:
                    plt.plot([self.X[i][0], self.X[j][0]], [self.X[i][1], self.X[j][1]], color='k')

if __name__ == '__main__':
    '''
    G = graph()
    G.add(0)    # 1
    G.add(1)    # 2
    G.add(2)    # 3
    G.add(2)    # 4
    G.add(3, 4) # 5
    '''

    
    n = 16
    rho = 0.8  # Edge density of random graph
    rnd = np.vectorize(round)
    A = rnd(rho*np.random.rand(n, n))
    
    # Connect disconnected nodes
    for i in range(n):
        if all([A[i][j] == 0 for j in range(n)]):
            k = random.randint(0, n-1)
            A[i][k] = 1

    G = graph(A)
    G.SMACOF()
    G.plot()
    plt.show()


    '''
    while True:        
        plt.clf()
        alpha = G.sigma()
        G.majorize()
        beta = G.sigma()
            
        print(f"stress: {beta}")
        G.plot()

        # Break if stress reduction is smaller than threshold
        if alpha - beta < eps:
            break
        
        plt.pause(0.01)

    plt.show()
    '''