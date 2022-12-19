import numpy as np 
import matplotlib.pyplot as plt 
def generate_data(n: int) -> np.ndarray: 
    x = np.linspace(0, 1, n) 
    x = x.reshape(len(x), 1)
    y = np.sin(2 * np.pi * x)
    return x, y

class NN:
    def __init__(self, noinput: int, nohidden: int, nooutput: int, num_layers: int) -> None:
        self.noinput = noinput 
        self.nohidden = nohidden 
        self.nooutput = nooutput 
        self.num_layers = num_layers 
        self.w = []
        self.b = []
        self.gradw = []
        self.gradb = []
        self.deltas = []
        self.initWeights()
        self.initBiases()

    def initWeights(self):
        self.w.append(np.random.randn(self.noinput, self.nohidden))
        for i in range(self.num_layers - 1):
            self.w.append(np.random.randn(self.nohidden, self.nohidden))
        self.w.append(np.random.randn(self.nohidden, self.nooutput))

    def initBiases(self):
        self.b.append(np.zeros((1, self.nohidden)))
        for i in range(self.num_layers - 1):
            self.b.append(np.zeros((1, self.nohidden)))
        self.b.append(np.zeros((1, self.nooutput)))

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def activation_deriv(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(self.activation(x), 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = []
        self.a = []
        self.z.append(x)
        self.a.append(x)
        for i in range(self.num_layers):
            self.z.append(np.dot(self.a[i], self.w[i]) + self.b[i])
            self.a.append(self.activation(self.z[i + 1]))
        return self.a[-1]

    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.power(self.predict(x) - y, 2))

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gradw = []
        self.gradb = []
        self.deltas = []
        self.deltas.append(self.a[-1] - y)
        self.gradw.append(np.dot(self.a[-2].T, self.deltas[-1]))
        self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))

        for i in range(self.num_layers - 1, 0, -1):
            self.deltas.append(np.dot(self.deltas[-1], self.w[i].T) * self.activation_deriv(self.z[i]))
            self.gradw.append(np.dot(self.a[i - 1].T, self.deltas[-1]))
            self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))
        self.gradw.reverse()
        self.gradb.reverse()
        self.deltas.reverse()

    def update(self, lr: float) -> None:
        for i in range(self.num_layers):
            self.w[i] -= lr*self.gradw[i]
            self.b[i] -= lr*self.gradb[i]

    def train(self, x: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> None:
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)
            self.update(lr)
            if i % 100 == 0:
                print(f'Epoch {i}: {self.mse(x, y)}')
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
        
x, y = generate_data(100)
nn = NN(1, 3, 1, 3)
nn.train(x, y, 0.01, 10000)
y_pred = [np.mean(a) for a in nn.predict(x)]
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()