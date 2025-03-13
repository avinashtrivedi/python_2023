import numpy as np
import matplotlib.pyplot as plt

# 1. Load y-coordinates from the .txt file
y_coordinates = np.loadtxt('track.txt', skiprows=1)
n_samples = len(y_coordinates)

# 2. x-coordinates are the line number
x_coordinates = np.arange(n_samples).reshape(-1, 1)

# 3. Create labels based on the mean of y-coordinates
labels = np.where(y_coordinates > np.mean(y_coordinates), 1, 0)

X = np.column_stack((x_coordinates, y_coordinates))

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output > 0, 1, 0)

    def fit(self, X, y, n_epochs=10):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)
        for epoch in range(n_epochs):
            for idx, x_i in enumerate(X):
                if y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

# 4. Train the perceptron
perceptron = Perceptron(learning_rate=0.1)
perceptron.fit(X, labels, n_epochs=10)

def plot_decision_boundary(X, y, model):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.title("location")
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Line Number')
    plt.ylabel('Value')
    plt.show()

# 5. Visualize the data and the decision boundary
plot_decision_boundary(X, labels, perceptron)


