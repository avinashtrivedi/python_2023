import numpy as np

def track_data(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()[1:]]  # Skipping the first line
    return np.array(data)

def perceptron_train(X, y, learning_rate=0.01, epochs=10):
    weights = np.zeros(X.shape[1] + 1)
    X = np.c_[np.ones(X.shape[0]), X]  # Adding bias term
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            update = learning_rate * (target - np.dot(xi, weights))
            weights += update * xi
    return weights

def perceptron_predict(X, weights):
    X = np.c_[np.ones(X.shape[0]), X]  # Adding bias term
    return np.dot(X, weights)

if __name__ == "__main__":
    # Load data from the "track.txt" file
    data = track_data("track.txt")

    # Assuming that the target values are not provided in the file.
    # You can add the target values to the "track.txt" file if you have them.

    # Generate synthetic target values (you can replace this with the actual targets)
    # For a simple linear perceptron, we can assume the target values are in a linear relationship with the input data.
    target_values = 2 * data + 3  # Example linear relationship (y = 2x + 3)

    # Train the perceptron on the data
    weights = perceptron_train(data.reshape(-1, 1), target_values)

    # Make predictions
    predicted_values = perceptron_predict(data.reshape(-1, 1), weights)

    # Print the weights of the trained perceptron
    print("Trained Weights:", weights)

    # Print predicted values
    print("Predicted Values:", predicted_values)


