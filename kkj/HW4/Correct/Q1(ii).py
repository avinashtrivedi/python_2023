import numpy as np
import matplotlib.pyplot as plt
D = np.array([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
     [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
     [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
     [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
     [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
     [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
     [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
     [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
     [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])
cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
lr = 0.01
max_iters = 5000
saved = []
losses = [1000000000,]
np.random.seed(64)


def main():
    x = np.random.rand(9, 2) * 1000
    for iter in range(max_iters):
        for i in range(9):
            g = np.zeros(2,)
            for j in range(9):
                if j != i:
                    g += (np.linalg.norm(x[i]-x[j], ord=2)-D[i, j])*(x[i]-x[j])\
                         / np.linalg.norm(x[i]-x[j], ord=2)
            g *= 4
            x[i] -= g * lr
        loss = 0
        for i in range(9):
            for j in range(9):
                loss += (np.linalg.norm(x[i]-x[j], ord=2)-D[i, j])**2
        losses.append(loss)
        if abs(losses[-1] - losses[-2]) < 0.001:
            print("iter: {}, loss: {}".format(iter, loss))
            break
        if iter % 10 == 0:
            print("iter: {}, loss: {}".format(iter, loss))

    x = np.transpose(x).tolist()

    plt.scatter(x[0], x[1])
    for i in range(len(cities)):
        plt.annotate(cities[i], xy=(x[0][i], x[1][i]), xytext=(x[0][i] + 0.1, x[1][i] + 0.1))
    plt.show()



if __name__ == "__main__":
    main()