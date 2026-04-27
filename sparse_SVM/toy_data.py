import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def gen_toy_data(num_class_1_samples, num_class_2_samples, plot=False):
    dimension = 2

    class_1 = np.random.randn(num_class_1_samples, dimension) + np.array([2, 2])
    class_2 = np.random.randn(num_class_2_samples, dimension) + np.array([-2, -2])

    X = np.vstack((class_1, class_2))

    y1 = np.ones((num_class_1_samples, 1))
    y2 = -np.ones((num_class_2_samples, 1))
    y = np.vstack((y1, y2))

    data = np.hstack((X, y))

    if plot:
        plt.scatter(class_1[:, 0], class_1[:, 1])
        plt.scatter(class_2[:, 0], class_2[:, 1])
        plt.title("Toy Data (2D separable)")
        plt.show()

    return data


# demo
gen_toy_data(2, 3, plot=True)