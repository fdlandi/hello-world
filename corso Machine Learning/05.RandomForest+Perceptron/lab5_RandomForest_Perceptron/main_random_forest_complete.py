import numpy as np
import matplotlib.pyplot as plt
from datasets import h_shaped_dataset
from datasets import two_moon_dataset
from datasets import gaussians_dataset
from utils import plot_boundary
from utils import plot_2d_dataset
from sklearn.ensemble import RandomForestClassifier

plt.ion()

def main_svm():
    """
    Main function for testing Adaboost.
    """

    X_train, Y_train, X_test, Y_test = h_shaped_dataset()
    #X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    #X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=300, noise=0.2)

    # visualize dataset
    plot_2d_dataset(X_train, Y_train, 'Training')

    classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
    classifier.fit(X_train, Y_train)
    P = classifier.predict(X_test)

    # visualize the boundary!
    plot_boundary(X_train, Y_train, classifier)

    # evaluate and print error
    error = float(np.sum(P != Y_test)) / Y_test.size
    print('Classification error: {}'.format(error))


# entry point
if __name__ == '__main__':
    main_svm()
