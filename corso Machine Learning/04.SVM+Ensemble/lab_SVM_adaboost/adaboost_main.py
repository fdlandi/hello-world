import numpy as np
import matplotlib.pyplot as plt
from datasets import h_shaped_dataset
from datasets import two_moon_dataset
from datasets import gaussians_dataset
from utils import plot_boundary
from utils import cmap
from utils import plot_2d_dataset

plt.ion()


class AdaBoostClassifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of weak classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.alphas = np.zeros(shape=n_learners)
        self.dims = np.zeros(shape=n_learners, dtype=np.int32)
        self.splits = np.zeros(shape=n_learners)
        self.label_above_split = np.zeros(shape=n_learners, dtype=np.int32)

        self.possible_labels = None

    def fit(self, X, Y, verbose=False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        # some inits
        n, d = X.shape
        if d != 2:
            verbose = False  # only plot learning if 2 dimensional

        self.possible_labels = np.unique(Y)

        # only binary problems please
        assert self.possible_labels.size == 2, 'Error: data is not binary'

        # initialize the sample weights as equally probable
        sample_weights = np.ones(shape=n) / n

        # start training
        for l in range(0, self.n_learners):

            # choose the indexes of 'difficult' samples (np.random.choice)
            cur_idx = np.random.choice(a=range(0, n), size=n, replace=True, p=sample_weights)  #scegliamo n indici random da 0 a n, con ripetizioni, usando sample_weights come pesi

            # extract 'difficult' samples
            cur_X = X[cur_idx]  #estraiamo gli elementi dal dataset con indici calcolati prima
            cur_Y = Y[cur_idx]  #prendiamo le loro label

            # search for a weak classifier
            error = 1
            n_trials = 0
            while error > 0.5:

                # select random feature (np.random.choice)
                cur_dim = np.random.choice(a=range(0, d))  #ogni decision stump considera una feature a caso e assegna una classe a seconda che il valore della feature è sopra o sotto una certa soglia

                # select random split (np.random.uniform) --> scegliamo una soglia a caso tra il minimo e il massimo valore che assume la feature cur_dm
                M, m = np.max(cur_X[:, cur_dim]), np.min(cur_X[:, cur_dim])
                cur_split = np.random.uniform(low=m, high=M)  # SOGLIA=valore random tra min e max

                # select random verse (np.random.choice)
                label_above_split = np.random.choice(a=self.possible_labels)  #stabilisco quale classe assegnare agli elementi che hanno valore della feature cur_dim sopra soglia
                label_below_split = -label_above_split  # e quale label assegnare agli elementi con valore sotto soglia

                # compute assignment
                cur_assignment = np.zeros(shape=n)
                cur_assignment[cur_X[:, cur_dim] >= cur_split] = label_above_split  #assegno le label sopra soglia agli elementi con feature cur_dim sopra soglia
                cur_assignment[cur_X[:, cur_dim] < cur_split] = label_below_split  #assegno le label sotto soglia agli elementi con feature cur_dim sotto soglia

                # compute error
                error = np.sum(sample_weights[cur_idx[cur_Y != cur_assignment]])  #sommo i pesi degli elementi classificati male

                n_trials += 1
                if n_trials > 100:  # se dopo 100 volte non ho raggiunto errore minore di 0.5, inizializzo da capo i pesi degli elementi al valore originale
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n

            # save weak learner parameter
            alpha = np.log((1 - error) / error) / 2
            self.alphas[l] = alpha
            self.dims[l] = cur_dim
            self.splits[l] = cur_split
            self.label_above_split[l] = label_above_split

            # update sample weights
            sample_weights[cur_idx[cur_Y != cur_assignment]] *= np.exp(alpha)  #aggiorniamo i pesi degli elementi classificati male con e(alpha)
            sample_weights[cur_idx[cur_Y == cur_assignment]] *= np.exp(-alpha)  #aggiorniamo i pesi degli elementi classificati bene con e(-alpha)
            sample_weights /= np.sum(sample_weights)  #normalizziamo i pesi rispetto alla loro somma (devono sommare a 1)

            if verbose:
                # plot
                plt.clf()
                plt.scatter(cur_X[:, 0], cur_X[:, 1], c=cur_assignment, s=sample_weights[cur_idx] * 50000,
                            cmap=cmap, edgecolors='k')
                M1, m1 = np.max(X[:, 1]), np.min(X[:, 1])
                M0, m0 = np.max(X[:, 0]), np.min(X[:, 0])
                if cur_dim == 0:
                    plt.plot([cur_split, cur_split], [m1, M1], 'k-', lw=5)
                else:
                    plt.plot([m0, M0], [cur_split, cur_split], 'k-', lw=5)
                plt.xlim([m0, M0])
                plt.ylim([m1, M1])
                plt.xticks([])
                plt.yticks([])
                plt.title('Iteration: {:04d}'.format(l))
                plt.waitforbuttonpress(timeout=0.1)

    def predict(self, X):
        """
        Function to perform predictions over a set of samples.

        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """
        n, d = X.shape

        pred_all_learners = np.zeros(shape=(n, self.n_learners))

        for l, cur_dim, cur_split, label_above_split in zip(range(0, self.n_learners), self.dims, self.splits, self.label_above_split):

            label_below_split = -label_above_split  #abbiamo salvato la label assegnata agli elementi sopra soglia in linea 109: ricaviamo la label assegnata agli elementi sotto soglia

            # compute assignment
            cur_assignment = np.zeros(shape=n)
            cur_assignment[X[:, cur_dim] >= cur_split] = label_above_split  #assegniamo label sopra soglia agli elementi con cur_dim (salvata in linea 107) sopra soglia
            cur_assignment[X[:, cur_dim] < cur_split] = label_below_split  #assegniamo label sotto soglia agli elementi con cur_dim (salvata in linea 107) sotto soglia

            pred_all_learners[:, l] = cur_assignment

        # weight for learners efficiency
        pred_all_learners *= self.alphas  #moltiplichiamo per le affidabilità

        # compute predictions
        pred = np.sign(np.sum(pred_all_learners, axis=1))  #prendiamo il segno della somma pesata (se + vince casse 1, se - vince classe -1)

        return pred


def main_adaboost():
    """
    Main function for testing Adaboost.
    """

    #X_train, Y_train, X_test, Y_test = h_shaped_dataset()
    #X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=300, noise=0.2)

    # visualize dataset
    plot_2d_dataset(X_train, Y_train, 'Training')

    # train model and predict
    model = AdaBoostClassifier(n_learners=100)
    model.fit(X_train, Y_train, verbose=True)
    P = model.predict(X_test)

    # visualize the boundary!
    plot_boundary(X_train, Y_train, model)

    # evaluate and print error
    error = float(np.sum(P != Y_test)) / Y_test.size
    print('Classification error: {}'.format(error))


# entry point
if __name__ == '__main__':
    main_adaboost()
