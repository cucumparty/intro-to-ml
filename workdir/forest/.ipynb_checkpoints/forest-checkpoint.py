import numpy as np
import random
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier


class RandomForest(ABC):
    """
    Base class for the random forest algorithm
    """
    def __init__(self, n_trees=100):
        """
        Constructor

        :param n_trees: Count of trees in ensemble
        """
        self.n_trees = n_trees
        self.trees = []

    def __make_bootstraps(self, data):
        """
        Private function that generates bootstrap samples for given data

        :param data: np.array of type float with shape (n_datapoints, n_features)
                     Dataset
        :return: dict
                 Dictionary with n_samples bootstrap samples
        """

        bootstrap = {}
        sample_size = data.shape[0]
        n_samples = 1000

        for sample in range(n_samples):
            bootstrap_sample_indices = np.random.choice(sample_size, size=sample_size, replace=True)
            bootstrap_sample_data = data[bootstrap_sample_indices]
        
            # Split bootstrap sample into train and test
            train_sample, test_sample = train_test_split(bootstrap_sample_data, test_size=0.2)  # Modify test_size as needed
        
            bootstrap["tree_" + str(sample)] = {'train': train_sample, 'test': test_sample}

        return bootstrap


    def get_params(self, deep=False):
        """
        Public function to return model parameters

        :param deep: bool that may be used in the descendant class
        :return: dict
                 Returns model parameters
        """
        return {'n_trees': self.n_trees}

    @abstractmethod
    def _make_tree_model(self):
        """
        Protected function to obtain the right decision tree

        :return:
        """
        pass

    def _train(self, X_train, y_train):
        """
        Protected function to train the ensemble

        Algorithm:
        - make bootstrap samples
        - iterate through each bootstrap sample & fit a model
             - make a clone of the model
             - fit a decision tree model to the current sample
             - append the fitted model
             - store the out-of-bag test set for the current bootstrap
        - return the oob data set

        :param X_train: np.array-like - train part of dataset without labels
        :param y_train: np.array-like - train labels
        :return: dict
                 Out-of-bag test datasets
        """

        training_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        bootstraps = self.__make_bootstraps(training_data)
        tree = self._make_tree_model()
        out_of_bag = {}
        models = []

        for sample in bootstraps:
            model = clone(tree)
            train_data = bootstraps[sample]['train']
            models.append(model.fit(train_data[:, :-1], train_data[:, -1]))
            
            if bootstraps[sample]['test'].size:
                out_of_bag[sample] = bootstraps[sample]['test']
            else:
                out_of_bag[sample] = np.array([])
                
        self.trees = models
        
        return out_of_bag

    def _predict(self, X):
        """
        Protected function to predict from the ensemble

        Algorithm:
        - check we've fit the ensemble
        - loop through each fitted model
             - make predictions on the input X
             - append predictions to storage list
        - compute the ensemble prediction


        :param X: np.array-like - data to make predict
        :return:  np.array-like - predicted answers
                  Predicted answers
        """

        if not self.trees:
            print('You must train the ensemble before making predictions!')
            return None

        predictions = []
        for model in self.trees:
            predictions.append(model.predict(X))

        y_pred = np.mean(predictions, axis=0) 
        return y_pred


class RandomForestClassifier(RandomForest):
    """
    Class for random forest classifier
    """

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, loss='gini', balance_class_weights=False):
        """
        Constructor

        :param n_trees: int - count of trees
        :param max_depth: int - max depth of each tree
        :param min_samples_split: int or float - the minimum number of samples required to split an internal node
        :param loss: {“gini”, “entropy”, “log_loss”} - the function to measure the quality of a split
        :param balance_class_weights: bool - if True uses the values of y to automatically adjust weights inversely
                                      proportional to class frequencies in the input data as
                                      n_samples / (n_classes * np.bincount(y))
        """
        super().__init__(n_trees)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.balance_class_weights = balance_class_weights

    def _make_tree_model(self):
        """
        Protected function to obtain the right decision tree

        :return: sklearn.tree.DecisionTreeClassifier
                 Returns decision tree
        """
        return (DecisionTreeClassifier(max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       criterion=self.loss,
                                       class_weight=("balanced" if self.balance_class_weights else None)))

    def get_params(self, deep=False):
        """
        Public function to return model parameters

        :param deep: bool that may be used in the descendant class
        :return: dict
                 Returns model parameters
        """
        return {'n_trees': self.n_trees,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'loss': self.loss,
                'balance_class_weights': self.balance_class_weights}

    def fit(self, X_train, y_train, print_metrics=False):
        """
        Train the ensemble

        Algorithm:
        - call the protected training method
        - if print_metrics selected, compute the standard errors and print them
            - initialise metric arrays
            - loop through each bootstrap sample
            - compute the predictions on the out-of-bag test set & compute metrics
            - store the error metrics
            - print standard errors

        :param X_train: np.array-like - train part of dataset without labels
        :param y_train: np.array-like - train labels
        :param print_metrics: bool - if True prints standard errors
        :return:
        """

        # TODO: Write one string of code below
        out_of_bag = super()._train(X_train, y_train)

        if print_metrics:
            accuracies = np.array([])
            precisions = np.array([])
            recalls = np.array([])
            for test_sample, tree in zip(out_of_bag, self.trees):
                if out_of_bag[test_sample].size:
                    y_pred = tree.predict(out_of_bag[test_sample][:, :-1])
                    accuracy = accuracy_score(out_of_bag[test_sample][:, -1], y_pred)
                    precision = precision_score(out_of_bag[test_sample][:, -1], y_pred, average='weighted')
                    recall = recall_score(out_of_bag[test_sample][:, -1], y_pred, average='weighted')

                    accuracies = np.concatenate((accuracies, accuracy.flatten()))
                    precisions = np.concatenate((precisions, precision.flatten()))
                    recalls = np.concatenate((recalls, recall.flatten()))

            print("Standard error in accuracy: %.2f" % np.std(accuracies))
            print("Standard error in precision: %.2f" % np.std(precisions))
            print("Standard error in recall: %.2f" % np.std(recalls))
            
    def predict(self, X):
        """
        Predict from the ensemble

        Algorithm:
        - call the protected prediction method
        - convert the results into integer values & return

        :param X: np.array-like - data to make predict
        :return: int
                 Predicted class
        """

        # TODO: YOUR CODE HERE
        res = super()._predict(X)
        return(np.round(res))
        