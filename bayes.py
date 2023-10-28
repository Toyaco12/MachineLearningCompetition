import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import tqdm

weather_dataset_train = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
weather_dataset_test = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

np.set_printoptions(precision=10, suppress=True)

def preprocess(data, label_subset, feature_subset, n_train):

    """Effectue une partition aléatoire des données en sous-ensembles
    train set et test set avec le sous-ensemble de classes label_subset
    et le sous-ensemble de feature feature_subset"""
    
    # on extrait seulement les classes de label_subset
    data = data[np.isin(data[:,-1],label_subset),:]

    # on extrait les features et leurs étiquettes
    data = data[:, feature_subset + [-1]]

    # on ajoute une colonne pour le biais
    data = np.insert(data, -1, 1, axis=1)

    # on sépare en train et test
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    train_inds = inds[:n_train]
    test_inds = inds[n_train:]
    trainset = data[train_inds]
    testset = data[test_inds]

    # on normalise les données pour qu'elles soient de moyenne 0
    # et d'écart-type 1 par caractéristique et on applique
    # ces mêmes transformations au test set 
    mu = trainset[:,:-2].mean(axis=0)
    sigma  = trainset[:,:-2].std(axis=0)
    trainset[:,:-2] = (trainset[:,:-2] -mu)/sigma
    testset[:,:-2] = (testset[:,:-2] -mu)/sigma

    return trainset, testset

def preprocess_V2(data_train,data_test, label_subset, feature_subset):

    # on extrait seulement les classes de label_subset
    data_train = data_train[np.isin(data_train[:,-1],label_subset),:]

    # on extrait les features et leurs étiquettes
    data_train = data_train[:, feature_subset + [-1]]
    data_test = data_test[:, feature_subset]
    # on ajoute une colonne pour le biais

    trainset = data_train
    testset = data_test

    # on normalise les données pour qu'elles soient de moyenne 0
    # et d'écart-type 1 par caractéristique et on applique
    # ces mêmes transformations au test set 
    mu = trainset[:,:-1].mean(axis=0)
    sigma  = trainset[:,:-1].std(axis=0)
    trainset[:,:-1] = (trainset[:,:-1] -mu)/sigma
    testset[:,:] = (testset[:,:] -mu)/sigma

    trainset = np.insert(data_train, -1, 1, axis=1)
    testset = np.insert(data_test,testset.shape[1], 1, axis=1)

    return trainset, testset


class NaiveBayes:
"""
    Utiliser le classifieur de Bayes Naïf pour classifier les données
    en utilisant la loi normale pour estimer les densités univariés.
"""
    def __init__(self, sigma):
        self.sigma = sigma
        self.prior = []

    def train():
        """
            Estimate the density of points for each class.
            Estimate the prior probability of each class.
        """
        pass

