import numpy as np

weather_dataset_train = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
weather_dataset_test = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

np.set_printoptions(precision=10, suppress=True)

class Gaussian:
    def __init__(self, dim):
        self.dim = dim  # dimension de la gaussienne
        self.mu = np.zeros(dim)
        self.sigmasq = None
        self.covariance = None

    # Apprendre les valeurs de mu et sigmasq
    def train(self, data):
        self.mu = np.mean(data, axis=0)
        # self.sigmasq = np.sum((data - self.mu) ** 2.0) / (self.dim * data.shape[0])
        self.covariance = np.cov(data, rowvar=False)
    
    # Calculer la log-vraisemblance
    def log_likelihood(self, data):
        # normalisation_log = self.dim * -(np.log(np.sqrt(self.sigmasq)) + (1 / 2) * np.log(2 * np.pi))
        # return normalisation_log - np.sum((data - self.mu) ** 2.0, axis=1) / (2.0 * self.sigmasq)
        normalisation_log = -np.log(np.sqrt(np.linalg.det(self.covariance))) - (self.dim / 2) * np.log(2 * np.pi)
        return normalisation_log - (np.dot( (data - self.mu), np.linalg.inv(self.covariance)) * (data - self.mu) ).sum(axis=1) / 2

class BayesClassifier:
    def __init__(self, ML_models, priors):
        self.ML_models = ML_models
        self.priors = priors
        self.n_classes = len(ML_models)
    
    def log_likelihood(self, data):
        log_pred = np.zeros((data.shape[0], self.n_classes))
        for i in range(self.n_classes):
            log_pred[:, i] = self.ML_models[i].log_likelihood(data) + np.log(self.priors[i])
        return log_pred

data = weather_dataset_train[1:,1:]
test_data = weather_dataset_test[1:,1:]

# Séparer les exemples par classe
X0 = data[data[:,-1] == 0]
X0 = X0[:,:-1]
X1 = data[data[:,-1] == 1]
X1 = X1[:,:-1]
X2 = data[data[:,-1] == 2]
X2 = X2[:,:-1]

# Créer un modèle par classe et les entraîner
model0 = Gaussian(X0.shape[1])
model0.train(X0)
model1 = Gaussian(X1.shape[1])
model1.train(X1)
model2 = Gaussian(X2.shape[1])
model2.train(X2)

ML_models = [model0, model1, model2]
priors = [X0.shape[0] / data.shape[0], X1.shape[0] / data.shape[0], X2.shape[0] / data.shape[0]]

# Classifieur avec modèles gaussiens et priors
classifier = BayesClassifier(ML_models, priors)

# Calculer l'erreur
log_pred = classifier.log_likelihood(test_data)
pred_vect = np.argmax(log_pred, axis=1)

# err = np.mean(pred != data[:,-1])
# print(f"Erreur sur le train: {err:.4f}")

filename = "bayes.csv"
with open(f"submission/{filename}.csv", 'w') as f: 
    for idx, pred in enumerate(pred_vect, 1):  
        f.write(f"{idx},{pred}\n")
