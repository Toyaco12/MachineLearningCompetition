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

class LogisticRegression():
    #Régression logistique multinomiale avec descente de gradient

    def __init__(self,n_class,n_features,reg):
        self.w = np.random.randn(n_class, n_features) * 0.01
        self.w = self.w.T
        self.reg = reg
    
    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        score = np.dot(X, self.w)
        proba = self.softmax(score)
        return np.argmax(proba, axis=1)
    
    def error_rate(self, X, y):
        return np.mean(self.predict(X) != y)
    
    def loss(self, X, y):
        y = y.astype(int)
        score = np.dot(X, self.w)
        proba = self.softmax(score)
        entropy = -np.log(proba[range(X.shape[0]), y])
        loss = np.sum(entropy)
        return loss + (self.reg * np.linalg.norm(self.w)**2)
    
    def gradient(self, X, y):
        y = y.astype(int)
        score = np.dot(X, self.w)
        proba = self.softmax(score)
        dloss = proba
        dloss[range(X.shape[0]), y] -= 1
        dloss /= X.shape[0]

        grad = np.dot(X.T, dloss)
        grad += self.reg * self.w
        return grad
    
    def train(self,data,L,t0,n_steps):
        
        X = data[:,:-1]
        y = data[:,-1]
        losses = []
        errors = []
        
        eta = L/t0
        for i in tqdm.tqdm(range(n_steps)):
            if i % 100 == 0:
                eta /= 2
            self.w -= eta * self.gradient(X, y)
            losses.append(self.loss(X, y))
            errors.append(self.error_rate(X, y))
        
        print("Entrainement terminé :l'erreur d'entrainement est {:.2f}%".format(errors[-1]*100))
        return np.array(losses), np.array(errors)

n_class = 3
n_features = 20
reg = 0.01
# stepsize = 0.01
L = 1
t0 = 2

def case1(L, t0, n):
    #Train the model with a subset of the data
    model = LogisticRegression(n_class, n_features, reg)
    trainset, testset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=30000)
    training_loss, training_error = model.train(trainset,L,t0, n)

    #Save the model with the test error as name
    test_error = model.error_rate(testset[:,:-1], testset[:,-1])*100
    print("The test error is {:.2f}%".format(test_error))
    filename = f"model/error_{test_error:.2f}.joblib"
    dump(model, filename)

# L = [1]
# t0 = [2,3,4,5,6,7,8]
# n = [1000,2000,3000]
# for n in n:
#     for t0 in t0:
#         print(f"{L}/{t0}+i/100 & {n} ->")
#         case1(L[0], t0, n)
case1(2, 1, 2000)

# print("1 to train the model // 2 to train with all the data // 3 to load a model and predict")
# answer = input()
# match int(answer):
#     case 1:
#         #Train the model with a subset of the data
#         model = LogisticRegression(n_class, n_features, reg)
#         trainset, testset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=30000)
#         training_loss, training_error = model.train(trainset,L,t0, 1000)

#         #Save the model with the test error as name
#         test_error = model.error_rate(testset[:,:-1], testset[:,-1])*100
#         print("The test error is {:.2f}%".format(test_error))
#         filename = f"model/error_{test_error:.2f}.joblib"
#         dump(model, filename)

#     case 2:
#         #Train the model with all the data
#         trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
        
#         model = LogisticRegression(n_class, n_features, reg)
#         training_loss, training_error = model.train(trainset, L,t0, 100)

#         #Save the model with a random name
#         rgn = str(np.random.default_rng().integers(0, 50))+str(np.random.default_rng().integers(0, 50))
#         filename = f"model/alldata_{rgn}.joblib"
#         dump(model, filename)
#     case 3:

#         trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
        
#         #Load a model and predict
#         print("Enter the name of the model to load :")

#         filename = input()
#         model = load(f"model/{filename}")

#         predictions = model.predict(testset)

#         #Save predictions in a file with the format n,prediction
#         with open(f"submission/{filename.rsplit('.',1)[0]}.csv", 'w') as f: 
#             for idx, pred in enumerate(predictions, 1):  
#                 f.write(f"{idx},{pred}\n")

