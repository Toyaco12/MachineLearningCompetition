import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from  tqdm import tqdm

weather_dataset_train = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
weather_dataset_test = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

np.set_printoptions(precision=10, suppress=True)

# Pour faire un prétraitement des données du fichier train.csv
def preprocess(data, label_subset, feature_subset, n_train):

    """Effectue une partition aléatoire des données en sous-ensembles
    train set et val set avec le sous-ensemble de classes label_subset
    et le sous-ensemble de feature feature_subset"""
    
    # on extrait seulement les classes de label_subset
    data = data[np.isin(data[:,-1],label_subset),:]

    # on extrait les features et leurs étiquettes
    data = data[:, feature_subset + [-1]]

    # on ajoute une colonne pour le biais
    data = np.insert(data, -1, 1, axis=1)

    # on sépare en train et val
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    train_inds = inds[:n_train]
    val_inds = inds[n_train:]
    trainset = data[train_inds]
    valset = data[val_inds]

    # on normalise les données pour qu'elles soient de moyenne 0
    # et d'écart-type 1 par caractéristique et on applique
    # ces mêmes transformations au val set 
    mu = trainset[:,:-2].mean(axis=0)
    sigma  = trainset[:,:-2].std(axis=0)
    trainset[:,:-2] = (trainset[:,:-2] -mu)/sigma
    valset[:,:-2] = (valset[:,:-2] -mu)/sigma

    return trainset, valset

# Pour faire un prétraitement des données du fichier train.csv et test.csv(sans etiquettes)
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

    def __init__(self, n_class, n_features, reg):
        limit = np.sqrt(6 / (n_features + n_class))
        self.w = np.random.uniform(-limit, limit, (n_features, n_class))
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
        # L1 regularization term, sum of absolute values of weights
        return loss + self.reg * np.sum(np.abs(self.w))
    
    def gradient(self, X, y):
        y = y.astype(int)
        score = np.dot(X, self.w)
        proba = self.softmax(score)
        dloss = proba
        dloss[range(X.shape[0]), y] -= 1
        dloss /= X.shape[0]

        grad = np.dot(X.T, dloss)
        # L1 gradient is the sign of weights
        grad += self.reg * np.sign(self.w)
        return grad
    
    def train(self,data,stepsize,n_steps,decay_rate):
        
        X = data[:,:-1]
        y = data[:,-1]
        losses = []
        errors = []
        initial_stepsize = stepsize
        for i in tqdm(range(n_steps)):
            stepsize = initial_stepsize / (1 + decay_rate * i)
            self.w -= stepsize * self.gradient(X, y)
            losses.append(self.loss(X, y))
            errors.append(self.error_rate(X, y))
        
        print("Entrainement terminé :l'erreur d'entrainement est {:.2f}%".format(errors[-1]*100))
        return np.array(losses), np.array(errors)
    
    def train_with_early_stopping(self, train_data, val_data, stepsize, n_steps, early_stopping_rounds, decay_rate):
        X_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        X_val = val_data[:,:-1]
        y_val = val_data[:,-1]
        
        best_w = None
        best_error = float("inf")
        rounds_without_improvement = 0

        losses = []
        errors = []
        initial_stepsize = stepsize
        for i in tqdm(range(n_steps)):
            stepsize = initial_stepsize / (1 + decay_rate * i)
            self.w -= stepsize * self.gradient(X_train, y_train)
            losses.append(self.loss(X_train, y_train))
            train_error = self.error_rate(X_train, y_train)
            errors.append(train_error)

            val_error = self.error_rate(X_val, y_val)
            if val_error < best_error:
                best_error = val_error
                best_w = self.w.copy()
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement >= early_stopping_rounds:
                print(f"Early stopping after {i} iterations!")
                self.w = best_w
                break

        print(f"Training error after {i+1} iterations: {train_error*100:.2f}%")
        print(f"Validation error after {i+1} iterations: {best_error*100:.2f}%")

        return np.array(losses), np.array(errors)
    
n_class = 3
n_features = 10

def create_trained_model(n_steps,stepsize,reg,decay_rate,mode = "train"):
    errors_val = None
    if mode == "train":
        trainset, valset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[3,4,5,8,9,13,14,15,17], n_train=37500)
        model = LogisticRegression(n_class,n_features,reg)
        losses, errors = model.train(trainset,stepsize,n_steps,decay_rate)
        errors_val = model.error_rate(valset[:,:-1],valset[:,-1])*100
        print("L'erreur de val est {:.2f}%".format(errors_val))
    elif mode == "test":
        trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[3,4,5,8,9,13,14,15,17])
        model = LogisticRegression(n_class,n_features,reg)
        losses, errors = model.train(trainset,stepsize,n_steps,decay_rate)
        predictions = model.predict(testset)
        with open(f"submission/submit.csv", 'w') as f: 
            for idx, pred in enumerate(predictions, 1):  
                f.write(f"{idx},{pred}\n")

    return model,errors_val

def create_trained_modelV2(n_steps,stepsize,reg,decay_rate,early_stopping_rounds=200):

    trainset, valset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[3,4,5,8,9,13,14,15,17], n_train=37500)
    model = LogisticRegression(n_class, n_features, reg)
    losses, errors = model.train_with_early_stopping(trainset, valset, stepsize, n_steps, early_stopping_rounds, decay_rate)
    errors_val = model.error_rate(valset[:,:-1],valset[:,-1])*100
        # learning curves
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,2))
    ax0.plot(losses)
    ax0.set_title('loss')
    ax1.plot(errors)
    ax1.set_title('error rate')
    plt.show()

    return model,errors_val

def find_best_hypermaters(n_steps):
    best_reg = None
    best_stepsize = None
    best_decay_rate = None
    best_model = None
    best_error = 100
    best_early_stopping_rounds = None

    for reg in [0.01,0.001,0.0001]:
        for stepsize in [0.5,1,1.5]:
            for decay_rate in [0.001,0.01]:
                for early_stopping_rounds in [200,300,400]:
                    model, error = create_trained_modelV2(n_steps,stepsize,reg,decay_rate)
                    if error < best_error:
                        best_error = error
                        best_reg = reg
                        best_stepsize = stepsize
                        best_decay_rate = decay_rate
                        best_model = model
                        best_early_stopping_rounds = early_stopping_rounds
    print("Les meilleurs hyperparamètres sont : reg = {}, stepsize = {}, decay_rate = {}, early_stop = {}, pour une erreur de : {}".format(best_reg,best_stepsize,best_decay_rate,best_early_stopping_rounds,best_error))
    return best_model,best_reg,best_stepsize,best_decay_rate,best_early_stopping_rounds

# mymodel,myreg, mystepsize, mydecay_rate,my_early_stopping_rounds = find_best_hypermaters(2000)
# trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[3,4,5,8,9,13,14,15,17])
# predictions = mymodel.predict(testset)
# with open(f"submission/submit.csv", 'w') as f: 
#     for idx, pred in enumerate(predictions, 1):  
#          f.write(f"{idx},{pred}\n")

#mymodel = create_trained_model(1000,1.5,0.00001,0.001,mode = "test")
mymodel = create_trained_modelV2(10000,1.5,0.01,0.001,300)
#mymodel = create_trained_modelV2(10000)
