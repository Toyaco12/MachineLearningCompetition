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
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # prevent overflow
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
        return loss + (self.reg * np.linalg.norm(self.w)**2)  # régularisation
    
    def gradient(self, X, y):
        y = y.astype(int)
        score = np.dot(X, self.w)
        proba = self.softmax(score)
        dloss = proba
        dloss[range(X.shape[0]), y] -= 1
        dloss /= X.shape[0]

        grad = np.dot(X.T, dloss)
        grad += self.reg * self.w  # régularisation = dérivée de reg * la norme au carré
        return grad
    
    def train(self,data,stepsize,n_steps, valdata):
        X = data[:,:-1]
        y = data[:,-1]
        losses = []
        errors = []
        error_val = []
        stock_w = []
        
        for i in tqdm.tqdm(range(n_steps)):
            if i % 100 == 0:
                stepsize /= 2
            grad = self.gradient(X, y)
            self.w -= stepsize * grad
            stock_w.append(self.w)  #
            losses.append(self.loss(X, y))
            errors.append(self.error_rate(X, y))
            error_val.append(self.error_rate(valdata[:,:-1], valdata[:,-1]))
        
        # print("Entrainement terminé :l'erreur d'entrainement est {:.2f}%".format(errors[-1]*100))
        # print("La plus petite erreur de validation est {:.2f}%".format(min(error_val)*100), "à l'itération", error_val.index(min(error_val)), "avec les poids", stock_w[error_val.index(min(error_val))])

        return np.array(losses), np.array(errors), np.array(error_val), stock_w[error_val.index(min(error_val))]

# Fonction pour entraîner le modèle avec un sous-ensemble des données.
def train_part(n_class, n_features, reg, stepsize, n_step):
    model = LogisticRegression(n_class, n_features, reg)
    trainset, valset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=35000)

    # Entraînement
    training_loss, training_error, val_error, best_w = model.train(trainset, stepsize, n_step, valset)

    test_error = model.error_rate(valset[:,:-1], valset[:,-1])*100
    print("The test error is {:.2f}%".format(test_error))

    best_test_error = min(val_error)
    print("The best test error is {:.2f}%".format(best_test_error*100), "at iteration", np.where(val_error == best_test_error)[0][0])
          
    return training_loss, training_error, val_error, best_w

# Fontion pour entraîner le modèle avec toutes les données.
def train_all():
    pass

# Fonction pour charger un modèle et faire des prédictions.
def test(filename):
    trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    
    #Load a model and predict
    model = load(f"model/{filename}")

    predictions = model.predict(testset)

    #Save predictions in a file with the format n,prediction
    with open(f"submission/{filename.rsplit('.',1)[0]}.csv", 'w') as f: 
        for idx, pred in enumerate(predictions, 1):  
            f.write(f"{idx},{pred}\n")
            exit()

def val_exist(filename):
    #Train the model with a subset of the data from an existing model
    model = load(f"model/{filename}")
    trainset, testset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=1000)
    # stepsize = 0.1
    # training_loss, training_error = model.train(trainset, stepsize, n_steps)

    #Save the model with the test error as name
    test_error = model.error_rate(testset[:,:-1], testset[:,-1])*100
    print("The test error is {:.2f}%".format(test_error))
    # filename = f"model/error_{test_error:.2f}.joblib"
    # dump(model, filename)

res = train_part(3, 20, 0.0001, 5, 500)
# val_exist("error_15.81.joblib")
# test("error_15.81.joblib")


### VAL OPT
# n_step = 500
# n_class = 3
# n_features = 20
# step_size = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100]
# reg = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]
# Pour chaque combinaison d'hyperparamètres, je stocke la meilleure erreur de validation trouvée ainsi que les poids associés.
# Parmi ces combinaisons, je choisis celle qui a la plus petite erreur de validation.
# n_it = len(step_size) * len(reg) * 20
# iteration = 0

# best_error = 100
# best_error_w = np.zeros((n_features, n_class))  # w.T
# best_error_step = 0
# best_error_reg = 0

# for i in range(len(step_size)):
#     for j in range(len(reg)):
#         # Répété pour ces hyperparamètres car l'entraînement est stochastique.
#         for k in range(20):
#             iteration += 1
#             print("—————————— ITERATION ", iteration, "/", n_it, "——————————")
#             print("---------- ETA ", step_size[i], "----------")
#             print("---------- LAMBDA ", reg[j], "----------")
#             print("---------- REPETITION ", k+1, "----------")
#             model = LogisticRegression(n_class, n_features, reg[j])
#             trainset, valset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=35000)

#             # Entraînement
#             training_loss, training_error, val_error, best_w = model.train(trainset, step_size[i], n_step, valset)

#             test_error = min(val_error)
#             if test_error < best_error:
#                 best_error = test_error
#                 best_error_w = best_w
#                 best_error_step = step_size[i]
#                 best_error_reg = reg[j]

#                 # Sauvegarder
#                 filename = f"model/error_{best_error*100:.2f}.joblib"
#                 dump(model, filename)

# print("The best validation error is {:.4f}%".format(best_error*100))
# print("with param w\n", best_error_w)
# print("with stepsize", best_error_step)
# print("with reg", best_error_reg)



# print("1 to train the model // 2 to train with all the data // 3 to load a model and predict")
# anwser = input()
# match int(anwser):
#     case 1:
#         #Train the model with a subset of the data
#         model = LogisticRegression(n_class, n_features, reg)
#         trainset, valset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=35000)
#         training_loss, training_error = model.train(trainset, stepsize, 300, valset)

#         #Save the model with the test error as name
#         test_error = model.error_rate(valset[:,:-1], valset[:,-1])*100
#         print("The test error is {:.2f}%".format(test_error))
#         filename = f"model/error_{test_error:.2f}.joblib"
#         dump(model, filename)

#     case 2:
#         #Train the model with all the data
#         trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
        
#         model = LogisticRegression(n_class, n_features, reg)
#         training_loss, training_error = model.train(trainset, stepsize, 100)

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

