import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

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
    
    def train(self,data,data_test,stepsize,n_steps):
        
        X = data[:,:-1]
        y = data[:,-1]
        losses = []
        errors = []
        errors_test = []
        
        for i in range(n_steps):
                            
            losses.append(self.loss(X, y))
            errors.append(self.error_rate(X, y))
            errors_test.append(self.error_rate(data_test[:,:-1], data_test[:,-1]))
            if(i>5 and (errors_test[i-1] - errors_test[i])>0 and (errors_test[i-2] - errors_test[i-1])<0 and (errors_test[i-3] - errors_test[i-2])>0 and (errors_test[i-4] - errors_test[i-3])<0):
               stepsize = stepsize *0.90
            elif(i>5 and  errors_test[i] == errors_test[i-1] and errors_test[i-1] == errors_test[i-2] and errors_test[i-2] == errors_test[i-3] and errors_test[i-3] == errors_test[i-4] and errors_test[i-4] == errors_test[i-5]):
                stepsize = stepsize * 1.1
            print(stepsize)
            self.w -= stepsize * self.gradient(X, y)


        print("Entrainement terminé :l'erreur d'entrainement est {:.2f}%".format(errors[-1]*100))
        print("la plus petite erreur de test est {:.2f}%".format(min(errors_test)*100))

        return np.array(losses), np.array(errors)
    
def find_hyperparameters(trainset):
    reg_test = [0.001,0.0001,0.00001]
    stepsize_test = [1,3,5,7]
    n_step = [100,150,250,500,1000]
    best_error = 1
    best_reg = 0
    best_stepsize = 0
    for n_steps in n_step:
        for reg in reg_test:
            for stepsize in stepsize_test:
                model = LogisticRegression(n_class, n_features, reg)
                training_loss, training_error = model.train(trainset, stepsize, n_steps)
                if training_error[-1] < best_error:
                    best_error = training_error[-1]
                    best_reg = reg
                    best_stepsize = stepsize
                    best_n_steps = n_steps
    return best_reg, best_stepsize, best_n_steps

n_class = 3
n_features = 20
reg = 1e-6
stepsize = 0.1
n_steps = 10000


print(" -1 to train the model \n -2 to train with all the data \n -3 to load a model and predict \n -4 to find the best hyperparameters \n -5 to start training from an existing model \n -6 to start training from an existing model with all the data \n -7 to train in a loop of training")
anwser = input()
match int(anwser):
    case 1:
        #Train the model with a subset of the data
        model = LogisticRegression(n_class, n_features, reg)
        trainset, testset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=35000)
        training_loss, training_error = model.train(trainset,testset,stepsize, n_steps)
        # learning curves
        """ fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,2))
        ax0.plot(test_error)
        ax0.set_title('test_error')
        ax1.plot(training_loss)
        ax1.set_title('loss ( risque empirique )')
        plt.show() """
        #Save the model with the test error as name
        test_error = model.error_rate(testset[:,:-1], testset[:,-1])*100
        print("The test error is {:.2f}%".format(test_error))
        filename = f"model/error_{test_error:.2f}.joblib"
        dump(model, filename)

    case 2:
        #Train the model with all the data
        trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
        
        model = LogisticRegression(n_class, n_features, reg)
        training_loss, training_error = model.train(trainset, stepsize, n_steps)

        #Save the model with a random name
        rgn = str(np.random.default_rng().integers(0, 50))+str(np.random.default_rng().integers(0, 50))
        filename = f"model/alldata_{rgn}.joblib"
        dump(model, filename)
    case 3:

        trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
        
        #Load a model and predict
        print("Enter the name of the model to load :")

        filename = input()
        model = load(f"model/{filename}")

        predictions = model.predict(testset)

        #Save predictions in a file with the format n,prediction
        with open(f"submission/{filename.rsplit('.',1)[0]}.csv", 'w') as f: 
            for idx, pred in enumerate(predictions, 1):  
                f.write(f"{idx},{pred}\n")

    case 4:
        #Find the best hyperparameters
        trainset, testset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=30000)
        reg, stepsize,n_steps = find_hyperparameters(trainset)
        print("The best hyperparameters are reg = {} and stepsize = {} and n_steps = {}".format(reg,stepsize,n_steps))

    case 5:
        #Train the model with a subset of the data from an existing model
        print("Enter the name of the model to load :")
        filename = input()
        model = load(f"model/{filename}")
        trainset, testset = preprocess(weather_dataset_train, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], n_train=30000)
        stepsize = 0.1
        training_loss, training_error = model.train(trainset, stepsize, n_steps)

        #Save the model with the test error as name
        test_error = model.error_rate(testset[:,:-1], testset[:,-1])*100
        print("The test error is {:.2f}%".format(test_error))
        filename = f"model/error_{test_error:.2f}.joblib"
        dump(model, filename)

    case 6:
        #Train the model with all the data from an existing model
        print("Enter the name of the model to load :")
        filename = input()
        model = load(f"model/{filename}")
        trainset, testset = preprocess_V2(weather_dataset_train,weather_dataset_test, label_subset=[0,1,2], feature_subset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
        stepsize = 0.1
        training_loss, training_error = model.train(trainset, stepsize, n_steps)
        #Save the model with a random name
        rgn = str(np.random.default_rng().integers(0, 50))+str(np.random.default_rng().integers(0, 50))
        print(rgn)
        filename = f"model/alldata_{rgn}.joblib"
        dump(model, filename)
