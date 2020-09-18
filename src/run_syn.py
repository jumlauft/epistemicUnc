import numpy as np
import negsep
import dropout
import gpmodel
import bnn
import matplotlib.pyplot as plt
from tabulate import tabulate
from time import time
np.random.seed(1)


# data_name = "synthetic_data_1D"
# data_name = "synthetic_data_1D_split"
# data_name = "synthetic_data_2D_square"
# data_name = "synthetic_data_2D_gaussian"
# data_name = "sarcos"
data_name = "wine_quality"
data_name = "motor_temperature"

print('Read data' + data_name + '...')

train_data = np.genfromtxt('../data/'+data_name+'_train.csv', delimiter=',')
test_data = np.genfromtxt('../data/'+data_name+'_test.csv', delimiter=',')

# Ntr = 5000
# xtr, ytr = train_data[:Ntr,:-1], train_data[:Ntr,-1:]
xtr, ytr = train_data[:,:-1], train_data[:,-1:]

ntr,dx = xtr.shape
dy = ytr.shape[1]

# nte = test_data.shape[0]
# idx = np.random.choice(ntr,min(ntr,nte), replace=False)
# xte = np.concatenate((train_data[idx,:-1],test_data[:,:-1]), axis=0)
# yte = np.concatenate((train_data[idx,-1:],test_data[:,-1:]), axis=0)
xte, yte = test_data[:,:-1], test_data[:,-1:]

nte = xte.shape[0]



print('Read ' + str(ntr) + ' datapoints with ' + str(dx) + ' dimensions')

# Model setup
models, results = [],[]
 #models.append(bnn.BayesianNeuralNetwork(dx,dy))
# models.append(gpmodel.GPmodel(dx,dy))
models.append(negsep.NegSEp(dx,dy,1,2))
models.append(dropout.Dropout(dx,dy))

for model in models:
    model_name = model.__class__.__name__
    results.append([model_name])
    print('Processing '+ model_name + ':')
    print('Adding Data...')
    t0= time()
    model.add_data(xtr,ytr)
    print('Training...')
    model.train()
    ttrain = time() - t0
    # Evaluation
    print('Evaluating on ' + str(nte) + ' data points...')
    t0 = time()
    result = model.weighted_RMSE(xte,yte)
    tevaluate = time() - t0
    
    results[-1] = result
    results[-1]['ttrain'] = ttrain
    results[-1]['tevaluate'] = tevaluate


    # Visualization
    if dx == 1:
        modelfig = plt.figure(figsize=(10, 5))
        modelfig.suptitle(model.__class__.__name__)
        modelte, epi = model.predict(xte)
        ax = modelfig.add_subplot(121)
        ax.set_title('Model Prediction')
        ax.plot(xtr, ytr, 'o', color="blue", label='training')
        ax.plot(xte, modelte, color="red", label='model')
        ax.plot(xte, yte, color="orange", label='test')
        ax.legend()

        ax = modelfig.add_subplot(122)
        ax.set_title('Epistemic Uncertainty')
        plt.plot(xte, epi, color="blue")
        plt.plot(xtr, np.zeros(ntr),'x', color="red")
        try:
            plt.plot(model.get_x_epi()[:, 0], model.get_y_epi(), 'o', color="green")
        except AttributeError:
            pass
        plt.show()

    elif dx == 2:
        modelfig = plt.figure(figsize=(10, 5))
        modelfig.suptitle(model.__class__.__name__)
        modelte, epi = model.predict(xte)
        ax = modelfig.add_subplot(121, projection='3d')
        ax.set_title('Model Prediction')
        ax.scatter(xtr[:, 0], xtr[:, 1], ytr, color="blue", label='training')
        ax.scatter(xte[:, 0], xte[:, 1], modelte, color="red", label='model')
        ax.scatter(xte[:, 0], xte[:, 1], yte[:,0], color="orange", label='test')
        ax.legend()

        ax = modelfig.add_subplot(122, projection='3d')
        ax.set_title('Epistemic Uncertainty')
        ax.scatter(xte[:, 0], xte[:, 1], epi, color="blue")
        ax.scatter(xtr[:, 0], xtr[:, 1], np.zeros(ntr), color="red")
        try:
            ax.scatter(model.get_x_epi()[:, 0], model.get_x_epi()[:, 1], model.y_epi, color="green")
        except AttributeError:
            pass


tab = tabulate([a.values() for a in results], headers=results[0].keys())
print(tab)
with open(data_name + '_results.txt', 'w') as f:
    print(tab, file=f) 

print('Pau')


