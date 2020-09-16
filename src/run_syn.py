import numpy as np
import negsep
import dropout
import gpmodel
import bnn
import matplotlib.pyplot as plt
from tabulate import tabulate

np.random.seed(1)

data_name = "synthetic_data_1D"
# data_name = "synthetic_data_2D_square"
# data_name = "synthetic_data_2D_gaussian"
# data_name = "sarcos"
print('Read data' + data_name + '...')

train_data = np.genfromtxt('../data/'+data_name+'_train.csv', delimiter=',')
test_data = np.genfromtxt('../data/'+data_name+'_test.csv', delimiter=',')

Ntr = 2000
xtr, ytr = train_data[:Ntr,:-1], train_data[:Ntr,-1:]
xte, yte = test_data[:,:-1], test_data[:,-1:]


ntr,dx = xtr.shape
dy = ytr.shape[1]


# Model setup
models, results = [],[]
 #models.append(bnn.BayesianNeuralNetwork(dx,dy))
models.append(gpmodel.GPmodel(dx,dy))
models.append(negsep.NegSEp(dx,dy,1,2))
models.append(negsep.NegSEp(dx,dy,10,2))
models.append(dropout.Dropout(dx,dy))

for model in models:
    model_name = model.__class__.__name__
    results.append([model_name])
    print('Processing '+ model_name + ':')
    print('Adding Data...')
    model.add_data(xtr,ytr)
    print('Training...')
    model.train()
    
    # Evaluation
    results[-1].extend(model.weighted_RMSE(xte,yte))
    results[-1].extend([model.compare(xte,models[0])])

    # Visualization
    modelfig = plt.figure(figsize=(10, 5))
    modelfig.suptitle(model.__class__.__name__)
    if dx == 1:
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
        plt.plot(xtr, np.zeros(ntr),'o', color="red")
        try:
            plt.plot(model.x_epi[:, 0], model.y_epi, 'o', color="green")
        except AttributeError:
            pass
        plt.show()

    elif dx == 2:
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
            ax.scatter(model.x_epi[:, 0], model.x_epi[:, 1], model.y_epi, color="green")
        except AttributeError:
            pass
    elif dx == 21:
        modelte, epi = model.predict(xte)
        RMSE = np.sqrt(np.sum((modelte - yte)**2,axis=1)).mean()


print(tabulate(results, headers=['weighted RMSE', 'RMSE', 'discounted RMSE', 'total RMSE','EPI RMSE to ref']))


print('Pau')


