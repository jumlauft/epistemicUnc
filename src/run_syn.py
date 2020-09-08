import numpy as np
from src import negsep
from src import dropout
from src import gpmodel
from src import bnn
import matplotlib.pyplot as plt

print('Read data...')
name = "synthetic_data_1D"
name = "synthetic_data_2D_square"
# name = "synthetic_data_2D_gaussian"
name = "sarcos"

train_data = np.genfromtxt('data/'+name+'_train.csv', delimiter=',')
test_data = np.genfromtxt('data/'+name+'_test.csv', delimiter=',')

xtr, ytr = train_data[:,:-1], train_data[:,-1:]
xte, yte = test_data[:,:-1], test_data[:,-1:]


ntr,dx = xtr.shape
dy = ytr.shape[1]


# Model setup
models = []
# models.append(bnn.BayesianNeuralNetwork(dx,dy))
# models.append(gpmodel.GPmodel(dx,dy))
models.append(negsep.NegSEp(dx,dy,[-0.5, -0.5],[1.5, 1.5]))
# models.append(dropout.Dropout(dx,dy))

for model in models:
    model_name = model.__class__.__name__
    print('Processing:'+ model_name+'....')
    model.add_data(xtr,ytr)
    model.train()
    # Evaluation

    # Visualization
    modelfig = plt.figure(figsize=(10, 5))
    modelfig.suptitle(model.__class__.__name__)
    if dx == 1:
        modelte, epi = model.predict(xte)
        ax = modelfig.add_subplot(121)
        ax.set_title('Model Prediction')
        ax.plot(xtr, ytr, 'o', color="blue", label='training')
        ax.plot(xte, modelte, color="red", label='model')
        ax.legend()

        ax = modelfig.add_subplot(122)
        ax.set_title('Epistemic Uncertainty')
        plt.plot(xte, epi, color="blue")
        plt.plot(xtr, np.zeros(ntr), color="red")
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






print('Pau')


