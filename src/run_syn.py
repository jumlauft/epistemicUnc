import numpy as np
import negsep
import dropout
import gpmodel
# import bnn
from tabulate import tabulate
from time import time
from utils import visualize

np.random.seed(1)

# Add Datasets
data_sets = []
data_sets.append("synthetic_data_1D")
data_sets.append("synthetic_data_1D_split")
# data_sets.append("synthetic_data_2D_square")
# data_sets.append("synthetic_data_2D_gaussian")
# data_sets.append("sarcos")
# data_sets.append("wine_quality")
# data_sets.append("motor_temperature")

for data_name in data_sets:

    # Read data
    print('Read data: ' + data_name + '...')
    train_data = np.genfromtxt('../data/'+data_name+'_train.csv', delimiter=',')
    test_data = np.genfromtxt('../data/'+data_name+'_test.csv', delimiter=',')

    xtr, ytr = train_data[:,:-1], train_data[:,-1:]
    xte, yte = test_data[:,:-1], test_data[:,-1:]

    ntr,dx = xtr.shape
    nte,dy = yte.shape

    print('Read ' + str(ntr) + ' training  and ' + str(nte) + ' test data points')
    print('Input dimension: ' + str(dx) + ', Output dimension: ' + str(dy))

    # Add Models
    models, results = [],[]
    # models.append(bnn.BayesianNeuralNetwork(dx,dy))
    models.append(gpmodel.GPmodel(DX = dx, DY = dy,ARD = True, LENGTHSCALE = 0.5))
    models.append(negsep.Negsep(DX = dx, DY = dy, N_HIDDEN = 50, TRAIN_EPOCHS = 5,TRAIN_ITER = 5,
                                 LEARNING_RATE = 0.01, R_EPI = 1, N_EPI = 4))
    models.append(dropout.Dropout(DX = dx, DY = dy, N_HIDDEN = 50, TRAIN_EPOCHS = 25,
                                 LEARNING_RATE = 0.01, N_SAMPLES = 100, DROPOUT_RATE = 0.05))

    for model in models:
        model_name = model.__class__.__name__
        results.append({'model_name':model_name})
        print('Processing '+ model_name + ':')
          
        # Training
        print('Training...')
        t0 = time()
        model.train(xtr,ytr)
        results[-1].update({'ttrain':time() - t0})
        
        # Evaluation
        print('Evaluating...')
        t0 = time()
        results[-1].update(model.evaluate(xte,yte, xtr, ytr))
        results[-1].update({'tevaluate':time() - t0})
        
        visualize(model,xtr, ytr, xte, yte)
    
    # Print and save results
    tab = tabulate([a.values() for a in results], headers=results[0].keys())
    print(tab)
    with open('../results/' + data_name + '.txt', 'w') as f:
        print(tab, file=f) 

print('Pau')


