import numpy as np
from src.negsep import Negsep
from src.dropout import Dropout
from src.gpmodel import GPmodel
from src.bnn import BNN
from tabulate import tabulate
from time import time
from src.utils import visualize, data2csv

def main(SMOKE_TEST):

    # Add Datasets
    data_sets = []
    if not SMOKE_TEST:
        data_sets.append("synthetic_data_1D")
        data_sets.append("synthetic_data_1D_split")
        data_sets.append("synthetic_data_2D_square")
        data_sets.append("synthetic_data_2D_gaussian")
        data_sets.append("pmsm_temperature")
        data_sets.append("sarcos")
    else:
        data_sets.append("smoke")



    for data_name in data_sets:
        np.random.seed(1)
        # Read data
        print('Read data: ' + data_name + '...')
        train_data = np.genfromtxt('../data/'+data_name+'_train.csv', delimiter=',')
        test_data = np.genfromtxt('../data/'+data_name+'_test.csv', delimiter=',')

        xtr, ytr = train_data[:,:-1], train_data[:,-1:]
        xte, yte = test_data[:,:-1], test_data[:,-1:]

        ntr, dx = xtr.shape
        nte, dy = yte.shape

        print('Read ' + str(ntr) + ' training  and ' + str(nte) + ' tests data points')
        print('Input dimension: ' + str(dx) + ', Output dimension: ' + str(dy))

        # Add Models
        models, results = [],[]
        models.append(GPmodel(dx = dx, dy = dy, ARD = True,
                                      LENGTHSCALE = 0.5))
        models.append(BNN(dx = dx, dy = dy, N_HIDDEN = 50, TRAIN_EPOCHS = 2000,
                                      LEARNING_RATE = 0.01, N_SAMPLES = 1000,))
        models.append(Dropout(dx = dx, dy = dy, N_HIDDEN = 50,
                                      TRAIN_EPOCHS = 100, LEARNING_RATE = 0.01,
                                      N_SAMPLES = 100, DROPOUT_RATE = 0.05))
        models.append(Negsep(dx = dx, dy = dy, N_HIDDEN = 50,
                                    TRAIN_EPOCHS = 20,TRAIN_ITER = 5,
                                    LEARNING_RATE = 0.01, R_EPI = 1, N_EPI = 4))

        for model in models:
            np.random.seed(1)
            model_name = model.__class__.__name__
            results.append({'model_name':model_name})
            print(model_name + ':')

            # Training
            print('Training...')
            t0 = time()
            loss = model.train(xtr,ytr, display_progress = False)
            results[-1].update({'ttrain':time() - t0})

            # Evaluation
            print('Evaluating...')
            t0 = time()
            results[-1].update(model.evaluate(xte, yte, xtr, ytr))
            results[-1].update({'tevaluate':time() - t0})

            if dx == 1 or dx == 2:
                modelte, epi = visualize(model,xtr, ytr, xte, yte)

                data2csv('../results/' + data_name + '_' + model_name + '.csv',
                         xtr = xtr, ytr = ytr, xte = xte, yte = yte,
                         modelte = modelte, epi = epi, loss = loss,
                         x_epi = model.get_x_epi(), y_epi = model.get_y_epi())
        # Print and save results
        tab = tabulate([a.values() for a in results], headers=results[0].keys())
        print(tab)
        with open('../results/' + data_name + '.txt', 'w') as f:
            print(tab, file=f)

    print('Pau')

if __name__ == "__main__":
    main(False)

