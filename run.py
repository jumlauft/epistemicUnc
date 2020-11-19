import numpy as np
import pandas as pd
from src import negsep
from src import dropout
from src import gpmodel
from src import bnn
from tabulate import tabulate
from time import time
from src import utils


def main(SMOKE_TEST):
    fresults = './results/' 
    # Add Datasets
    data_sets = []
    if not SMOKE_TEST:
        data_sets.append("1D_centered")
        data_sets.append("1D_split")
        data_sets.append("2D_square")
        data_sets.append("2D_gaussian")
        data_sets.append("pmsm_temperature")
        data_sets.append("sarcos")

    else:
        data_sets.append("smoke")

    for data_name in data_sets:
        np.random.seed(1)

        # Read data
        print('Reading: ' + data_name + '...')
        train_data = np.genfromtxt('./data/' + data_name + '_train.csv',
                                   delimiter=',')
        test_data = np.genfromtxt('./data/' + data_name + '_test.csv',
                                  delimiter=',')

        xtr, ytr = train_data[:, :-1], train_data[:, -1:]
        xte, yte = test_data[:, :-1], test_data[:, -1:]

        ntr, dx = xtr.shape
        nte, dy = yte.shape

        print('Read ' + str(ntr) + ' training and ' + str(
            nte) + ' tests data points')
        print('Input dimension: ' + str(dx) + ', Output dimension: ' + str(dy))


        # Add Models
        models, mresults = [], []
        models.append(gpmodel.GPmodel(dx=dx, dy=dy, ARD=True,
                              LENGTHSCALE=0.5))
        models.append(bnn.BNN(dx=dx, dy=dy, N_HIDDEN=50, TRAIN_EPOCHS=2000,
                          LEARNING_RATE=0.01, N_SAMPLES=100, ))
        models.append(dropout.Dropout(dx=dx, dy=dy, N_HIDDEN=50,
                              TRAIN_EPOCHS=100, LEARNING_RATE=0.01,
                              N_SAMPLES=100, DROPOUT_RATE=0.05))
        models.append(negsep.Negsep(dx=dx, dy=dy, N_HIDDEN=50,
                             TRAIN_EPOCHS=20, TRAIN_ITER=5,
                             LEARNING_RATE=0.01, R_EPI=1, N_EPI=4))
        model_names = [m.__class__.__name__ for m in models]
        for model in models:
            np.random.seed(1)
            model_name = model.__class__.__name__
            mresults.append(dict()) #{'model_name': model_name}
            print(model_name + ':')

            # Training
            print('Training...')
            t0 = time()
            loss = model.train(xtr, ytr, display_progress=False)
            mresults[-1].update({'ttrain': time() - t0})

            # Evaluation
            print('Evaluating...')
            t0 = time()
            mresults[-1].update(model.evaluate(xte, yte, xtr, ytr, min_alpha=False))
            mresults[-1].update({'tevaluate': time() - t0})

            # Visualization
            if dx == 1 or dx == 2:
                modelte, epi = utils.visualize(model, xtr, ytr, xte, yte)

                utils.data2csv(fresults+ data_name + '_' + model_name + '.csv',
                         xtr=xtr, ytr=ytr, xte=xte, yte=yte,
                         modelte=modelte, epi=epi, loss=loss,
                         x_epi=model.get_x_epi(), y_epi=model.get_y_epi())
                if dx == 2:
                    utils.data2csv(
                        fresults + data_name + '_' + model_name + '_surf.csv',
                        xte=xte, yte=yte, modelte=modelte, epi=epi)



        # Print and save results
        df = pd.DataFrame(index=model_names,columns=mresults[0].keys(),
                         data=[a.values() for a in mresults])
        df.to_csv(fresults + data_name + '.csv')
        print(tabulate(df, headers="keys"))


    print('Pau')


if __name__ == "__main__":
    main(False)
