import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def discounted_mse(y, ypred, epi):
    """ Computes discounted mean squared error
    Args:
        y: outputs
        ypred: predicted outputs
        epi: epistemic uncertainty

    Returns:
        mse: mean squared error
        mse_discounted: discounted mean squared error
        total_discount: mean of predicted epistemic uncertainty
    """
    squared_error = (ypred - y) ** 2
    mse = squared_error.mean()
    total_discount = epi.mean()
    mse_discounted = (squared_error * (1 - epi)).sum() / (1 - epi).sum()
    return mse, mse_discounted, total_discount


def eval_discounted_mse(yte, ypredte, epite, ytr, ypredtr, epitr):
    """ Wrapper to computed discounted mean square error

    Args:
        yte: outputs test
        ypredte: predicted test
        epite: epistemic uncertainty test
        ytr: outputs train
        ypredtr: predicted train
        epitr: epistemic uncertainty train

    Returns:
        dict: contains mse, mse_discounted, total_discount for test/train data
    """
    mse_test, mse_discounted_test, total_discount_test = \
        discounted_mse(yte, ypredte, epite)
    mse_train, mse_discounted_train, total_discount_train = \
        discounted_mse(ytr, ypredtr, epitr)
    return {'mse_test': mse_test,
            'mse_discounted_test': mse_discounted_test,
            'total_discount_test': total_discount_test,
            'mse_train': mse_train,
            'mse_discounted_train': mse_discounted_train,
            'total_discount_train': total_discount_train}


def data2csv(fout, **kwargs):
    """ save lists or 1D/2D numpy arrays to csv

    Args:
        fout: filename
        **kwargs: arrays/lists to store
    """
    dfs = []
    for name in kwargs:
        try:
            if type(kwargs[name]) is list:
                kwargs[name] = np.array(kwargs[name])
            if kwargs[name].ndim == 1:
                dfs.append(pd.DataFrame(data = kwargs[name], columns = [name]))
            elif kwargs[name].ndim == 2:
                for i, col in enumerate(kwargs[name].T):
                    dfs.append(pd.DataFrame(data=col, columns=[name + '_' + str(i)]))
            else:
                raise Exception("must be 1 or 2 dimensional")
        except:
            print('Ignored ' + name)
    pd.concat(dfs,axis=1).to_csv(fout)

def visualize(model, xtr, ytr, xte, yte):
    """ Plots 1D or 2D visualization of train/test data and epistemic prediction

    Args:
        model: EpiModel
        xtr: input train data
        ytr: output train data
        xte: input test data
        yte: output test data

    Returns:
        modelte: model prediction at test inputs
        epi: epistemic uncertaint at test inputs
    """
    ntr, dx = xtr.shape
    if dx == 1:
        modelfig = plt.figure(figsize=(10, 5))
        modelfig.suptitle(model.__class__.__name__)
        modelte, epi = model.predict(xte)
        ax = modelfig.add_subplot(121)
        ax.set_title('Model Prediction')
        ax.plot(xtr, ytr, 'o', color="blue", label='training')
        ax.plot(xte, modelte, color="red", label='model')
        ax.plot(xte, yte, color="orange", label='tests')
        ax.legend()

        ax = modelfig.add_subplot(122)
        ax.set_title('Epistemic Uncertainty')
        plt.plot(xte, epi, color="blue")
        plt.plot(xtr, np.zeros(ntr), 'x', color="red")
        x_epi, y_epi = model.get_x_epi(), model.get_y_epi()
        if x_epi is not None:
            plt.plot(x_epi[:, 0], y_epi, 'o', color="green")

    elif dx == 2:
        modelfig = plt.figure(figsize=(10, 5))
        modelfig.suptitle(model.__class__.__name__)
        modelte, epi = model.predict(xte)
        ax = modelfig.add_subplot(121, projection='3d')
        ax.set_title('Model Prediction')
        ax.scatter(xtr[:, 0], xtr[:, 1], ytr, color="blue", label='training')
        ax.scatter(xte[:, 0], xte[:, 1], modelte, color="red", label='model')
        ax.scatter(xte[:, 0], xte[:, 1], yte[:, 0], color="orange",
                   label='tests')
        ax.legend()

        ax = modelfig.add_subplot(122, projection='3d')
        ax.set_title('Epistemic Uncertainty')
        ax.scatter(xte[:, 0], xte[:, 1], epi, color="blue")
        ax.scatter(xtr[:, 0], xtr[:, 1], np.zeros(ntr), color="red")
        x_epi, y_epi = model.get_x_epi(), model.get_y_epi()
        if x_epi is not None:
            ax.scatter(x_epi[:, 0], x_epi[:, 1], y_epi, color="green")
    else:
        modelte, epi = None, None
    return modelte, epi

