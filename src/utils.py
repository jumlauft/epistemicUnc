import numpy as np
import matplotlib.pyplot as plt

def discounted_mse(y, ypred, epi):
    squared_error = (ypred - y) ** 2
    mse = squared_error.mean()
    total_discount = epi.mean()
    mse_discounted = (squared_error * (1-epi)).sum() / (1-epi).sum()
    return mse, mse_discounted, total_discount

def eval_discounted_mse(yte, ypredte, epite, ytr, ypredtr, epitr):
    mse_test, mse_discounted_test, total_discount_test = discounted_mse(yte, ypredte, epite)
    mse_train, mse_discounted_train, total_discount_train = discounted_mse(ytr, ypredtr, epitr)
    return {'mse_test': mse_test,
            'mse_discounted_test': mse_discounted_test,
            'total_discount_test': total_discount_test, 
            'mse_train': mse_train,
            'mse_discounted_train': mse_discounted_train,
            'total_discount_train': total_discount_train}


def visualize(model,xtr, ytr, xte, yte):
    ntr,dx = xtr.shape
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