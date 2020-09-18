import numpy as np
import matplotlib.pyplot as plt

def discounted_rmse(y, ypred, epi):
    squared_error = (ypred - y) ** 2
    rmse = np.sqrt(squared_error.mean())
    total_discount = epi.mean()
    rmse_discounted = np.sqrt((squared_error * (1-epi)).mean()) * total_discount
    return rmse, rmse_discounted, total_discount

def eval_discounted_rmse(yte, ypredte, epite, ytr, ypredtr, epitr):
    rmse_test, rmse_discounted_test, total_discount_test = discounted_rmse(yte, ypredte, epite)
    rmse_train, rmse_discounted_train, total_discount_train = discounted_rmse(ytr, ypredtr, epitr)
    return {'rmse_test': rmse_test,
            'rmse_discounted_test': rmse_discounted_test,
            'total_discount_test': total_discount_test, 
            'rmse_train': rmse_train,
            'rmse_discounted_train': rmse_discounted_train,
            'total_discount_train': total_discount_train}


def visualize(model,xtr, ytr, xte, yte):
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