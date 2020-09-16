import numpy as np

def weighted_RMSE(yte, ypred, epi):
    epis = epi + 0.000001
    RMSE = np.sqrt(((ypred - yte) ** 2).mean())
    print('RMSE: ' + str(RMSE))
    discounted_RMSE = np.sqrt((((ypred - yte) ** 2) / epis).mean())
    print('Discounted RMSE: ' + str(discounted_RMSE))
    total_discount = epis.mean()
    print('Total Discount: ' + str(total_discount))
    return discounted_RMSE * total_discount, RMSE, \
           discounted_RMSE, total_discount