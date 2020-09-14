import numpy as np

def scale_to_unit(epi):
    return (epi - np.min(epi)) / (np.max(epi) - np.min(epi))


def weighted_RMSE(yte, ypred, epi):
    epis = scale_to_unit(epi) + 0.000001
    RMSE = np.sqrt(((ypred - yte) ** 2).mean())
    print('RSME: ' + str(RMSE))
    discounted_RMSE = np.sqrt((((ypred - yte) ** 2) / epis).mean())
    print('Discounted RMSE: ' + str(discounted_RMSE))
    total_discount = epis.mean()
    print('Total Discount: ' + str(total_discount))
    return discounted_RMSE * total_discount