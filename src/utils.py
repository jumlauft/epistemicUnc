import numpy as np

def Epi_RMSE(yte, ypred, epi):
    epis = epi + 0.000001
    RMSE = np.sqrt(((ypred - yte) ** 2).mean())
    discounted_RMSE = np.sqrt((((ypred - yte) ** 2) / epis).mean())
    discounted_RMSEi = np.sqrt((((ypred - yte) ** 2) * (1-epis)).mean())
    total_discount = epis.mean()

    return {'Epi_RMSE': discounted_RMSE * total_discount,
            'Epi_RMSEi': discounted_RMSEi / (1-total_discount),
            'RMSE': RMSE,
            'discounted_RMSE': discounted_RMSE,
            'discounted_RMSEi':discounted_RMSEi,
            'total_discount': total_discount}
