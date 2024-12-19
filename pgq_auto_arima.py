import numpy as np
import pmdarima as pm

def get_pdq_by_auto_arima(wineind):

    # fitting a stepwise model:
    stepwise_fit = pm.auto_arima(wineind, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=False, d=1, D=1, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)  # set to stepwise

    stepwise_fit.summary()
    qq = 0
    pdq = stepwise_fit.order
    return pdq[0],pdq[1],pdq[2]