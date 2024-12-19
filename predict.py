from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def make_series(col10,years,column_name):
    import pandas as pd
    df = pd.DataFrame(col10, index=years, columns=[column_name])

    return df[column_name]

def predict(p,d,q,series):
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    # поточечная проверка на проверочном множестве


    return predictions

def find_pdq_predict(series):
    best_rmse = 1e6
    best_pdq = []

    for p in range(3):
        for d in range(1):
            for q in range(3):
                # разбиение на обучающееи проверочное подмножества
                X = series.values
                size = int(len(X) * 0.66)
                train, test = X[0:size], X[size:len(X)]
                history = [x for x in train]
                predictions = list()

                # поточечная проверка на проверочном множестве

                for t in range(len(test)):
                    model = ARIMA(history, order=(p,d,q))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                    print('p %d d %d q %d  predicted=%f, expected=%f' % (p,d,q,yhat, obs))
                    # оценка качества прогноза
                rmse = sqrt(mean_squared_error(test, predictions))
                print('Test RMSE: %.3f best_RMSE %e ' % (rmse,best_rmse))
                if rmse < best_rmse:
                   best_rmse = rmse
                   best_pdq = [p,d,q]
                # plot forecasts against actual outcomes
                # plt.plot(test)
                # plt.plot(predictions, color='red')
                # plt.show()
    from pgq_auto_arima import get_pdq_by_auto_arima
    p,d,q = get_pdq_by_auto_arima(series.values)
