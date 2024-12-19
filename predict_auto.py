import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from math import sqrt
from matplotlib import pyplot

def prolong_years(num_years,future_points):
    delta = num_years[-1]-num_years[-2]
    list_years = list(num_years)
    for i in range(1,future_points+1):
        list_years.append(num_years[-1]+i*delta)
    return np.array(list_years)



def predict(series,future_points,name,num_years):
    from pgq_auto_arima import get_pdq_by_auto_arima

    p,d,q = get_pdq_by_auto_arima(series.values)
    # разбиение на обучающееи проверочное подмножества
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # поточечная проверка на проверочном множестве
    for t in range(len(test)+future_points):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        if t < test.shape[0]:
           obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # оценка качества прогноза
    rmse = sqrt(mean_squared_error(test, predictions[:test.shape[0]]))
    mape = mean_absolute_percentage_error(test, predictions[:test.shape[0]])
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
   # pyplot.figure()
    years_train, years_test = num_years[0:size], num_years[size:len(X)]
    #pyplot.plot(years_test,test)
    num_years_prolonged = prolong_years(num_years,future_points)
    #pyplot.plot(num_years_prolonged,predictions, color='red')
    # pyplot.xlabel('years')
    # pyplot.show(block=True)
    # pyplot.savefig(name+'.png')
    extended_predictions = np.concatenate((train,predictions))

    return extended_predictions,mape,num_years_prolonged