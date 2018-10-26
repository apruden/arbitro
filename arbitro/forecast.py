from statsmodels.tsa.arima_model import ARIMA
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot 


class ARIMAForecaster:
    def init(self, data):
        n = int(len(data) * 0.9)
        train, test = data[:n], data[n:]
        self.model = ARIMA(train, order=(5, 0, 2))
        self.model_fit = self.model.fit(trend='nc', disp=0)
        pred = self.predict(n, n + len(test)-1)
        df = DataFrame({'Test': test, 'Pred': pred})
        df.plot()
        pyplot.savefig('totoarima.png')

    def predict(self, start, end):
        return self.model_fit.predict(start=start, end=end)


class Forecaster:
    def __init__(self):
        pass
 
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def init(self, data):
        import math
        pyplot.close('all')
        # load dataset
        dataset = DataFrame(data)
        values = dataset.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, 1, 1)
        #reframed.drop(reframed.columns[[3]], axis=1, inplace=True)

        values = reframed.values
        n_train = int(len(values) * 0.9) 
        train = values[:n_train, :]
        test = values[n_train:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-2], train[:, -2:]
        test_X, test_y = test[:, :-2], test[:, -2:]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        # design network
        self.model = Sequential()
        self.model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
        self.model.add(Dense(2))
        self.model.compile(loss='mae', optimizer='adam')
        # fit network
        history = self.model.fit(
                train_X,
                train_y,
                epochs=50,
                batch_size=4,
                validation_data=(test_X, test_y),
                verbose=0,
                shuffle=False)

        # make a prediction
        yhat = self.model.predict(test_X)

        logging.info(yhat.shape, test_X.shape)
    
        def invert(v, x):
            #v = concatenate((v, x), axis=1)
            v = scaler.inverse_transform(v)
            return v[:,:2]

        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = invert(yhat, test_X)
        # invert scaling for actual
        #test_y = test_y.reshape((len(test_y)/2, 2))
        inv_y = invert(test_y, test_X)
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        logging.info('Test RMSE: %.3f' % rmse)
        DataFrame({'pred': inv_yhat[:,0], 'pred1': inv_yhat[:,1], 'real': inv_y[:,0], 'real1': inv_y[:,1]}).plot()
        #DataFrame({'real': inv_y[:,0], 'real1': inv_y[:,1]}).plot()
        pyplot.savefig('toto.png')

    def predict(self, x, n):
        t = DataFrame(x)
        valuest = t.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(valuest)
        values = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
    
        yhat = self.model.predict(values)
   
        values = values.reshape((values.shape[0], values.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, values[:, 1:]), axis=1)
   
        return scaler.inverse_transform(inv_yhat)
    
    def update(self, data):
        t = DataFrame(data)
        valuest = t.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(valuest)
        reframed = self.series_to_supervised(scaled, 1, 1)
        reframed.drop(reframed.columns[[3]], axis=1, inplace=True)
        logging.info(list(reframed))
        values = reframed.values
        X, y = values[:, :-2], values[:, -2:]
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        for i in range(10):
            self.model.fit(X, y, epochs=1, batch_size=4, verbose=0, shuffle=False)
            self.model.reset_states()
