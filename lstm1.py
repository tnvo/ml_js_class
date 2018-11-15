import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# same as lstm, without description and is used for api
def run_lstm():
    dataset = pd.read_csv('Classification.csv', header=0)
    columns = ['JobSatisfaction', 'Age', 'DistanceFromHome', 'YearsInCurrentRole']
    df = dataset[columns]
    df.fillna(0, inplace=True)
    X = np.array(df.iloc[:, 1:])
    y = np.array(df['JobSatisfaction'])
    print(X.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(50, batch_size=1, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=1, verbose=1)
    X_test = np.array([[41, 1, 4]])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    predicted = model.predict(X_test)
    print(predicted[0][0])

    return str(predicted)
