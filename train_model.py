
import numpy as np 
import pandas as pd 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv('data.csv', usecols=['Param1','Param2','Power', 'Gamma Point','Harmonic Number','A1 Real','A1 Imaginary','B1 Real','B1 Imaginary','A2 Real','A2 Imaginary','B2 Real','B2 Imaginary','I1','I2']) 

df = df.sample(frac=1).reset_index(drop=True)

X = df[['Param1','Param2','Power', 'Gamma Point','Harmonic Number','A1 Real','A1 Imaginary','B1 Real','B1 Imaginary','A2 Real','A2 Imaginary','B2 Real','B2 Imaginary']].values.tolist()

df.apply(pd.to_numeric)

I1I2 = df[['I1', 'I2']].values.tolist()

I2 = df[['I2']].values.tolist()
I2 = [item for sublist in I2 for item in sublist]

X_train = np.array(X[:6500])
y_train = np.array(I1I2[:6500])
from sklearn import tree

reg = tree.DecisionTreeRegressor(random_state=2).fit(X_train, y_train)

import pickle

with open('DecisionTree.pkl', 'wb') as f:
    pickle.dump(reg, f)


with open('DecisionTree.pkl', 'rb') as f:
  reg = pickle.load(f)

X_test = np.array(X[500:])
y_test = np.array(I1I2[500:])

y_pred = reg.predict(np.array(X_test))

_y0 = [item[0] for item in y_pred]
y0 = [item[0] for item in y_test]

_y1 = [item[1] for item in y_pred]
y1 = [item[1] for item in y_test]


mean_absolute_percentage_error_I1 = np.sqrt(mean_absolute_percentage_error(y0, _y0))
mean_absolute_percentage_error_I2 = np.sqrt(mean_absolute_percentage_error(y1, _y1))

print("Average percentage error when predicting I1: ", mean_absolute_percentage_error_I1, "%");

print("Average percentage error when predicting I2: ", mean_absolute_percentage_error_I2, "%");
