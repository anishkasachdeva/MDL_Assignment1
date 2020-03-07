import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

file1 = open('./Q2/X_test.pkl', 'rb')
file2 = open('./Q2/Fx_test.pkl', 'rb')
file3 = open('./Q2/X_train.pkl', 'rb')
file4 = open('./Q2/Y_train.pkl', 'rb')

x_test=pickle.load(file1)
y_test=pickle.load(file2)
x_train=pickle.load(file3)
y_train=pickle.load(file4)
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

bias_arr=np.zeros((9))
bias_sq_arr=np.zeros((9))
variance_arr=np.zeros((9))
error_arr=np.zeros((9))


for degree in range(1,10):
    temp1=np.zeros((80, 1))
    first = 0
    matrix =[]
    for train_index in range(20):
        x=x_train[train_index]
        y=y_train[train_index]
        x = x.reshape( -1,1 )
        y = y.reshape(-1, 1)
        model = make_pipeline(PolynomialFeatures(degree=degree),LinearRegression())
        model.fit(x,y)
        pred = model.predict(x_test)
        if first == 0:
            matrix = pred
            first = 1
        else:
            matrix = np.hstack((matrix, pred))
        temp1+= pred
    temp1/=20
    bias_sq_arr[degree-1]=np.mean((temp1-y_test)**2)
    # bias_arr[degree-1]=np.mean(np.abs(temp1-y_test))
    variance_arr[degree-1]=np.mean(np.var(matrix, axis=1))
    error_arr[degree-1]=bias_sq_arr[degree-1]+variance_arr[degree-1]
bias_arr=np.sqrt((bias_sq_arr))
print(pd.DataFrame(data={ "bias": bias_arr, "bias^2": bias_sq_arr, "variance": variance_arr ,"error": error_arr}))
# print(bias_arr)
# print(bias_sq_arr)
# print(variance_arr)
xaxis = list(range(1, 10))
# plt.plot(xaxis ,bias_arr)
plt.plot(xaxis,bias_sq_arr)
plt.plot(xaxis ,variance_arr)
plt.show()
