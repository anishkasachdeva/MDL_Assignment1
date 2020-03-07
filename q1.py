import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

file1 = open('./Q1/data.pkl', 'rb')
file2 = pickle.load(file1)

training_set, testing_set = train_test_split(file2, test_size = 0.1, shuffle=True,random_state=42)

arr_of_training_sets = np.array(np.split(training_set,10))
# print(arr_of_training_sets.shape)
models=[]
predictions=[]
avg_of_predictions=[]
mean=0
y_test=testing_set[:, 1]
x_test=testing_set[:,0]
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

bias_arr=np.zeros((9))
bias_sq_arr=np.zeros((9))
variance_arr=np.zeros((9))
error_arr=np.zeros((9))


for degree in range(1,10):
    temp1=np.zeros((500, 1))
    first = 0
    matrix =[]

    for train_index in range(10):
        x=arr_of_training_sets[train_index,:, 0]
        y=arr_of_training_sets[train_index,:, 1]
        x = x.reshape( -1,1 )
        # print(x.shape)
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
    temp1/=10

    bias_sq_arr[degree-1]=np.mean((temp1-y_test)**2)
    # take row wise variance, then take mean of all resulting variances
    variance_arr[degree-1]=np.mean(np.var(matrix, axis=1))
    error_arr[degree-1]=variance_arr[degree-1]+bias_sq_arr[degree-1]
bias_arr=np.sqrt((bias_sq_arr))

print(pd.DataFrame(data={ "bias": bias_arr, "bias^2": bias_sq_arr, "variance": variance_arr ,"error": error_arr}))
xaxis = list(range(1, 10))
plt.plot(xaxis ,bias_arr)
plt.plot(xaxis ,variance_arr)
plt.show()
