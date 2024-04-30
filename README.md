# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: R Vignesh
RegisterNumber: 212222230172

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```

## Output:
## Array Value of x:
![238196875-48635643-f05c-48cb-8b21-d5b0d7e60bf5](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/1883ae61-8378-4e77-88dd-f6a842280106)
## Array Value of y:
![238196912-03841143-dec0-4bdf-beb0-08894db9b552](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/78809b5d-c018-44c5-b8c6-a336be23480e)
## Score graph:
![238196954-8a1ab219-1661-4a15-ba5a-222974a4a948](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/9915fb5b-cfe7-463a-b7fb-f935e61f9e39)
## Sigmoid function graph:
![238196954-8a1ab219-1661-4a15-ba5a-222974a4a948](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/ee1b4cd5-24ee-477d-abc6-ac4b63c820a8)
## X_train_grad value:
![238197045-ff2cbc3f-c96d-4044-96a0-b60f72d18c67](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/b922f066-8cc2-4bbc-869d-63caf7f4ad19)
## Y_train_grad value:
![238197137-50d9a6fa-887a-4950-b765-1111b31db73e](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/9c5615bd-c494-4838-ae72-47c5508362eb)
## res.x:
![238197098-d7bb6b4e-1bab-45a8-8f9a-4b6f6dd23000](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/4b7c6340-a06f-4aae-9157-79c8c61c6e38)
## Decision boundary:
![238197103-f16e3f91-1835-4381-951f-d11f1892a209](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/46d2eb75-9c1e-41ae-b146-8cc165135391)
## Proability value:
![238197170-b43d07fe-66d0-4d54-bea3-034ee9d34eac](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/d83cc59c-aeb6-4177-8c40-fd7721629f1c)
## Prediction value of mean:
![238196842-87b2bcb1-9f69-40fd-9da7-789d03b3db7a](https://github.com/Senthamil1412/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119120228/fa679517-e30b-48c3-ad83-ffa103396638)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

