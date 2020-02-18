#!/usr/bin/env python
# coding: utf-8

# In[732]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot


# In[733]:


excel_file = 'house_prices_data_training_data.csv'
housePrices = pd.read_csv("house_prices_data_training_data.csv", error_bad_lines=False )


# In[734]:


housePrices.head()


# In[735]:


#HousePricesNew = pd.read_csv("house_prices_data_training_data.csv",  index_col=0)
#HousePricesNew.head()


# In[736]:


housePrices.shape


# In[737]:


housePrices.tail()


# In[738]:


HousePricesNoEmpty=housePrices.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


# In[739]:


HousePricesNoEmpty.shape


# In[740]:


train, validate, test = np.split(HousePricesNoEmpty.sample(frac=1), [int(.6*len(HousePricesNoEmpty)), int(.8*len(HousePricesNoEmpty))])


# In[741]:


train


# In[742]:


validate


# In[743]:


test


# In[744]:


def plotBedrooms (Set):
    Price = Set["price"] 
    NoOfBedrooms= Set["bedrooms"]
    PriceList=[]
    BedroomsList=[]
    PriceList=list(Price)
    BedroomsList=list(NoOfBedrooms)
    plt.plot(BedroomsList, PriceList, 'ro', ms=10, mec='k')
    plt.xlabel('Number Of Bedrooms->')
    plt.ylabel('Prices->')
    plt.title('House Prices')
    plt.show()


# In[745]:


plotBedrooms(train)


# In[746]:


def featureNorm(X):

#i=3
#while i < 21:
#    X = [setNorm.iloc[0::,i]]
#    normalized_X = preprocessing.normalize(X)
#    setNorm.iloc[0::,i]=(normalized_X).transpose()
#    i+=1
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# In[ ]:





# In[747]:


training= train.copy()
trainnorm, mu, sigma = featureNorm(training.iloc[0::,3::])
y=training.iloc[0::,2]
yT = test.iloc[0::,2]
yV = validate.iloc[0::,2]
m=y.size
mT = yT.size
mV = yV.size
X = np.concatenate([np.ones((m, 1)), trainnorm], axis=1)
validating= validate.copy()
yval=validating.iloc[0::,2]
mval=yval.size
validatenorm, mu, sigma = featureNorm(validating.iloc[0::,3::])
validation = np.concatenate([np.ones((mval, 1)), validatenorm], axis=1)
testc= test.copy()
testnorm, mu, sigma = featureNorm(testc.iloc[0::,3::])
testingg = np.concatenate([np.ones((mT, 1)), testnorm], axis=1)
first_theta = np.zeros(X.shape[1])
hyp1 = np.dot(X, first_theta)
hyp2 = np.dot(np.power(X,2) , first_theta)
hyp3X= X.copy()
hyp3X[:, 2] = np.power(hyp3X[:, 2], 3)
hyp3 = np.dot(hyp3X , first_theta)
hyp1Test = np.dot(testingg, first_theta)
hyp2Test = np.dot(np.power(testingg,2) , first_theta)
hyp3t= testingg.copy()
hyp3t[:, 2] = np.power(hyp3t[:, 2], 3)
hyp3test = np.dot(hyp3t , first_theta)
lambdaList = [0.02, 0, 0.005, 1 , 5]
arrayH1 =[]
arrayH2 =[]
arrayH3 =[]
validation.shape[1]


# In[748]:


def computeCostMulti(theta, X, y, h, lambda_,m):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    
    #prod= y-(np.dot(X , theta));
    #div=2*m;
    #J = np.dot((prod),(prod))/ div;
    J = np.dot((h - y), (h - y)) / (2 * m) + ((lambda_/(2 * m))* np.sum(np.dot(theta, theta)))
    # ==================================================================
    return J


# In[749]:


def costFunction(y, h, m):
    J= np.dot((h - y), (h - y)) / (2 * m)
    return J


# In[750]:


def gradientDescentMulti(X, y, theta, alpha, num_iters, lambda_):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
       
        #summation= np.dot(X, theta)
        #mul=np.dot(X.T,summation-y)
        #theta=theta-((alpha/m)*(mul))
        # =================================================================
        
        # save the cost J in every iteration
        #J_history.append(computeCostMulti(X, y, theta))
        alphabym = alpha / m
        h = np.dot(X, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(computeCostMulti(theta, X, y, h, lambda_,m))

    
    return theta, J_history


# In[751]:


def gradientDescent2(X, y, theta, alpha, num_iters , lambda_):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(np.power(X,2), theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(computeCostMulti(theta, X, y, h, lambda_,m))

    return theta, J_history


# In[752]:


def gradientDescent3(X, y, theta, alpha, num_iters ,lambda_):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):
        alphabym = alpha / m
        h = np.dot(hyp3X, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(computeCostMulti(theta, X, y, h, lambda_,m))

    return theta, J_history


# In[753]:



iterations = 150
alpha = 0.05
alpha2=0.0029
alpha3=0.02 
hyp3vali= validation.copy()
hyp3vali[:, 2] = np.power(hyp3vali[:, 2], 3)

for i in lambdaList:
    theta, J_history = gradientDescentMulti(X,y, first_theta, alpha, iterations, i )

    theta2, J_history2 = gradientDescent2(X,y, first_theta, alpha2, iterations, i)

    theta3, J_history3 = gradientDescent3(hyp3X,y, first_theta, alpha3, iterations, i)
    
    h1validating = np.dot(validation, theta)
    
    h2validating = np.dot(np.power(validation, 2), theta2)
    
    h3validating = np.dot(hyp3vali, theta3)
    
    arrayH1.append(costFunction(yV, h1validating, mV))
    
    arrayH2.append(costFunction(yV, h2validating, mV))
    
    arrayH3.append(costFunction(yV, h3validating, mV))

lambda_1= lambdaList[min(range(len(arrayH1)), key=arrayH1.__getitem__)]
lambda_2= lambdaList[min(range(len(arrayH2)), key=arrayH2.__getitem__)]
lambda_3= lambdaList[min(range(len(arrayH3)), key=arrayH3.__getitem__)]


# In[754]:


theta, J_history = gradientDescentMulti(X,y, first_theta, alpha, iterations,lambda_1)
theta2, J_history2 = gradientDescent2(X,y, first_theta, alpha2, iterations,lambda_2)
theta3, J_history3 = gradientDescent3(hyp3X,y, first_theta, alpha3, iterations,lambda_3)
plt.figure()
plt.plot(np.arange(len(J_history)), J_history,'r', lw=2, label='h1')
plt.plot(np.arange(len(J_history2)), J_history2, lw=2, label='h2')
plt.plot(np.arange(len(J_history3)), J_history3,'g', lw=2, label='h3')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost Function')
plt.show()


# In[755]:


def getFinalSet (Set):
    
    Setcopy= Set.copy()
    Setnorm, mu, sigma = featureNorm(Setcopy.iloc[0::,3::])
    y=Setcopy.iloc[0::,2]
    m=y.size
    finalSet = np.concatenate([np.ones((m, 1)), Setnorm], axis=1)
    
    return finalSet 


# In[756]:


def gethypothesis (Set, hypnumber):
    if hypnumber == "hyp1":
        hyp = np.dot(Set, first_theta)
    elif hypnumber == "hyp2":
        hyp = np.dot(np.power(Set,2) , first_theta)
    else:
        hyp3X= Set.copy()
        hyp3X[:, 2] = np.power(hyp3X[:, 2], 3)
        hyp = np.dot(hyp3X , first_theta)
        


# In[757]:


kfold=[1,2]
lambdaList = [0.02, 0, 0.005, 1 , 5]

for k in kfold:
    traink, validatek, testk = np.split(HousePricesNoEmpty.sample(frac=1), [int(.6*len(HousePricesNoEmpty)), int(.8*len(HousePricesNoEmpty))])
    plotBedrooms(traink)
    normalizedSet=getFinalSet(traink)
    normalizedvaliSet=getFinalSet(validatek)
    first_theta = np.zeros(normalizedSet.shape[1])
    hypothesis1=gethypothesis(normalizedSet,"hyp1")
    hypothesis2=gethypothesis(normalizedSet,"hyp2")
    hypothesis3=gethypothesis(normalizedSet,"hyp3")
    arrayHk1 =[]
    arrayHk2 =[]
    arrayHk3 =[]
    hyp3vali= normalizedvaliSet.copy()
    hyp3vali[:, 2] = np.power(hyp3vali[:, 2], 3)
    for i in lambdaList:
        theta, J_history = gradientDescentMulti(normalizedSet,y, first_theta, alpha, iterations, i )

        theta2, J_history2 = gradientDescent2(normalizedSet,y, first_theta, alpha2, iterations, i)

        theta3, J_history3 = gradientDescent3(normalizedSet,y, first_theta, alpha3, iterations, i)
    
        h1validating = np.dot(normalizedvaliSet, theta)
    
        h2validating = np.dot(np.power(normalizedvaliSet, 2), theta2)
    
        h3validating = np.dot(hyp3vali, theta3)
    
        arrayHk1.append(costFunction(yV, h1validating, mV))
    
        arrayHk2.append(costFunction(yV, h2validating, mV))
    
        arrayHk3.append(costFunction(yV, h3validating, mV))

    lambda_1= lambdaList[min(range(len(arrayHk1)), key=arrayHk1.__getitem__)]
    lambda_2= lambdaList[min(range(len(arrayHk2)), key=arrayHk2.__getitem__)]
    lambda_3= lambdaList[min(range(len(arrayHk3)), key=arrayHk3.__getitem__)]
    theta, J_history = gradientDescentMulti(normalizedSet,y, first_theta, alpha, iterations,lambda_1)
    theta2, J_history2 = gradientDescent2(normalizedSet,y, first_theta, alpha2, iterations,lambda_2)
    theta3, J_history3 = gradientDescent3(normalizedSet,y, first_theta, alpha3, iterations,lambda_3)
    plt.figure()
    plt.plot(np.arange(len(J_history)), J_history,'r', lw=2, label='h1')
    plt.plot(np.arange(len(J_history2)), J_history2, lw=2, label='h2')
    plt.plot(np.arange(len(J_history3)), J_history3,'g', lw=2, label='h3')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost Function')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




