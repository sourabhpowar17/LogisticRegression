import numpy as np 
from helper import *
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
'''
Homework2: logistic regression classifier
'''

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def logistic_regression(data, label, max_iter, learning_rate):
	w=np.zeros((data.shape[1],1))
	a=0
	grad=0
	n= label.shape[0]
	y_new=np.zeros((n,1))
	#print('gradient',(learning_rate*gradient(w,data,label)))
	
	for i in range(max_iter):
		for i in range(n):
			
			if sigmoid(np.dot(data[i],w))>0.5:
				y_new[i]=1
			else:
				y_new[i]=-1

	grad=grad+np.multiply(data,label)
	ans=sigmoid((-1)*np.dot(label,np.dot(data,w)))
	for i in range(max_iter):
		for j in range(n):
			if y_new[j]!=label[j]:
				#print((learning_rate*gradient(w,data,label)))
				a=np.subtract(np.transpose(w),np.transpose((learning_rate*(-1)*grad*ans/n)))
				w=np.transpose(a)
	return w

def thirdorder(data):   
	data=np.append(data, np.ones((data.shape[0],8)), axis=1)
	data[:,2]=data[:,0]*data[:,1]#x1*x2
	data[:,3]=data[:,0]*data[:,0]#x1**2
	data[:,4]=data[:,1]*data[:,1]#x2**2
	data[:,5]=data[:,3]*data[:,0]#x1**3
	data[:,6]=data[:,4]*data[:,1]#x2**3
	data[:,7]=data[:,3]*data[:,1]#x1**2*x2
	data[:,8]=data[:,4]*data[:,0]#x2**2*x1
	return data

def accuracy(x, y, w):
	n, _ = x.shape
	mistakes = 0
	y_new=np.zeros((n,1))
	for i in range(n):
		if sigmoid(np.dot(x[i],w))>0.5:
			y_new[i]=1
		else:
			y_new[i]=-1
	for i in range(n):
		if y_new[i]!=y[i]:
			mistakes += 1


	return (n-mistakes)/n
	'''
	This function is used to compute accuracy of a logsitic regression model.
    
	    Args:
	    x: input data with shape (n, d), where n represents total data samples and d represents
		total feature numbers of a certain data sample.
		y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
	where d represents total feature numbers of a certain data sample.

    Return 
	accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
	which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
	'''


