#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


inputs = [2.3,4.5,1.3]
weights = [3.2,-1.9,2.5]
bias = -1


# In[3]:


def neuron(x,w,b):
    z = np.dot(x,w)+b
    return z


# In[4]:


neuron(inputs, weights, bias)


# In[5]:


def sigmoid(z):
    return 1/(1+ np.exp(-z))


# In[6]:


def relu(z):
    return np.maximum(0,z)


# In[7]:


def neuron_act(x,w,b,phi):
    z = np.dot(w,x)+b
    y = phi(z)
    return y


# In[8]:


neuron_act(inputs,weights,bias,sigmoid)


# In[9]:


neuron_act(inputs,weights,bias,relu)


# In[11]:


x = np.linspace(-10,10,100)
y = sigmoid(x)


# In[14]:


plt.plot(x,y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sigmoidal Function')


# In[15]:


x = np.linspace(-10,10,100)
y = relu(x)


# In[19]:


plt.plot(x,y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Relu Function')

