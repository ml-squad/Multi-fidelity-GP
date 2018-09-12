# -*- coding: utf-8 -*-

import GPy
import numpy as np
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError

def train_multifidelity_GP(x_train_high,y_train_high,x_train_low,y_train_low,i):

    input_dim=x_train_high.shape[1]

    '''Train level 1'''
    k1=GPy.kern.RBF(input_dim,ARD=False)
    scaler=preprocessing.StandardScaler().fit(y_train_low.reshape(-1,1))
    y_train_low_scaled=scaler.transform(y_train_low.reshape(-1,1))
    model1 = GPy.models.GPRegression(X=x_train_low, Y=y_train_low_scaled, kernel=k1)
    
    model1[".*Gaussian_noise"] = model1.Y.var()*0.1
    model1[".*Gaussian_noise"].fix()
    
    model1.optimize(max_iters = 500)
    model1[".*Gaussian_noise"].unfix()
    model1[".*Gaussian_noise"].constrain_positive()
    
    model1.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000,verbose=False)
    
    mu1,v1=model1.predict(x_train_high)

    
    #    start_2=time.time()
    '''Train level 2'''
    XX = np.hstack((x_train_high, mu1))
    

    k2 = GPy.kern.RBF(1,active_dims=[input_dim])*GPy.kern.RBF(input_dim,active_dims=np.arange(input_dim)) + GPy.kern.RBF(input_dim,active_dims=np.arange(input_dim))
    y_train_high_scaled=scaler.transform(y_train_high.reshape(-1,1))
    model2 = GPy.models.GPRegression(X=XX, Y=y_train_high_scaled, kernel=k2)
    
    model2[".*Gaussian_noise"] = model2.Y.var()*0.01
    model2[".*Gaussian_noise"].fix()
    
    model2.optimize(max_iters = 500)
    model2[".*Gaussian_noise"].unfix()
    model2[".*Gaussian_noise"].constrain_positive()
    
    model2.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000,verbose=False)

    return [model1,model2],scaler

def predict_multifidelity_GP(model_list,Nts,x_test_high,scaler):
    model1=model_list[0]
    model2=model_list[1]
    

    '''Predict at test points'''
    # sample f_1 at xtest
    nsamples = 100
    mu_1, C_1 = model1.predict(x_test_high, full_cov=True)
    Z = np.random.multivariate_normal(mu_1.flatten(),C_1,nsamples)

    # push samples through f_2
    tmp_m = np.zeros((nsamples,Nts))
    tmp_v = np.zeros((nsamples,Nts))
    for j in range(0,nsamples):
        mu, v = model2.predict(np.hstack((x_test_high, Z[j,:][:,None]))) #predict得到的是Nts*1的结果
        tmp_m[j,:] = mu.flatten()
        tmp_v[j,:] = v.flatten()

    # get posterior mean and variance
    mean = np.mean(tmp_m, axis = 0)[:,None]
    var = np.mean(tmp_v, axis = 0)[:,None]+ np.var(tmp_m, axis = 0)[:,None]
    var = np.abs(var)
    var=var.ravel()
    
    mean_rescaled=scaler.inverse_transform(mean.ravel())
    
    return mean_rescaled,var



if __name__=='__main__':
    x_train_high=np.loadtxt('X_H.txt')
    x_train_low=np.loadtxt('X_L.txt')
    x_test=np.loadtxt('X_test.txt')
    y_train_high=np.loadtxt('Y_H.txt')
    y_train_low=np.loadtxt('Y_L.txt')
    y_test=np.loadtxt('Y_test.txt')
    
    Nobs=y_train_high.shape[1]  # dimension of outputs
    Nts=x_test.shape[0]      # number of testing points
    y_pred=np.zeros((Nts,Nobs))
    var_pred=np.zeros_like(y_pred)
    start=time.time()
    
    for i in range(Nobs):   # train each output dimension independently
        print('this is {0} dimension'.format(i))
        model_list,scaler=train_multifidelity_GP(x_train_high,y_train_high[:,i],x_train_low,y_train_low[:,i],i)
        y_pred[:,i],var_pred[:,i]=predict_multifidelity_GP(model_list,Nts,x_test,scaler)

    
    coef=np.corrcoef(y_pred.ravel(),y_test.ravel())
    print('corr coef:',coef[0,1])
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.plot([1.1*np.min(y_pred),1.1*np.max(y_pred)],[1.1*np.min(y_test),1.1*np.max(y_test)],'r',lw=2)
    plt.scatter(y_pred.ravel(),y_test.ravel())
    plt.show()
    
    
    
    
    
   