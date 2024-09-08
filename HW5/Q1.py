import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(datapath):
    input = np.loadtxt(datapath)
    x = input[:,0].reshape(-1,1)
    y = input[:,1].reshape(-1,1)
    return x,y

def calculate_rational_quadratic_kernel(x1,x2,alpha,lengthscale):
    dis = np.sum(x1*x1, axis=1).reshape(-1, 1)+np.sum(x2*x2, axis=1)-2*x1@x2.T
    return np.power((1+dis/(2*alpha*(lengthscale**2))),-alpha)


def calculate_neg_log_likelihood(theta,x_train,y_train,b):
    k_train = calculate_rational_quadratic_kernel(x_train,x_train,theta[0],theta[1])
    covar = k_train+np.eye(len(x_train))*(1/b)

    neg_log_likelihood = np.log(np.linalg.det(covar))
    neg_log_likelihood += y_train.T@np.linalg.inv(covar)@y_train
    neg_log_likelihood += len(x_train)*np.log(2*np.pi)
    return 0.5*neg_log_likelihood

def perfrom_gaussian_process(x_train,y_train,b,alpha,lengthscale):
    x_test = np.linspace(-60.0,60,1000).reshape(-1,1)
    k_train = calculate_rational_quadratic_kernel(x_train,x_train,alpha,lengthscale)
    covar = k_train+np.eye(len(x_train))*(1/b)
    covar_inv = np.linalg.inv(covar)
    k_train_test = calculate_rational_quadratic_kernel(x_train,x_test,alpha,lengthscale)
    k_test = calculate_rational_quadratic_kernel(x_test,x_test,alpha,lengthscale)
    
    mean = k_train_test.T@covar_inv@y_train
    var = k_test+np.eye(len(x_test))*(1/b)-k_train_test.T@covar_inv@k_train_test
    
    return x_test,mean,var

def draw_fig(ax,x_train,y_train,x_test,mean,var,title):
    interval = 1.96 *np.sqrt(var.diagonal())    
    x_test = x_test.reshape(-1,)
    mean = mean.reshape(-1,)
    
    upper_line = mean+interval
    lowwer_line = mean-interval

    ax.set_title(title) 
    ax.plot(x_test, upper_line, color='orange',linewidth=0.5)
    ax.plot(x_test, lowwer_line, color='orange',linewidth=0.5)
    ax.fill_between(x_test, upper_line, lowwer_line, color='lemonchiffon')
    ax.plot(x_test, mean, color='b',linewidth=1.5)
    ax.scatter(x_train,y_train, color='k', s=8)
    
    plt.draw()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#05')
    parser.add_argument('--dataPath', type=str, default="input.data", help="path of the file")
    parser.add_argument('--b', type=float, default=5.0, help="noise for the function")
    setting = parser.parse_args()

    x_train, y_train = load_data(setting.dataPath)
    
    fig,ax = plt.subplots(1,2,figsize=(12,5))

    x_test,mean,var = perfrom_gaussian_process(x_train,y_train,setting.b,1.0,1.0)
    title = "alpha = 1.000, length scale = 1.000"
    draw_fig(ax[0],x_train,y_train,x_test,mean,var,title)


    init_par = [1.0,1.0]
    opt_par = minimize(calculate_neg_log_likelihood,init_par,args=(x_train,y_train,setting.b))
    
    x_test,mean,var = perfrom_gaussian_process(x_train,y_train,setting.b,opt_par.x[0],opt_par.x[1])
    title = "alpha =%.3f " %opt_par.x[0] +", length scale = %.3f"%opt_par.x[1] 
    draw_fig(ax[1],x_train,y_train,x_test,mean,var,title)

    fig.tight_layout()
    plt.show()