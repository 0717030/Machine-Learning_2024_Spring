import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy

def get_univar_gaussian_data(m,s):
    u1 = np.random.uniform(0,1)
    while u1 == 0:
        u1 = np.random.uniform(0,1)
    u2 = np.random.uniform(0,1)
    y = np.sqrt(-2.0*np.log(u1)*s)*np.cos(2.0*np.pi*u2)+m
    #y = np.sqrt(-2*np.log(u1)*s)*np.sin(2.0*np.pi*u2)+m
    return y

def draw_fig(ax,phi,label,num,title):
    ax.set_title(title)
    
    for idx in range(num):
        if label[idx]: # class 1
            ax.scatter(phi[idx][0],phi[idx][1],color='blue',s=6)
        else:
            ax.scatter(phi[idx][0],phi[idx][1],color='red',s=6)
            
    plt.draw()
    return

def calculate_first_derivative(phi,label,w):
    #return phi.transpose()@(1/(1+np.exp(-phi@w))-label)
    return phi.transpose()@(scipy.special.expit(phi@w)-label)
    
def perform_gradient_descent(phi,label):
    w = np.random.rand(3,1)
    dw = calculate_first_derivative(phi,label,w)
    w_new = w-dw
    count=1
    while((np.linalg.norm(dw)>1e-3) and count<1000):
        count += 1
        w = w_new
        dw = calculate_first_derivative(phi,label,w)
        w_new = w-dw
    print("count: ",count)
    return w_new

def perform_newton_method(phi,label):
    #w = np.random.rand(3,1)
    w = np.zeros((3,1))
    dw = calculate_first_derivative(phi,label,w)
    
    #D = np.diag((np.exp(-phi@w)*scipy.special.expit(-phi@w)*scipy.special.expit(-phi@w)).flatten())
    D = np.diag((np.exp(-phi@w)/(1+np.exp(-phi@w))/(1+np.exp(-phi@w))).flatten())
    H = phi.transpose()@D@phi
    if np.linalg.det(H) != 0: # H is invertible
        dw = np.linalg.inv(H)@dw
        flag=1
        
    
    w_new = w-dw
    count=1

    #print("count ",count," flag: ",flag)
    while((np.linalg.norm(dw)>1e-3) and count<2000):
        count += 1
        w = w_new
        
        dw = calculate_first_derivative(phi,label,w)
        #D = np.diag((np.exp(-phi@w)*(scipy.special.expit(-phi@w))*scipy.special.expit(-phi@w)).flatten())
        D = np.diag((np.exp(-phi@w)/(1+np.exp(-phi@w))/(1+np.exp(-phi@w))).flatten())
        H = phi.transpose()@D@phi
        if np.linalg.det(H): # H is invertible
            dw = np.linalg.inv(H)@dw
        
        w_new = w-dw
    print("count: ",count)
    
    return w_new
    

def perform_prediction(w,phi,num):
    prediction = np.zeros(num)
    
    for idx in range(num): #data (1,3)
        if (1/(1+np.exp(-phi[idx]@w))) >= 0.5:
            prediction[idx] = 1
        else:
            prediction[idx] = 0
        
    return prediction

def print_result(w,label,prediction,num):
    print("w:",w,"\n")
    
    confusion_matrix = np.zeros((2,2),dtype=int)
    for idx in range(num):
        if label[idx]==0 and prediction[idx]==0:
            confusion_matrix[0][0]+=1
        elif label[idx]==0 and prediction[idx]==1:
            confusion_matrix[0][1]+=1
        elif label[idx]==1 and prediction[idx]==0:
            confusion_matrix[1][0]+=1
        elif label[idx]==1 and prediction[idx]==1:
            confusion_matrix[1][1]+=1
        else:
            continue

    print("confusion matrix:")
    print(" "*15,"Predict cluster 1 Predict cluster 2")
    print("Is cluster 1\t\t",confusion_matrix[0][0],"\t"*2,confusion_matrix[0][1])
    print("Is cluster 2\t\t",confusion_matrix[1][0],"\t"*2,confusion_matrix[1][1])

    print("Sensitivity (Successfully predict cluster 1): ",confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]))
    print("Specificity (Successfully predict cluster 2): ",confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1]))
    print("\n","-"*50)
    return

def perform_logestic_regression(N,mx1,my1,mx2,my2,vx1,vy1,vx2,vy2):
    phi = np.ones((2*N,3)) #[[dx1,dy1,1]...[dx2,dy2,1]]
    label = np.zeros((2*N,1))
    label[N:] = 1

    for idx in range(N): #class 1
        phi[idx][0] = get_univar_gaussian_data(mx1,vx1)
        phi[idx][1] = get_univar_gaussian_data(my1,vy1)
    for idx in range(N,2*N): #class 2
        phi[idx][0] = get_univar_gaussian_data(mx2,vx2)
        phi[idx][1] = get_univar_gaussian_data(my2,vy2)
   
    
    fig,ax = plt.subplots(1,3,figsize=(5,5))
    draw_fig(ax[0],phi,label.flatten(),2*N,"ground truth")

    print("Gradient descent:\n")
    w_gd = perform_gradient_descent(phi,label)
    prediction_gd = perform_prediction(w_gd,phi,2*N)
    print_result(w_gd,label,prediction_gd,2*N)
    draw_fig(ax[1],phi,prediction_gd,2*N,"gradient descent")

    print("Newton's method:\n")    
    w_nm = perform_newton_method(phi,label)
    prediction_nm = perform_prediction(w_nm,phi,2*N)
    print_result(w_nm,label,prediction_nm,2*N)
    draw_fig(ax[2],phi,prediction_nm,2*N,"newton's method")
    
    fig.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#04')
    parser.add_argument('--N', type=int, default=50, help="number of data points")
    parser.add_argument('--mx1', type=float, default=1.0, help="Expectation value or mean")
    parser.add_argument('--vx1', type=float, default=2.0, help="Variance")
    parser.add_argument('--my1', type=float, default=1.0, help="Expectation value or mean")
    parser.add_argument('--vy1', type=float, default=2.0, help="Variance")
    parser.add_argument('--mx2', type=float, default=3.0, help="Expectation value or mean")
    parser.add_argument('--vx2', type=float, default=4.0, help="Variance")
    parser.add_argument('--my2', type=float, default=3.0, help="Expectation value or mean")
    parser.add_argument('--vy2', type=float, default=4.0, help="Variance")

    setting = parser.parse_args()
    perform_logestic_regression(setting.N,setting.mx1,setting.my1,setting.mx2,setting.my2,setting.vx1,setting.vy1,setting.vx2,setting.vy2)