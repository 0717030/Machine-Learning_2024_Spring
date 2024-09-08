import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_univar_gaussian_data(m,s):
    u1 = np.random.uniform(0,1)
    while u1 == 0:
        u1 = np.random.uniform(0,1)
    u2 = np.random.uniform(0,1)
    y = np.sqrt(-2.0*np.log(u1)*s)*np.cos(2.0*np.pi*u2)+m
    #y = np.sqrt(-2*np.log(u1)*s)*np.sin(2.0*np.pi*u2)+m
    return y

def get_poly_basis_linear_data(n,a,w):# [w0,w1...]
    e = get_univar_gaussian_data(0,a)
    x_value = np.random.uniform(-1,1)
    #print("e = ",e,"; x_value = ",x_value)
    x=np.zeros((n, 1)) #[1,x_value, x_value^2,...]
    
    while x_value == -1:
        x_value = np.random.uniform(-1,1)
    
    for expo in range(n):
        x[expo][0]=(x_value ** expo)
       
    y_value = w@x + e
    
    return x_value,float(y_value)

def draw_fig(ax,data_list,a,w,n,posterior_mean,posterior_var,title):
    ax.set_title(title)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-15, 20)
    x_line = np.linspace(-2.0,2.0,50)

    if len(data_list[0])==1:
        func = np.poly1d(np.flip(w))        
        y_line = func(x_line)
        ax.plot(x_line,y_line,color='black')
        y_line += a                          
        ax.plot(x_line,y_line,color='red')
        y_line -= 2 * a                      
        ax.plot(x_line,y_line,color='red')
    else:
        func = np.poly1d(np.flip(posterior_mean.flatten()))
        y_line = func(x_line)
        y_line_add_var = func(x_line)
        y_line_sub_var = func(x_line)

        for i in range(len(x_line)):
            x_nparray=np.zeros((1, n))
            for expo in range(n):
                x_nparray[0][expo]=(x_line[i] ** expo) 
            pred_var = float(a + x_nparray@posterior_var@np.transpose(x_nparray))
            y_line_add_var[i] += pred_var
            y_line_sub_var[i] -= pred_var

        ax.scatter(data_list[0], data_list[1],s=4)
        ax.plot(x_line,y_line,color='black')
        ax.plot(x_line,y_line_add_var,color='red')
        ax.plot(x_line,y_line_sub_var,color='red')
        
    plt.draw()
    return

def perform_Baysian_linear_regression(n,a,w,b):
    '''
    initial prior  ~ N(0,b(^-1)I)
    '''
    prior_mean = 0
    count = 1
    data_list = [[],[]]#[0]: new_x; [1]: new_y

    prior_var = (1/b)*np.eye(n)
    x_value,y_value = get_poly_basis_linear_data(n,a,w)
    data_list[0].append(x_value)
    data_list[1].append(y_value)
    x_nparray=np.zeros((1, n))
    for expo in range(n):
        x_nparray[0][expo]=(x_value ** expo)
    posterior_precision = b*np.eye(n)+(1/a)*np.transpose(x_nparray)@x_nparray
    posterior_var = np.linalg.inv(posterior_precision)
    posterior_mean = (1/a)*posterior_var@np.transpose(x_nparray)*y_value
    
    predicitive_mean = float(np.transpose(posterior_mean)@np.transpose(x_nparray))
    predicitive_var = float(a + x_nparray@posterior_var@np.transpose(x_nparray))
    print("Add data point (",x_value,", ",y_value,"):\n")
    print("posterior mean: \n",posterior_mean,"\n")
    print("posterior variance: \n",posterior_var,"\n")
    print("Predictive distribution ~ N(",predicitive_mean,", ",predicitive_var,")\n")
    
    fig,ax = plt.subplots(2,2,figsize=(7,7))
    draw_fig(ax[0][0],data_list,a,w,n,predicitive_mean,predicitive_var,"Ground Truth")

    while((abs(np.sum(prior_mean-posterior_mean))>1e-6 or abs(np.sum(prior_var-posterior_var))>1e-6)):
        count += 1
        print("count: ", count)
        prior_mean = posterior_mean.copy()
        prior_var = posterior_var.copy()
        prior_precision = np.linalg.inv(prior_var)

        x_value,y_value = get_poly_basis_linear_data(n,a,w)
        data_list[0].append(x_value)
        data_list[1].append(y_value)
        x_nparray=np.zeros((1, n))
        for expo in range(n):
            x_nparray[0][expo] = (x_value ** expo)
        posterior_precision = prior_precision+(1/a)*np.transpose(x_nparray)@x_nparray
        posterior_var = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_var@(prior_precision@prior_mean + (1/a)*np.transpose(x_nparray)*y_value)
        
        predicitive_mean = float(np.transpose(posterior_mean)@np.transpose(x_nparray))
        predicitive_var = float(a + x_nparray@posterior_var@np.transpose(x_nparray))
        print("Add data point (",x_value,", ",y_value,"):\n")
        print("posterior mean: \n",posterior_mean,"\n")
        print("posterior variance: \n",posterior_var,"\n")
        print("Predictive distribution ~ N(",predicitive_mean,", ",predicitive_var,")\n")
        
        if count == 10:
            draw_fig(ax[1][0],data_list,a,w,n,posterior_mean,posterior_var,"After 10 incomes")
        elif count == 50:
            draw_fig(ax[1][1],data_list,a,w,n,posterior_mean,posterior_var,"After 50 incomes")
        else:
            continue

    draw_fig(ax[0][1],data_list,a,w,n,posterior_mean,posterior_var,"Predict Result")
    fig.tight_layout()
    plt.show()
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#03')
    parser.add_argument('--n', type=int, default=2, help="basis number")
    parser.add_argument('--a', type=float, default=1.0, help="e ~ N(0,a)")
    parser.add_argument('--w', nargs='+', type=float, help='W is a n*1 vector', required=True)
    parser.add_argument('--b', type=float, default=1, help="The precision for initial prior  ~ N(0,b(^-1)I)")


    setting = parser.parse_args()
    perform_Baysian_linear_regression(setting.n,setting.a,setting.w,setting.b)