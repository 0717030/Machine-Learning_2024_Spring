import argparse
import numpy as np

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
    x2=[]
    while x_value == -1:
        x_value = np.random.uniform(-1,1)
    
    for expo in range(n):
        x[expo][0]=(x_value ** expo)
       
    y = w@x + e
    
    return x_value,float(y)

def perform_sequential_estimate(m,s):
    print("Data point source function: N(",m,", ",s,")")
    print("")
    current_mean = m
    current_var = s
    therehold = 2.5*1e-3
    
    data_sum = 0
    data_square_sum = 0

    count = 1.0
    new_data = get_univar_gaussian_data(m,s)
    data_sum += new_data
    data_square_sum += new_data **2
    new_mean = data_sum/count
    new_var = data_square_sum/count - new_mean*new_mean
    print("Add data point: ",new_data)
    print("Mean = ",new_mean," Variance = ",new_var)

    while(((abs(new_mean-current_mean)>therehold) or abs(new_var-current_var)>therehold)):
        count += 1.0
        current_mean = new_mean
        current_var = new_var

        new_data = get_univar_gaussian_data(m,s)
        data_sum += new_data
        data_square_sum += new_data **2
        new_mean = data_sum/count
        new_var = data_square_sum/count - new_mean*new_mean
        print("Add data point: ",new_data)
        print("Mean = ",new_mean," Variance = ",new_var)
        
    print("count = ",count)
    print("therehold = ",therehold)
    print("new_mean-current_mean = ",abs(new_mean-current_mean))
    print("new_var-current_var = ",abs(new_var-current_var))
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#03')
    parser.add_argument('--m', type=float, default=0.0, help="Expectation value or mean")
    parser.add_argument('--s', type=float, default=1.0, help="Variance")
    

    setting = parser.parse_args()
    
    perform_sequential_estimate(setting.m, setting.s)
