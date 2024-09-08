import argparse
import numpy as np
import matplotlib.pyplot as plt

def GJ(A,y): ## Ax=y, calculate x by Gauss-Jordan elimination
    m = A.shape[0]
    E = np.eye(m)
 
    L = np.eye(m)
    U = A.copy()

    # LU decomposition but also getting L inverse
    for pivot in range(m): 
        for row in range(pivot+1,m):
            scalar = -U[row][pivot]/U[pivot][pivot]
            U[row] += scalar*U[pivot]
            L[row,pivot] = -scalar
            E_current=np.eye(m)
            E_current[row,pivot] = scalar
            E = E_current@E

    c = E@y # Ux=c
    x = []
    
    #back substitution to get x_{m-1},x_{m-2}...x0
    for row in range(m-1,-1,-1): 
        for index in range(len(x)):
            c[row]-=x[index]*U[row][m-1-index]        
        x.append(c[row]/U[row][row])
    
    # return x0,x1,...x_{m-1}
    return np.array(x[::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#01')
    parser.add_argument('--LSElambda', type=int, default=0, help="lamdba for LSE case")
    parser.add_argument('--bases', type=int, default=2, help="number of bases")
    parser.add_argument('--file_path', type=str, default="testfile.txt", help="Training data path")
    
    setting = parser.parse_args()

    bases = setting.bases
    LSElambda = setting.LSElambda

    a_list=[]
    data_x=[]
    b=[]#data_y
    file = open(setting.file_path, 'r')

    for line in file.readlines():
        if line != "\n":
            x,y = line.rstrip().split(',') 
            for exp in range(bases):
                a_list.append(float(x) ** exp)
            b.append(float(y))
            data_x.append(float(x))
    data_size = len(b)

    ### a. For closed-form LSE approach
    A = np.reshape(a_list, (data_size, bases))
    I = np.identity(bases)
    AT = np.transpose(A)
    ATA = AT@A
    ATA_LSElambdaI = ATA + LSElambda*I 
    inv_ATA_LSElambdaI = GJ(ATA_LSElambdaI,I)
    x = inv_ATA_LSElambdaI@AT@b

    LSEerror = 0
    error = np.transpose(A@x-b)@(A@x-b)
   
    LSE_fitting_line_str = "Fitting line: "
    len_x = len(x)
    for i in range(len_x-1,0,-1):
        LSE_fitting_line_str+=str(x[i])+"x^"+str(i)+" + "
    LSE_fitting_line_str+=str(x[0])

    print("LSE:")
    print(LSE_fitting_line_str)
    print("Total error: ",error)
    print("")

    ###LSE line plot
    LSE_x = np.linspace(-6,6,100)
    LSE_x_list=[]
    for num in range(100): 
        for exp in range(bases):
            LSE_x_list.append(LSE_x[num] ** exp)
    LSE_x_matrix = np.reshape(LSE_x_list, (100, bases))
    LSE_y = LSE_x_matrix@x
    
    ### b. For steepest descent method
    SD_x = [0]*bases # for x^0, x^1,...
    lr=1e-4
    
    for epoch in range(500):
        b_pred = A@SD_x
        
        gradient=[0]*bases
        sign_SD_x = np.sign(SD_x)
        for index in range(data_size):
            for expo in range(bases):
                gradient[expo]+=((b[index]-b_pred[index])*(-2)*(data_x[index]**expo))
        for expo in range(bases):
            SD_x[expo] -= lr*(gradient[expo] +LSElambda*sign_SD_x[expo])

    
    SD_error = np.transpose(A@SD_x-b)@(A@SD_x-b)

    SD_fitting_line_str = "Fitting line: "
    len_SD_x = len(SD_x)
    for i in range(len_SD_x-1,0,-1):
        SD_fitting_line_str+=str(SD_x[i])+"x^"+str(i)+" + "
    SD_fitting_line_str+=str(SD_x[0])
    print("Steepest descent method:")
    print(SD_fitting_line_str)
    print("Total error: ",SD_error)
    print("")

    ###SD line plot
    SD_line_x = np.linspace(-6,6,100)
    SD_x_list=[]
    for num in range(100): 
        for exp in range(bases):
            SD_x_list.append(SD_line_x[num] ** exp)
    SD_x_matrix = np.reshape(SD_x_list, (100, bases))
    
    SD_y = SD_x_matrix@SD_x


    # Newton's method
    N_x = [0]*bases # for x^0, x^1,...
    Hessian_matrix = 2*ATA
    for epoch in range(500):
        b_pred = A@N_x
        
        gradient=[0]*bases
        for index in range(data_size):
            for expo in range(bases):
                gradient[expo]+=((b[index]-b_pred[index])*(-2)*(data_x[index]**expo))
        
        inv_Hessian_matrix = GJ(Hessian_matrix,np.eye(bases))
        N_x -= inv_Hessian_matrix@(gradient)
    
    N_error = np.transpose(A@N_x-b)@(A@N_x-b)

    N_fitting_line_str = "Fitting line: "
    len_N_x = len(N_x)
    for i in range(len_N_x-1,0,-1):
        N_fitting_line_str+=str(N_x[i])+"x^"+str(i)+" + "
    N_fitting_line_str+=str(N_x[0])
    print("Newton's method:")
    print(N_fitting_line_str)
    print("Total error: ",N_error)

    ###Newton's method line plot
    N_line_x = np.linspace(-6,6,100)
    N_x_list=[]
    for num in range(100): 
        for exp in range(bases):
            N_x_list.append(N_line_x[num] ** exp)
    N_x_matrix = np.reshape(N_x_list, (100, bases))
    
    N_y = N_x_matrix@N_x  

    ###plt
    fig,ax = plt.subplots(3,1,figsize=(7,6))
    ax[0].scatter(np.transpose(data_x),np.transpose(b), color='red')
    ax[0].plot(np.transpose(LSE_x_matrix)[1],LSE_y, color='black')
    ax[0].text(-6,90,LSE_fitting_line_str, size=8)
    ax[0].set_title("Closed-form LSE approach")

    ax[1].scatter(np.transpose(data_x),np.transpose(b), color='red')
    ax[1].plot(np.transpose(SD_x_matrix)[1],SD_y, color='black')
    ax[1].text(-6,90,SD_fitting_line_str, size=8)
    ax[1].set_title("Steepest descent method")

    ax[2].scatter(np.transpose(data_x),np.transpose(b), color='red')
    ax[2].plot(np.transpose(N_x_matrix)[1],N_y, color='black')
    ax[2].text(-6,90,N_fitting_line_str, size=8)
    ax[2].set_title("Newton's method")

    fig.tight_layout()
    plt.show()    
