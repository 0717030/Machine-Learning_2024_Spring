import argparse
import numpy as np
from libsvm.svmutil import *
import time

kernel = ["linear","polynomial","RBF"]

def compare_svm_accuracy(problem,testImg,testLabel): 
    for idx in range(len(kernel)):
        start = time.time()
        print(kernel[idx]+" kernel function")
        model = svm_train(problem, f'-t {idx} -q')
        result = svm_predict(testLabel, testImg, model)
        end = time.time()
        print("The time of execution is :","{:.2f}".format((end-start)), "s\n")
    return

def load_data(trainImg_path,trainLabel_path,testImg_path,testLabel_path):
    trainImg = np.loadtxt(trainImg_path, delimiter=",")
    trainLabel = np.loadtxt(trainLabel_path)
    testImg = np.loadtxt(testImg_path, delimiter=",")
    testLabel = np.loadtxt(testLabel_path)
    
    return trainImg,trainLabel,testImg,testLabel

def grid_search_kernel_parameters(problem,testImg,testLabel):
    best_acc_list = [0.0,0.0,0.0]
    opt_par = {'c': 0.0, 'g': 0.0, 'd': 0.0, 'r': 0.0 }
   
    for idx in range(len(kernel)):
        if idx == 0:
            start = time.time()
            cost_list = list(map(lambda n: 2**n, range(-10,11,2)))

            for cost in cost_list:
                model = svm_train(problem, f"-t {idx} -c {cost} -v 3 -q")
                
                if best_acc_list[idx]<model:
                    best_acc_list[idx]=model
                    opt_par['c']=cost

            print("\n"+kernel[idx]+" kernel function")            
            print("optimized parameters: "+f"-t {idx} -c {opt_par['c']} -q")
            model = svm_train(problem, f"-t {idx} -c {opt_par['c']} -q")
            result = svm_predict(testLabel, testImg, model)
        elif idx == 1:
            start = time.time()
            print(kernel[idx]+" kernel function")
            cost_list = list(map(lambda n: 2**n, range(-2,5,2)))
            degree_list = list(range(2,6))
            gamma_list = list(map(lambda n: 2**n, range(-4,3,2)))
            coef_list = list(map(lambda n: 2**n, range(-2,3,2)))

            for cost in cost_list:
                for d in degree_list:
                    for g in gamma_list:
                        for coef in coef_list:
                            model = svm_train(problem, f"-t {idx} -c {cost} -d {d} -g {g} -r {coef} -v 3 -q")
                            if best_acc_list[idx]<model:
                                best_acc_list[idx]=model
                                opt_par['c']=cost
                                opt_par['d']=d
                                opt_par['g']=g
                                opt_par['r']=coef
            
            print("\n"+kernel[idx]+" kernel function")            
            print("optimized parameters: "+f"-t {idx} -c {opt_par['c']} -d {opt_par['d']} -g {opt_par['g']} -r {opt_par['r']} -q")
            model = svm_train(problem, f"-t {idx} -c {opt_par['c']} -d {opt_par['d']} -g {opt_par['g']} -r {opt_par['r']} -q")
            result = svm_predict(testLabel, testImg, model)
        elif idx == 2:
            start = time.time()
            print(kernel[idx]+" kernel function")
            cost_list = list(map(lambda n: 2**n, range(-4,5,2)))
            gamma_list = list(map(lambda n: 2**n, range(-4,5,2)))

            for cost in cost_list:
                for g in gamma_list:
                    model = svm_train(problem, f"-t {idx} -c {cost} -g {g} -v 3 -q")
                    if best_acc_list[idx]<model:
                        best_acc_list[idx]=model
                        opt_par['c']=cost
                        opt_par['g']=g
                        
            print("\n"+kernel[idx]+" kernel function")            
            print("optimized parameters: "+f"-t {idx} -c {opt_par['c']} -g {opt_par['g']} -q")
            model = svm_train(problem, f"-t {idx} -c {opt_par['c']} -g {opt_par['g']} -q")
            result = svm_predict(testLabel, testImg, model)
        end = time.time()
        print("The time of execution is :","{:.2f}".format((end-start)), "s\n")
    
    return

def get_linear_kernel(x1,x2):
    return np.dot(x1,x2.T)

def get_RBF_kernel(x1,x2,gamma):
    dis = np.sum(x1*x1, axis=1).reshape(-1, 1)+np.sum(x2*x2, axis=1)-2*x1@x2.T
    return np.exp(-gamma*dis)

def get_precomputed_kernel(x1,x2,gamma):
    linear_kernel = get_linear_kernel(x1,x2)
    RBF_kernel = get_RBF_kernel(x1,x2,gamma)
    precomputed_kernel = np.hstack((np.arange(1, x1.shape[0]+1).reshape((-1, 1)),linear_kernel+RBF_kernel))
    return precomputed_kernel

def perform_user_defined_kernel_svm(trainImg,trainLabel,testImg,testLabel,gamma):
    start = time.time()
    pre_kernel_train = get_precomputed_kernel(trainImg,trainImg,gamma)
    problem = svm_problem(trainLabel, pre_kernel_train,isKernel=True)
    model = svm_train(problem, f"-t 4 -q")

    pre_kernel_test = get_precomputed_kernel(testImg,trainImg,gamma)
    svm_predict(testLabel,pre_kernel_test,model)
    
    end = time.time()
    print("The time of execution is :","{:.2f}".format((end-start)), "s\n")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#05')
    parser.add_argument('--trainImg', type=str, default="X_train.csv", help="training image file path")
    parser.add_argument('--trainLabel', type=str, default="Y_train.csv", help="training label file path")
    parser.add_argument('--testImg', type=str, default="X_test.csv", help="testing image file path")
    parser.add_argument('--testLabel', type=str, default="Y_test.csv", help="testing label file path")
    parser.add_argument('--mode', type=int, default=1, help="different sub-question")
    setting = parser.parse_args()

    trainImg,trainLabel,testImg,testLabel =  load_data(setting.trainImg,setting.trainLabel,
                                                       setting.testImg,setting.testLabel)
    problem = svm_problem(trainLabel, trainImg)

    if setting.mode == 0:
        compare_svm_accuracy(problem,testImg,testLabel)
    elif setting.mode == 1:
        grid_search_kernel_parameters(problem,testImg,testLabel)
    elif setting.mode == 2:
        perform_user_defined_kernel_svm(trainImg,trainLabel,testImg,testLabel,1.0/trainImg.shape[1])
    else:
        print("Invalid mode input. Please input in range of 0~2")