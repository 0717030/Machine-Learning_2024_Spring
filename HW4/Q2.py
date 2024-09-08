import argparse
import numpy as np
import numba as nb
import time

def loadData(imgPath,labelPath):
    imgFile = open(imgPath,"rb")
    labelFile = open(labelPath,"rb")

    parseInfo=[] #[magic, number, row, col]
    for i in range(4):# get [magic, number, row, col] of image file
        parseInfo.append(int.from_bytes(imgFile.read(4), byteorder='big'))
    for i in range(2):# delete magic, number value of label file
        labelFile.read(4)
    
    imgNum = parseInfo[1]
    imgSize = parseInfo[2]*parseInfo[3]
    data = np.zeros((imgNum, imgSize))
    label = np.zeros((imgNum), dtype=int)

    for count in range(imgNum):#image total number
        label_digit = int.from_bytes(labelFile.read(1), byteorder='big')
        label[count] = label_digit
        for pixel in range(imgSize):
            grayscale_value = int.from_bytes(imgFile.read(1), byteorder='big')
            data[count][pixel] = grayscale_value//128
    
    imgFile.close()
    labelFile.close()
    return imgNum,imgSize,label,data

@nb.njit()
def get_initial_guess(imgSize):
    '''
    imgSize: image size = 28*28 * 784
    k: initial guess of k, (10,)
    p: initial guess of p, (10*784)
    '''
    k = np.zeros((10))+0.1
    p = np.random.rand(10,imgSize)
    return k,p

@nb.njit()
def perform_E_step(imgNum,imgSize,k,p,data):
    w = np.zeros((imgNum,10))
    for imgIndex in range(imgNum):
        for category in range(10):
            w[imgIndex][category] = np.log(k[category])
            for pixelIndex in range(imgSize):
                #w[imgIndex][category] += np.log((p[category][pixelIndex]+1e-8)**data[imgIndex][pixelIndex])+np.log((1-p[category][pixelIndex]+1e-8)**(1-data[imgIndex][pixelIndex]))
                if data[imgIndex][pixelIndex] == 1.0:
                    w[imgIndex][category] += np.log(p[category][pixelIndex]+1e-8)
                else:
                    w[imgIndex][category] += np.log(1-p[category][pixelIndex]+1e-8)
        w[imgIndex] = np.exp(w[imgIndex] - max(w[imgIndex]))
        marginal = np.sum(w[imgIndex])
        
        if marginal == 0:
            marginal = 1
        w[imgIndex] = w[imgIndex]/marginal
        
    return w

@nb.njit()
def perform_M_step(imgNum,imgSize,w,k,p,data):
    w_col_sum = np.sum(w,axis=0)
    k_new = w_col_sum/imgNum

    p_new = w.transpose()@data
    for category in range(10):
        p_new[category] = p_new[category]/w_col_sum[category]
    return k_new,p_new

def print_img(p,row,col,label_flag,mapping_matrix):
    string="class : "
    mapping = np.arange(10).reshape(10,)
    if label_flag is True:
        string = "labeled "+string
        mapping = mapping_matrix.copy()
    for category in range(10):
        print(string,category)
        for row_idx in range(row):
            for col_idx in range(col):
                if p[mapping[category]][row_idx*row+col_idx] >= 0.5:
                    print("1",end=" ")
                if p[mapping[category]][row_idx*row+col_idx] < 0.5:
                    print("0", end=" ")
            print("")
        print("")
    return

def EM_steps(imgNum,imgSize,data):
    '''
    imgNum: total numbers of images = 60000
    data: result of binning the gray level value into two bins of every images, (60000,784)
    k: probability of  choosing 0~9, (10,)
    p: probability of each pixel(0~28*28-1) for each catigories(0~9), (10,784)
    count: number of iterations to converge or run out of loop
    '''
    k,p = get_initial_guess(imgSize)
    
    count = 1
    w = perform_E_step(imgNum,imgSize,k,p,data)
    k_new,p_new = perform_M_step(imgNum,imgSize,w,k,p,data)
    print_img(p_new,28,28,False,None)
    dp = np.linalg.norm(p_new-p)
    print("No. of Iteration: ",count, "Difference: ",dp)
    print("-"*50)
    
    while dp>1e-2 and count<20:
        #print("count: ", count, " dp: ",dp)
        count += 1
        k = k_new
        p = p_new

        w = perform_E_step(imgNum,imgSize,k,p,data)
        k_new,p_new = perform_M_step(imgNum,imgSize,w,k,p,data)
        dp = np.linalg.norm(p_new-p)

        print_img(p_new,28,28,False,None)
        print("No. of Iteration: ",count, "Difference: ",dp)
        print("-"*50)
        
    
    return count, k, p

@nb.njit()
def count_prediction_label(imgNum,imgSize,k,p,data,label,count):
    w = perform_E_step(imgNum,imgSize,k,p,data)
    prediction_label_counting_matrix = np.zeros((10,10),dtype=np.int32)#same row, same label
    for img_idx in range(imgNum):
        prediction_label_counting_matrix[label[img_idx]][np.argmax(w[img_idx])]+=1

    
    #testing
    print("prediction_label_counting_matrix:")
    for i in range(10):
        print(prediction_label_counting_matrix[i])
    print("")
    return prediction_label_counting_matrix

def map_prediction_label(prediction_label_counting_matrix):
    '''
    prediction_label_counting_matrix: count the real label for each prediction, (10,10)
    counting_ matrix: copy of prediction_label_counting_matrix, (10,10)
    mapping_matrix: map each label to its predicion, [label]: prediction, (10,)
    '''
    mapping_matrix = np.zeros((10),dtype=int)-1 #[label]: prediction
    counting_matrix = prediction_label_counting_matrix.copy()
    for category in range(10):
        ind = np.unravel_index(np.argmax(counting_matrix, axis=None), counting_matrix.shape) #ind: tuple
        mapping_matrix[ind[0]] = ind[1]
        counting_matrix[ind[0],:] = -1
        counting_matrix[:,ind[1]] = -1
        
    print("mapping_matrix:")
    print(mapping_matrix,"\n","-"*40,end="\n\n")
    return mapping_matrix

#@nb.njit()
def calculate_confusion_matrix(imgNum,prediction_label_counting_matrix,mapping_matrix):
    confusion_matrix = np.zeros((10,2,2),dtype=np.int32) #(TP,FN)(FP,TN)
    correctPredictionSum = 0
    for category in range(10):
        isCategory = np.sum(prediction_label_counting_matrix[category,:])
        predictCategory = np.sum(prediction_label_counting_matrix[:,mapping_matrix[category]])
        correctPrediction = prediction_label_counting_matrix[category][mapping_matrix[category]]
        confusion_matrix[category][0][0] = correctPrediction                  #TP
        confusion_matrix[category][0][1] = isCategory-correctPrediction       #FN
        confusion_matrix[category][1][0] = predictCategory-correctPrediction  #FP
        confusion_matrix[category][1][1] = imgNum-isCategory-predictCategory+correctPrediction #TN
        correctPredictionSum += correctPrediction
    return confusion_matrix,correctPredictionSum

def print_confusion_matrix(count,imgNum,confusion_matrix,correctPredictionSum):
    for category in range(10):
        print("Confusion Matrix ",category,":")
        print("\t"*2,"Predict number ",category," Predict not number ",category)
        print("Is number ",category,"\t   ",confusion_matrix[category][0][0],"\t"*2,confusion_matrix[category][0][1])
        print("Isn't number ",category," "*3,confusion_matrix[category][1][0],"\t"*2,confusion_matrix[category][1][1],end="\n\n")
        print("Sensitivity (Successfully predict number ",category,")\t :",(confusion_matrix[category][0][0]+0.0)/(confusion_matrix[category][0][0]+confusion_matrix[category][0][1]))
        print("Specificity (Successfully predict not number ",category,"):",(confusion_matrix[category][1][1]+0.0)/(confusion_matrix[category][1][0]+confusion_matrix[category][1][1]),end="\n\n")
        print("-"*40,end="\n\n")

    print("Total iteration to converge: ",count)
    print("Total error rate: ",1-((correctPredictionSum+0.0)/imgNum),end="\n\n")
    return

def perform_EM_algorithms(trainImg,trainLabel):
    print("loading data")
    imgNum, imgSize, label, data = loadData(trainImg,trainLabel)
    
    print("EM algorithm")
    count, k, p = EM_steps(imgNum,imgSize,data)

    print("prediction")
    prediction_label_counting_matrix = count_prediction_label(imgNum,imgSize,k,p,data,label,count)
    mapping_matrix = map_prediction_label(prediction_label_counting_matrix)
    print_img(p,28,28,True,mapping_matrix)
    confusion_matrix,correctPredictionSum = calculate_confusion_matrix(imgNum,prediction_label_counting_matrix,mapping_matrix)
    print_confusion_matrix(count,imgNum,confusion_matrix,correctPredictionSum)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#04')
    parser.add_argument('--trainImg', type=str, default="train-images.idx3-ubyte_", help="training image file path")
    parser.add_argument('--trainLabel', type=str, default="train-labels.idx1-ubyte_", help="training label file path")
    setting = parser.parse_args()

    start = time.time()
    perform_EM_algorithms(setting.trainImg,setting.trainLabel)
    end = time.time()
    print("The time of execution is :","{:.2f}".format((end-start)), "s")
