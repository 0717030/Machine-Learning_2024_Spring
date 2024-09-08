import argparse
import numpy as np
import time

def loadData(imgPath,labelPath):
    imgFile = open(imgPath,"rb")
    labelFile = open(labelPath,"rb")

    parseInfo=[] #[magic, number, row, col]
    for i in range(4):# get [magic, number, row, col] of image file
        parseInfo.append(int.from_bytes(imgFile.read(4), byteorder='big'))
    for i in range(2):# delete magic, number value of label file
        labelFile.read(4)
    
    return imgFile,labelFile,parseInfo[1],parseInfo[2],parseInfo[3]


def getResult(posterior, test_label_digit):
    print("Postirior (in log scale):")
    for digit in range(len(posterior)):
        print(digit, ":", posterior[digit])
    prediction = np.argmin(posterior)    
    print("Prediction: ",prediction,", Ans: ",test_label_digit)
    
    if prediction == test_label_digit:
        return 0 # wrongCount remain
    else:
        return 1


def printDigit(likelihood,row,col,opt):
    for digit in range(likelihood.shape[0]):
        print(digit,":")
        for row_idx in range(row):
            for col_idx in range(col):
                if opt == 0:
                    pixelValue = -np.sum(likelihood[digit][row_idx*row+col_idx][:16])+np.sum(likelihood[digit][row_idx*row+col_idx][16:])
                else:
                    pixelValue = likelihood[digit][row_idx*row+col_idx]-128.0
                if pixelValue > 0:
                    #print("@ ", end ="")
                    print("1 ", end ="")
                else:
                    #print(". ", end ="")
                    print("0 ", end ="")
            print("")
        print("")
    return


def discrete_mode(train_imgFile, tain_labelFile, tain_num, row, col,test_imgFile, test_labelFile, test_num, categorySize):
    imgSize = row*col
    prior = np.zeros(categorySize)
    likelihood = np.zeros((categorySize, imgSize, 32), dtype=int)
    likelihood_sum = np.zeros((categorySize, imgSize), dtype=int)
    wrongCount = 0.0

    for count in range(tain_num):
        label_digit = int.from_bytes(tain_labelFile.read(1), byteorder='big')
        prior[label_digit] += 1
        for pixel in range(imgSize):
            grayscale = int.from_bytes(train_imgFile.read(1), byteorder='big')
            likelihood[label_digit][pixel][grayscale//8] += 1
            
    likelihood_sum = np.sum(likelihood,axis=2)
    prior = prior/tain_num

    for count in range(test_num):
        test_label_digit = int.from_bytes(test_labelFile.read(1), byteorder='big')
        posterior = np.log(prior.copy())
        test_img = np.zeros(imgSize, dtype=int)
        for pixel in range(imgSize):
            test_img[pixel] = int.from_bytes(test_imgFile.read(1), byteorder='big')
        for digit in range(categorySize):
            for pixel in range(imgSize):
                bin_idx = test_img[pixel]//8
                
                current_likelihood = likelihood[digit][pixel][bin_idx]
                if current_likelihood == 0:                    
                    current_likelihood = float(1e-4)
                    
                posterior[digit] += np.log(current_likelihood/likelihood_sum[digit][pixel])
        
        posterior_sum = sum(posterior)
        posterior /= posterior_sum

        wrongCount += getResult(posterior,test_label_digit)
        
    print("Error rate: ",wrongCount/float(test_num))
    print("Imagination of numbers in Bayesian classifier:")
    printDigit(likelihood,row,col,0)
    return



def continuous_mode(train_imgFile, tain_labelFile, tain_num, row, col,test_imgFile, test_labelFile, test_num, categorySize):
    imgSize = row*col
    prior = np.zeros(categorySize)
    grayscale = np.zeros((categorySize, imgSize))
    grayscale_square = np.zeros((categorySize, imgSize))
    wrongCount = 0.0

    for count in range(tain_num):
        label_digit = int.from_bytes(tain_labelFile.read(1), byteorder='big')
        prior[label_digit] += 1
        for pixel in range(imgSize):
            grayscale_value = int.from_bytes(train_imgFile.read(1), byteorder='big')
            grayscale[label_digit][pixel] += grayscale_value
            grayscale_square[label_digit][pixel] += grayscale_value**2            
            
    mean = grayscale.copy()
    var = grayscale_square.copy()
    for digit in range(categorySize):
        mean[digit] = mean[digit]/float(prior[digit])
        var[digit] = var[digit]/float(prior[digit]) - mean[digit]*mean[digit]
    prior = prior/tain_num

    
    for count in range(test_num):
        test_label_digit = int.from_bytes(test_labelFile.read(1), byteorder='big')
        posterior = np.log(prior.copy())
        test_img = np.zeros(imgSize, dtype=int)
        for pixel in range(imgSize):
            test_img[pixel] = int.from_bytes(test_imgFile.read(1), byteorder='big')
        for digit in range(categorySize):
            for pixel in range(imgSize):
                current_mean = mean[digit][pixel]
                current_var = var[digit][pixel]
                if current_var == 0:                    
                    current_var = 100
                    
                posterior[digit] += 0.5*(-np.log(2*np.pi*current_var)+(-(test_img[pixel]-current_mean)**2/current_var))
                
        posterior_sum = sum(posterior)
        posterior /= posterior_sum

        wrongCount += getResult(posterior,test_label_digit)
        
    print("Error rate: ",wrongCount/float(test_num))
    print("Imagination of numbers in Bayesian classifier:")
    printDigit(mean, row, col,1)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#02')
    parser.add_argument('--trainImg', type=str, default="train-images.idx3-ubyte_", help="training image file path")
    parser.add_argument('--trainLabel', type=str, default="train-labels.idx1-ubyte_", help="training label file path")
    parser.add_argument('--testImg', type=str, default="t10k-images.idx3-ubyte_", help="testing image file path")
    parser.add_argument('--testLabel', type=str, default="t10k-labels.idx1-ubyte_", help="testing label file path")
    parser.add_argument('--opt', type=int, default=1, help="discrete mode(0) or continuous mode(1)")

    setting = parser.parse_args()

    train_imgFile, tain_labelFile, tain_num, tain_row, tain_col = loadData(setting.trainImg, setting.trainLabel)
    test_imgFile, test_labelFile, test_num, test_row, test_col = loadData(setting.testImg, setting.testLabel)

    if setting.opt == 0:
        start = time.time()
        discrete_mode(train_imgFile, tain_labelFile, tain_num, tain_row, tain_col,test_imgFile, test_labelFile, test_num, 10)
        end = time.time()
        print("The time of execution of discrete mode is :","{:.2f}".format((end-start)), "s")
    elif setting.opt == 1:
        start = time.time()
        continuous_mode(train_imgFile, tain_labelFile, tain_num, tain_row, tain_col,test_imgFile, test_labelFile, test_num, 10)
        end = time.time()
        print("The time of execution of continuous mode is :","{:.2f}".format((end-start)), "s")
    else:
        print("Please choose the toggle option: discrete mode(0) or continuous mode(1)?")