import numpy as np
import os
import matplotlib.pyplot as plt
import argparse 
from PIL import Image
from scipy.spatial.distance import cdist


def load_image(database,mode,rows,cols):
    folder_path = database+mode    
    file_list = os.listdir(folder_path)
    file_count = len(file_list)
    image_array = np.zeros((file_count,rows*cols))
    label_array = np.zeros(file_count, dtype=int)

    for idx, filename in enumerate(file_list):
        if filename.endswith('.pgm'):
            file_path = os.path.join(folder_path, filename)
            img = np.asarray(Image.open(file_path).resize((cols,rows))).flatten()
            image_array[idx,:] = img
            label_array[idx] = int(filename[7:9])

   
    return image_array,label_array

def calculate_kernel(mode,image,rows,cols,gamma,degree):
    if mode == 0: # without kernel
        kernel = np.cov(image.T,bias=True)        
    elif mode == 1:  # linear kernel
        kernel =  image.T.dot(image)
    elif mode == 2: # polynomial kernel
        kernel =  np.power((image.T.dot(image)+gamma),degree)
    else: # RBF kernel
        kernel =  np.exp(-gamma*cdist(image.T,image.T,'seuclidean'))
    
    return kernel

def get_largest_eigenvectors(kernel):
    eigenvalues, eigenvectors = np.linalg.eig(kernel)
    largest_idx = np.argsort(-eigenvalues)[:25]
    largest_eigenvectors = eigenvectors[:, largest_idx].real
    return largest_eigenvectors

    

def show_eigenface_fisherface(title,largest_eigenvectors,facetype,rows,cols):
    plt.figure()
    plt.suptitle(facetype)
    for idx in range(25):
        plt.subplot(5,5,idx+1)
        plt.axis('off')
        plt.imshow(largest_eigenvectors.T[idx,:].reshape((rows,cols)),cmap='gray')
    
    plt.savefig(title+facetype+".png")
    return

def reconstruct_face(train_image,rows,cols,projection):
    reconstructions = np.zeros((10,rows*cols))
    choices = np.random.choice(train_image.shape[0],10)
    for idx in range(10):
        reconstructions[idx,:] = train_image[choices[idx],:]@(projection)@(projection.T)
     
    return choices, reconstructions

def show_reconstruction_face(title,train_image,choices,reconstructions,rows, cols):
    fig = plt.figure()
    fig.suptitle('Original faces & Reconstructed faces')
    for idx in range(10):
        plt.subplot(2,10,idx*2+1)
        plt.axis('off')
        plt.imshow(train_image[choices[idx],:].reshape((rows,cols)),plt.cm.gray)

        plt.subplot(2,10,idx*2+2)
        plt.axis('off')
        plt.imshow(reconstructions[idx,:].reshape((rows,cols)),plt.cm.gray)
    
    plt.savefig(title+" reconstruciton.png")
    return

def perform_knn_prediction(low_dim_train,train_label,low_dim_test,k):
    train_num = low_dim_train.shape[0]
    test_num = low_dim_test.shape[0]
    prediction = np.zeros(test_num,dtype=int)

    for test_idx in range(test_num):
        dist = np.zeros(train_num)
        for train_idx in range(train_num):
            dist[train_idx] = np.linalg.norm(low_dim_test[test_idx]-low_dim_train[train_idx])
        k_nearest = train_label[np.argsort(dist)[:k]]
        prediction[test_idx] = np.argmax(np.bincount(k_nearest))
        
    return prediction

def calculate_performance(title, prediction,test_label):
    error = 0.0
    test_num = test_label.shape[0]
    for idx in range(test_num):
        if prediction[idx] != test_label[idx]:
            error+=1
    print(title)
    print("Error rate: ",error/test_num,"(",int(error),"/",test_num,")")
    return

def perform_PCA(kernelmode,train_image,train_label,test_image,test_label,rows,cols,gamma,degree,k):
    # matrix for eigen decompostion
    kernel_list=["linear","polynomial","RBF"]
    title = "PCA"
    if kernelmode != 0:
        title = str(kernel_list[kernelmode-1])+" kernel "+title

    kernel = calculate_kernel(kernelmode,train_image,rows,cols,gamma,degree)
    if kernelmode == 0:
        matrix = kernel
    else:
        one_N = np.ones((rows*cols,rows*cols),dtype=float)/(rows*cols)
        matrix = kernel-one_N.dot(kernel)-kernel.dot(one_N)+one_N.dot(kernel).dot(one_N)
    

    # get projection matrix and show reconstruction
    largest_eigenvectors = get_largest_eigenvectors(matrix)
    show_eigenface_fisherface(title,largest_eigenvectors,"eigenface",rows,cols)
    choices, reconstructions = reconstruct_face(train_image,rows,cols,largest_eigenvectors)
    show_reconstruction_face(title,train_image,choices,reconstructions,rows, cols)

    # project data to low-d space
    low_dim_train = train_image@largest_eigenvectors
    low_dim_test = test_image@largest_eigenvectors

    # face recognition
    prediction = perform_knn_prediction(low_dim_train,train_label,low_dim_test,k)   
    calculate_performance(title, prediction,test_label)
    
    return

def get_LDA_matrix(kernelmode,train_image,train_label,test_image,test_label,rows,cols,gamma,degree,k):
    labels, repeats = np.unique(train_label,return_counts=True)
    subject_num = len(labels)

    kernel = None
    if kernelmode == 0:
        kernel = train_image
    else:
        kernel = calculate_kernel(kernelmode,train_image,rows,cols,gamma,degree)

    train_num = train_image.shape[0]
    pixel_num = kernel.shape[1]
    
    # get mean
    total_mean = np.mean(kernel,axis=0)#.reshape(-1, 1)
    subject_mean = np.zeros((subject_num, pixel_num))

    for i in range(train_num):
        for p in range(pixel_num):
            subject_mean[train_label[i]-1,p] += kernel[i,p]
    for idx in range(subject_num):
        subject_mean[idx] /= repeats[idx]

    # calculate similarities
    s_within = np.zeros((pixel_num,pixel_num))
    s_between = np.zeros((pixel_num,pixel_num))

    for idx in range(train_num):
        distance = np.array(kernel[idx]-subject_mean[train_label[idx]-1]).reshape(-1,1)
        s_within += distance@distance.T
        
    for idx in range(subject_num):
        distance = np.array(subject_mean[idx]-total_mean[idx]).reshape(-1,1)
        s_between += repeats[idx]*(distance@distance.T)
    
    return np.linalg.pinv(s_within)@s_between

def get_kernel_LDA_matrix(kernelmode,train_image,train_label,test_image,test_label,rows,cols,gamma,degree,k):
    labels, repeats = np.unique(train_label,return_counts=True)
    subject_num = len(labels)
    
    # calculate kernel results
    kernel = calculate_kernel(kernelmode,train_image,rows,cols,gamma,degree)

    pixel_num = kernel.shape[1]

    subject_kernel = np.zeros((subject_num,pixel_num,pixel_num))
    for idx in range(subject_num):
        subject_image = train_image[train_label==idx+1]
        subject_kernel[idx] = calculate_kernel(kernelmode,subject_image,rows,cols,gamma,degree)
    
    
    # get mean
    total_mean = np.mean(kernel,axis=0)#.reshape(-1, 1)
    subject_mean = np.zeros((subject_num, pixel_num))

    for idx in range(subject_num):
        subject_mean[idx] = np.sum(subject_kernel[idx]).reshape(-1, 1)/subject_num
    
    # calculate similarities
    M = np.zeros((pixel_num,pixel_num))
    N = np.zeros((pixel_num,pixel_num))

    for idx in range(subject_num):
        distance = repeats[idx]*np.array(subject_mean[idx]-total_mean).reshape(-1,1)
        M += distance.dot(distance.T)
        
    idenity = np.eye(pixel_num)
    for idx in range(subject_num):
        one_li = np.ones((pixel_num,pixel_num))/repeats[idx]
        N += subject_kernel[idx].dot(idenity-one_li).dot(subject_kernel[idx].T)
        
    return np.linalg.pinv(N)@M



def perform_LDA(kernelmode,train_image,train_label,test_image,test_label,rows,cols,gamma,degree,k):
    kernel_list=["linear","polynomial","RBF"]
    title = "LDA"

    if kernelmode != 0:
        title = str(kernel_list[kernelmode-1])+" kernel "+title

    if kernelmode ==0:
        matrix = get_LDA_matrix(kernelmode,train_image,train_label,test_image,test_label,rows,cols,gamma,degree,k)
    else:
        matrix = get_kernel_LDA_matrix(kernelmode,train_image,train_label,test_image,test_label,rows,cols,gamma,degree,k)
    
    # get projection matrix and show reconstruction
    largest_eigenvectors = get_largest_eigenvectors(matrix)
    show_eigenface_fisherface(title, largest_eigenvectors,"fisherface",rows,cols)
    choices, reconstructions = reconstruct_face(train_image,rows,cols,largest_eigenvectors)
    show_reconstruction_face(title, train_image,choices,reconstructions,rows, cols)

    # project data to low-d space
    low_dim_train = train_image@largest_eigenvectors
    low_dim_test = test_image@largest_eigenvectors

    # face recognition
    prediction = perform_knn_prediction(low_dim_train,train_label,low_dim_test,k)
    calculate_performance(title,prediction,test_label)
    print("-"*50)
    
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#07')
    parser.add_argument('--dbPath', type=str, default="Yale_Face_Database/", help="database file path")
    parser.add_argument('--rows', type=int, default=41, help="resized row size")
    parser.add_argument('--cols', type=int, default=29, help="resized column size ")
    parser.add_argument('--gamma', type=float, default=1e-8, help="gamma hyperparameter for RBF kernel or linear kernel")
    parser.add_argument('--degree', type=int, default=3, help="different degree for linear kernel")
    parser.add_argument('--k', type=int, default=5, help="different k for KNN")
    
    setting = parser.parse_args()
    
    train_image, train_label = load_image(setting.dbPath,"Training",setting.rows,setting.cols)
    test_image, test_label = load_image(setting.dbPath,"Testing",setting.rows,setting.cols)
    for kernelmode in range(4):
        perform_PCA(kernelmode,train_image,train_label,test_image,test_label,setting.rows,setting.cols,setting.gamma,setting.degree,setting.k)
        perform_LDA(kernelmode,train_image,train_label,test_image,test_label,setting.rows,setting.cols,setting.gamma,setting.degree,setting.k)
    

    
    