import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from kernel_k_means import load_img, calculate_kernel,get_centers, get_current_image
import os
import time
import matplotlib.pyplot as plt

def compute_Laplacian_matrix(W,cut):
    D = np.zeros((W.shape))
    L = np.zeros((W.shape))
    D = np.diag(np.sum(W,axis=1))
    L = D-W
    
    if cut:
        for idx in range(D.shape[0]):
            D[idx,idx] = np.power(D[idx,idx],-0.5)        
        L = D.dot(L).dot(D)
        
    return L

def compute_eigenvector_matrix(L,clusters,cut):
    eigenvalues, eigenvectors = np.linalg.eig(L)
    eigenvectors = eigenvectors.T

    sort_idx = eigenvalues.argsort()
    sort_idx = sort_idx[eigenvalues[sort_idx]>0]

    U = eigenvectors[sort_idx[:clusters]].T 
    if cut:
        U_row_sum = np.sum(U,axis=1)
        for r in range(U.shape[0]):
            U[r,:] /= U_row_sum[r]

    return U

def get_clustering_info(filePath,clusters,gs,gc,mode,cut,kernel):
    title = filePath[:-4]+"_clusterNum"+str(clusters)+"_cut_"+str(cut)
    title += "_gs_"+str(gs)+"_gc_"+str(gc)+".npy"
    if os.path.isfile(title)==True:
        print("file found.")
        clustering_info = np.load(title, allow_pickle=True)
        U = clustering_info.item().get('U')        
    else:
        print("file not found.")
        L = compute_Laplacian_matrix(kernel,cut)
        U = compute_eigenvector_matrix(L,clusters,cut)
        clustering_info = {'U' : U}
        np.save(title,clustering_info)

    return U

def get_initial_centers(centers,U):
    center_list = []
    for k in range(len(centers)):
        center_list.append(U[centers[k],:])

    return np.array(center_list)

def get_new_cluster(rows,cols,clusters,U,centers):
    pixels = rows*cols
    cluster_result = np.zeros(pixels, dtype=int)
    
    for idx in range(pixels):
        distance = np.zeros(clusters)
        for k in range(clusters):
            distance[k] = np.linalg.norm(U[idx]-centers[k])
        cluster_result[idx] = np.argmin(distance)
    
    return cluster_result

def get_new_centers(clusters,U,cluster_result):
    center_list = []
    for k in range(clusters):
        category_k = U[cluster_result==k]
        center_list.append(np.average(category_k,axis=0))

    return np.array(center_list)

def visualize_eigenspace(U,result,clusters,title):
    colors = ['r','g','b']
    plt.clf()
    if clusters == 2:
        plt.xlabel("Feature #1")
        plt.ylabel("Feature #2")
        for p in range(U.shape[0]):
            plt.scatter(U[p,0],U[p,1], c=colors[result[p]],s=2)
        plt.savefig(title+"_eigenspace.png")   
    else:
        fig=plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("Feature #1")
        ax.set_ylabel("Feature #2")
        ax.set_zlabel("Feature #3")
        for p in range(U.shape[0]):
            ax.scatter(U[p,0],U[p,1],np.real(U[p,2]), c=colors[result[p]],s=2)
        plt.savefig(title+"_eigenspace.png") 

    return   


def perform_kmeans(rows,cols,clusters,filePath,mode,centers,U,cut,gs,gc):
    max_iter = 50   
    threshold = 1e-2
    img_list = []
    cluster_result = np.zeros(rows*cols, dtype=int)-1

    iter = 0
    while iter < max_iter:
        print(iter)
        iter += 1
        new_cluster_result = get_new_cluster(rows,cols,clusters,U,centers)
        if iter!=1 and (np.linalg.norm((new_cluster_result-cluster_result))<threshold):
            break
        cluster_result = new_cluster_result.copy()
        centers = get_new_centers(clusters,U,cluster_result)
        img_list.append(get_current_image(rows,cols,cluster_result))

    
    title = filePath[:-4]+"_spectral_clusterNum_"+str(clusters)+"_cut_"+str(cut)+"_mode_"+str(mode)
    title += "_gs_"+str(gs)+"_gc_"+str(gc)
    img_list[0].save(title+".gif",save_all=True,append_images=img_list[1:],optimize=False,loop=0,duration=300)
    img_list[-1].save(title+"_result.png")
    if clusters == 2 or clusters == 3:
        visualize_eigenspace(U,cluster_result,clusters,title)
        
    return

if __name__ == '__main__':
    gs=[1e-3,1e-4]
    gc=[1e-3,1e-4]
    img_list=["image1.png","image2.png"]
    for i in range(1,2):
        for clusterNum in range(2,5):
            for gs_idx in range(2):
                for gc_idx in range(2):
                    for m in range(2):
                        for c in range(2):
                            start = time.time()
                            print("image loading...")
                            img_c,img_s,rows,cols,channels = load_img(img_list[i])
                            
                            print("kernel calculation...")
                            kernel = calculate_kernel(img_s,gs[gs_idx],img_c,gc[gc_idx])

                            print("clustering information calculation...")
                            U = get_clustering_info(img_list[i],clusterNum,gs[gs_idx],gc[gc_idx],m,c,kernel)

                            print("spectral clustering...")
                            centers_idx = get_centers(rows,cols,clusterNum,U,m)
                            centers = get_initial_centers(centers_idx,U)
                            perform_kmeans(rows,cols,clusterNum,img_list[i],m,centers,U,c,gs[gs_idx],gc[gc_idx])
                            end = time.time()
                            print("time: ",end-start,"s.")
                            print("="*40,end = '\n\n')
                    
    
    

    