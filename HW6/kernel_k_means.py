import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import time

def load_img(filepath):
    img = Image.open(filepath)
    img = np.array(img)
    rows,cols,channels=img.shape

    img_c = img.copy().reshape((-1,channels))
    img_s = np.zeros((rows*cols,2),dtype=int)
    for r in range(rows):
        for c in range(cols):
            img_s[r*rows+c]=[r,c]
    
    return img_c,img_s,rows,cols,channels

def calculate_kernel(img_s,gs,img_c,gc):
    spacial_dis = cdist(img_s,img_s,'sqeuclidean')
    color_dis = cdist(img_c,img_c,'sqeuclidean')

    return np.exp(-gs*spacial_dis)*np.exp(-gc*color_dis)

def get_centers(rows,cols,clusters,img_s,mode):
    pixels = rows*cols
    if mode == 0:
        return np.random.choice(pixels,clusters)
    elif mode == 1:
        center_list = np.zeros(clusters,dtype=int)
        center_list[0] = np.random.choice(pixels)
        
        for k in range(1,clusters):
            dis = np.zeros(pixels)
            for p in range(pixels):
                min_dis_value = np.Inf
                for k_idx in range(k):
                    current_dis_value = np.linalg.norm(img_s[center_list[k_idx],:]-img_s[p,:])
                    if current_dis_value < min_dis_value:
                        min_dis_value = current_dis_value
                dis[p] = min_dis_value
            dis/=np.sum(dis)
            new_center_idx = np.random.choice(pixels,p=dis)
            center_list[k] = new_center_idx
        print("center list", center_list)               

        return center_list
    else:
        print("No such mode. Perform K-means++ instead")
        return get_centers(rows,cols,clusters,img_s,1)

def get_initial_cluster(mode,clusters,kernel,rows,cols,img_s):
    centers = get_centers(rows,cols,clusters,img_s,mode)
    pixels = rows*cols
    cluster_result = np.zeros(pixels, dtype=int)
    for idx in range(pixels):
        distance = np.zeros(clusters)
        for k in range(clusters):
            distance[k] = kernel[idx,idx]+kernel[centers[k],centers[k]]-2*kernel[idx,centers[k]]
        cluster_result[idx] = np.argmin(distance)
    
    return cluster_result

def get_current_image(rows,cols,cluster_result):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[0,0,0],[255,255,255],[0,255,255],[255,0,255],[255,255,0]])
    
    colored_cluster = np.zeros((rows*cols, 3))
    for idx in range(rows*cols):
        colored_cluster[idx,:] = colors[cluster_result[idx],:]
    img = colored_cluster.reshape(rows,cols,3)

    return Image.fromarray(np.uint8(img))

def get_new_cluster(rows,cols,clusters,cluster_result,kernel):
    pixels = rows*cols
    new_cluster_result = np.zeros(pixels, dtype=int)
    each_cluster_count = np.zeros(clusters, dtype=int)
    last_term = np.zeros(clusters)
    second_term = np.zeros((pixels,clusters))
    
    for k in range(clusters):
        each_cluster_count[k] = (cluster_result==k).sum()
    each_cluster_count[each_cluster_count==0] = 1
    
    for k in range(clusters):
        tmp_kernel = kernel.copy()
        for p in range(pixels):
            if cluster_result[p]!=k:
                tmp_kernel[p,:]=0
                tmp_kernel[:,p]=0
        last_term[k] = np.sum(tmp_kernel)/each_cluster_count[k]**2
    
    for p in range(pixels):
        for k in range(clusters):
            second_term[p,k] += np.sum(kernel[p,:][np.where(cluster_result==k)])
            second_term[p,k] *= 2.0/each_cluster_count[k]

    for p in range(pixels):
        distance = np.zeros(clusters)
        for k in range(clusters):
            distance[k] += kernel[p,p]-second_term[p,k]+last_term[k]

        new_cluster_result[p] = np.argmin(distance)
    
    return new_cluster_result

def perform_kernel_kmeans(rows,cols,kernel,clusters,cluster_result,filePath,mode,gs,gc):
    max_iter = 50
    threshold = 1e-2
    img_list = []
    img_list.append(get_current_image(rows,cols,cluster_result))

    iter = 1
    while iter < max_iter:
        print(iter)
        iter += 1
        new_cluster_result = get_new_cluster(rows,cols,clusters,cluster_result,kernel)
        if(np.linalg.norm((new_cluster_result-cluster_result))<threshold):
            break
        cluster_result = new_cluster_result.copy()
        img_list.append(get_current_image(rows,cols,cluster_result))

    title = filePath[:-4]+"_k_means_clusterNum_"+str(clusters)+"_mode_"+str(mode)
    title += "_gs_"+str(gs)+"_gc_"+str(gc)

    img_list[0].save(title+".gif",save_all=True,append_images=img_list[1:],optimize=False,loop=0,duration=300)
    img_list[-1].save(title+"_result.png")
    return

if __name__ == '__main__':
    gs=[1e-3,1e-4]
    gc=[1e-3,1e-4]
    img_list=["image1.png","image2.png"]
    for i in range(2):
        for clusterNum in range(2,5):
            for gs_idx in range(2):
                for gc_idx in range(2):
                    for m in range(2):
                        start = time.time()
                        img_c,img_s,rows,cols,channels = load_img(img_list[i])
                        
                        kernel = calculate_kernel(img_s,gs[gs_idx],img_c,gc[gc_idx])
                        cluster_result = get_initial_cluster(m,clusterNum,kernel,rows,cols,img_s)
                        
                        perform_kernel_kmeans(rows,cols,kernel,clusterNum,cluster_result,img_list[i],m,gs[gs_idx],gc[gc_idx])
                        end = time.time()
                        print("The time of mode ",str(m)," execution is :","{:.2f}".format((end-start)), "s\n")