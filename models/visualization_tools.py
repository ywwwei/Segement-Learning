import numpy as np
import torch
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import os
def MDS_plot(X, metric,color = True):
    X=X.numpy()
    dim_list=X.shape #B,T,N,d
    batch_size=dim_list[0]
    frame_num=dim_list[1]
    query_num=dim_list[2]
    feature_dim=dim_list[3]
    picture_num=frame_num*batch_size
    for i in np.arange(query_num):
        if i==0:
            X_ordered_list=X[:,:,i,:].reshape(-1, X[:,:,i,:].shape[-1])
        else:
            X_ordered_list=np.concatenate((X_ordered_list, X[:,:,i,:].reshape(-1, X[:,:,i,:].shape[-1])), axis=0)
    embedding = MDS(n_components=2, dissimilarity="precomputed")
    similarities = squareform(pdist(X_ordered_list, metric))
    X_transformed = embedding.fit_transform(similarities)
    #print(X_transformed)
    #torch.save( + "pt")
    plt.clf()

    if color:
        colors = plt.cm.rainbow(np.linspace(0,1,num=query_num))

        for i in range(query_num):
            #if magnitude[i]>=0.001*max_magnitude:
            print(X_transformed[i:i+picture_num, 0])
            print(colors.shape)
            plt.scatter(X_transformed[i*picture_num:(i+1)*picture_num, 0], X_transformed[i*picture_num:(i+1)*picture_num, 1], color=colors[i])
        #if X_transformed[-1, 0] ** 2 + X_transformed[-1, 1] ** 2 < 10000:
    else:
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1],color="blue")
    #plt.legend()
    dirname=os.path.dirname
    plt.savefig(os.path.join(dirname(dirname(__file__)), os.path.join("figs", "train_test_acc_change_noise.png")))
    return

def test_MDS_plot(metric="cosine", color=True):
    query_num=10
    feature_dim=10
    batch_size=2
    frame_num=2
    X=torch.zeros(batch_size,frame_num,query_num,feature_dim)
    for i in range(query_num):
        center=torch.normal(mean=0,std=10,size=(feature_dim,))
        #print(X[:,:,i,:].shape)
        center_dup=center.repeat((batch_size,frame_num,1))
        X[:,:,i,:]=center_dup+torch.normal(mean=0,std=0.5,size=X[:,:,i,:].shape)
    MDS_plot(X, metric=metric,color=color)
if __name__ == '__main__':
    test_MDS_plot()