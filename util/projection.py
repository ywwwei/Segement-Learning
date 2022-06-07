import numpy as np
import torch
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import os
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
def MDS_plot(X, metric,color = True, title=''):
    X=X.cpu().detach().numpy()
    dim_list=X.shape #B,T,N,d
    batch_size=dim_list[0]
    frame_num=dim_list[1]
    query_num=dim_list[2]
    feature_dim=dim_list[3]
    picture_num=frame_num*batch_size

    # for i in np.arange(query_num):
    #     if i==0:
    #         X_ordered_list=X[:,:,i,:].reshape(-1, X[:,:,i,:].shape[-1])
    #     else:
    #         X_ordered_list=np.concatenate((X_ordered_list, X[:,:,i,:].reshape(-1, X[:,:,i,:].shape[-1])), axis=0)
    X_ordered_list=np.transpose(X, (2,0,1,3 )).reshape(-1,X.shape[-1])
    embedding = MDS(n_components=2, dissimilarity="precomputed")
    similarities = squareform(pdist(X_ordered_list, metric))
    X_transformed = embedding.fit_transform(similarities)
    #print(X_transformed)
    #torch.save( + "pt")
    plt.clf()

    if color:
        color_index=np.linspace(0,1,num=query_num)
        color_index=np.repeat(color_index,picture_num)
        colors = plt.cm.rainbow(color_index)
        shape_index=np.array([['s','o','*','D','<']])
        shape_index=np.repeat(shape_index, query_num//5,axis=0)
        #shape_index=shape_index.transpose()
        #print(shape_index)
        shape_index=shape_index.flatten()
        shape_index=np.repeat(shape_index, picture_num)
        #print(shape_index)
        #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color=colors)
        scatter = mscatter(X_transformed[:, 0], X_transformed[:, 1], c=colors,  m=shape_index)
        # for i in range(query_num):
        #     #if magnitude[i]>=0.001*max_magnitude:
        #     print(X_transformed[i:i+picture_num, 0])
        #     print(colors.shape)
        #     plt.scatter(X_transformed[i*picture_num:(i+1)*picture_num, 0], X_transformed[i*picture_num:(i+1)*picture_num, 1], color=colors[i])
        #if X_transformed[-1, 0] ** 2 + X_transformed[-1, 1] ** 2 < 10000:
    else:
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1],color="blue")
    #plt.legend()
    # plt.title(title)
    plt.savefig(os.path.join("figs", f"{title}_MDS.png"))
    return


def TSNE_plot(X, metric,color = True, title=''):
    X=X.cpu().detach().numpy()
    dim_list=X.shape #B,T,N,d
    batch_size=dim_list[0]
    frame_num=dim_list[1]
    query_num=dim_list[2]
    feature_dim=dim_list[3]
    picture_num=frame_num*batch_size
    X_ordered_list=np.transpose(X, (2,0,1,3 )).reshape(-1,X.shape[-1])
    # for i in np.arange(query_num):
    #     if i==0:
    #         X_ordered_list=X[:,:,i,:].reshape(-1, X[:,:,i,:].shape[-1])
    #     else:
    #         X_ordered_list=np.concatenate((X_ordered_list, X[:,:,i,:].reshape(-1, X[:,:,i,:].shape[-1])), axis=0)
    embedding = TSNE(n_components=2,metric="precomputed", square_distances=True)
    similarities = squareform(pdist(X_ordered_list, metric))
    X_transformed = embedding.fit_transform(similarities)
    #print(X_transformed)
    #torch.save( + "pt")
    plt.clf()

    if color:
        color_index=np.linspace(0,1,num=query_num)
        color_index=np.repeat(color_index,picture_num)
        shape_index=np.array([['s','o','*','D','<']])
        shape_index=np.repeat(shape_index, query_num//5,axis=0)
        #shape_index=shape_index.transpose()
        #print(shape_index)
        shape_index=shape_index.flatten()
        shape_index=np.repeat(shape_index, picture_num)
        #print(shape_index)
        colors = plt.cm.rainbow(color_index)
        #print(color_index.shape)
        #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color=colors,marker=shape_index)
        scatter = mscatter(X_transformed[:, 0], X_transformed[:, 1], c=colors,  m=shape_index)
        # for i in range(query_num):
        #     #if magnitude[i]>=0.001*max_magnitude:
        #     print(X_transformed[i:i+picture_num, 0])
        #     print(colors.shape)
        #     plt.scatter(X_transformed[i*picture_num:(i+1)*picture_num, 0], X_transformed[i*picture_num:(i+1)*picture_num, 1], color=colors[i])
        # #if X_transformed[-1, 0] ** 2 + X_transformed[-1, 1] ** 2 < 10000:
    else:
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1],color="blue")
    #plt.legend()
    dirname=os.path.dirname
    # plt.title(title)
    plt.savefig(os.path.join("figs", f"{title}_TSNE.png"))
    return

def test_MDS_plot(metric="cosine", color=True):
    query_num=10
    feature_dim=10
    batch_size=5
    frame_num=5
    X=torch.zeros(batch_size,frame_num,query_num,feature_dim)
    for i in range(query_num):
        center=torch.normal(mean=0,std=10,size=(feature_dim,))
        #print(X[:,:,i,:].shape)
        center_dup=center.repeat((batch_size,frame_num,1))
        X[:,:,i,:]=center_dup+torch.normal(mean=0,std=0.5,size=X[:,:,i,:].shape)
    MDS_plot(X, metric=metric,color=color)
    TSNE_plot(X, metric=metric,color=color)
if __name__ == '__main__':
    test_MDS_plot()