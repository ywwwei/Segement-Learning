import argparse
import util.misc as utils
import numpy as np
import torch
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import os
import math
import sys
#sys.path.insert(0, os.path.abspath(".."))
from PIL import Image
import requests
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from main import get_args_parser
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from models import build_model
torch.set_grad_enabled(False);
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
def MDS_plot(X, metric,color = True):
    X=X.numpy()
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
    dirname=os.path.dirname
    plt.savefig(os.path.join(dirname(dirname(__file__)), os.path.join("figs", "train_test_acc_change_noise_MDS.png")))
    return


def TSNE_plot(X, metric,color = True):
    X=X.numpy()
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
    plt.savefig(os.path.join(dirname(dirname(__file__)), os.path.join("figs", "train_test_acc_change_noise_TSNE.png")))
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


    # convert boxes from [0; 1] to image scales

def get_image_attention_output(im,img,model):
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output.to('cpu'))
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1].to('cpu'))
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1].to('cpu'))
        ),
    ]

    # propagate through the model
    img=img.to('cuda')
    model.cuda()
    outputs = model(img)
    img=img.to('cpu')
    model.cpu()
    print(outputs)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    return  conv_features, enc_attn_weights,dec_attn_weights
def visualize_dec_atten_weights(im,img, conv_features, dec_attn_weights, query_num,vid_name,pic_name):
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(22, 7))
    colors = COLORS * 100
    for idx, ax_i in zip(np.arange(query_num),axs.T):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.axis('off')
        ax.set_title("Image")
    fig.tight_layout()
    plt.savefig(os.path.join("figs", "dec_plot"+vid_name+"_"+pic_name+".png"))
def visualize_enc_atten_weights(im,img, conv_features,enc_attn_weights,vid_name,pic_name):
    # output of the CNN
    f_map = conv_features['0']
    print("Encoder attention:      ", enc_attn_weights[0].shape)
    print("Feature map:            ", f_map.tensors.shape)
    # get the HxW shape of the feature maps of the CNN
    shape = f_map.tensors.shape[-2:]
    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape + shape)
    print("Reshaped self-attention:", sattn.shape)
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32

    # let's select 4 reference points for visualization
    idxs = [(200, 200), (280, 400), (200, 500), (440, 500),]

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        print(idx)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(im)
    for (y, x) in idxs:
        scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis('off')
    plt.savefig(os.path.join("figs", "dec_plot"+vid_name+"_"+pic_name+".png"))
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #utils.init_distributed_mode(args)
    #print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion= build_model(args)
    dirname=os.path.dirname
    #model_without_ddp = model
    checkpoint = torch.load(os.path.join("./checkpoints","checkpoint0034.pth"), map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    vid_name="0b102e6d83"
    pic_name="00135"
    image_position=os.path.join("./datasets","test","JPEGImages",vid_name,pic_name+".jpg")
    im = Image.open(image_position)
    transform1= T.Compose([
        T.Resize(640),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform1(im).unsqueeze(0)
    print(img.shape)
    img=img.to('cuda')
    print("sucess")
    model.cuda()
    outputs = model(img)
    img=img.to('cpu')
    model.cpu()
    print(outputs)
    conv_features, enc_attn_weights,dec_attn_weights=get_image_attention_output(im,img,model)
    #print(dec_attn_weights.shape)
    query_num=5
    visualize_dec_atten_weights(im,img, conv_features, dec_attn_weights, query_num,vid_name,pic_name)
    visualize_enc_atten_weights(im,img, conv_features,enc_attn_weights,vid_name,pic_name)
