import warnings
warnings.filterwarnings("ignore")
from GMVAE import Encoder,Decoder,GMVAE,log1p_layer,norm_layer,score
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
import seaborn as sns
import anndata as an
from anndata.experimental.pytorch import AnnLoader
import argparse
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scanpy as sc
import pandas as pd
import time
import scib
import umap
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
sns.set_theme()
data_dir="../data/"


def plot(test_data,name,dim,title):
    isol_lab_asw=scib.metrics.isolated_labels_asw(test_data,label_key="cell_type",batch_key="batch",embed="embed")
    kbet=scib.metrics.kBET(test_data, batch_key="batch", label_key="cell_type", type_="embed", embed="embed")
    batch_sil=scib.metrics.silhouette_batch(test_data, batch_key="batch", label_key="cell_type", embed="embed")
    resolutions=np.linspace(0.1,2.,20)
    sc.pp.neighbors(test_data,use_rep="embed")
    nmi=0
    for res in resolutions:
        sc.tl.leiden(test_data,resolution=res)
        new_nmi=scib.metrics.nmi(test_data,cluster_key="leiden",label_key="cell_type")
        nmi=max(nmi,new_nmi)
    ari=0
    clust_num=np.arange(1,len(test_data.obs["cell_type"].unique()))
    for num in clust_num:
        kmean=KMeans(n_clusters=num).fit_predict(test_data.obsm["embed"])
        test_data.obs["kmean"]=kmean
        ari_new=scib.metrics.ari(test_data,label_key="cell_type",cluster_key="kmean")
        ari=max(ari,ari_new)
    
    print("Cell type ARI: {0:.3f}".format(ari))
    print("Cell type ASW: {0:.3f}, Batch ASW: {1:.3f}".format(isol_lab_asw,batch_sil))
    print("Cell type NMI: {0:.3f}, Batch kBET: {1:.3f}".format(nmi,kbet))
    values=umap.UMAP().fit_transform(test_data.obsm["embed"])
    data=pd.DataFrame({"UMAP 1":values[:,0],"UMAP 2":values[:,1],"Cell type":test_data.obs["cell_type"]})
    data=data.sort_values('Cell type', ascending=True)
    fig=plt.figure(figsize=(12,8))
    ax=plt.gca()
    plot=sns.scatterplot(ax=ax,data=data,x="UMAP 1",y="UMAP 2",hue="Cell type",legend=True,marker=".",s=60)
    ax.text(0.5,-0.15,"Cell type ASW: {0:.3f}, Cell type NMI: {1:.3f}, Cell type ARI {2:.3f}".format(isol_lab_asw,nmi,ari),
            transform=ax.transAxes,
                 horizontalalignment='center',verticalalignment='center')
    ax.text(0.5,-0.2,"Batch ASW: {0:.3f}, Batch kBET: {1:.3f}".format(batch_sil,kbet),transform=ax.transAxes,
                 horizontalalignment='center',verticalalignment='center')
    ax.set_title(title)
    plt.legend( ncol = 1,fontsize=8,bbox_to_anchor=(1.0, 1.05)) 
    plt.tight_layout()
    plt.savefig(name)
    return np.array((ari,isol_lab_asw,nmi,batch_sil,kbet))

def train(args):
    dim_x=args.dimx
    dim_w=args.dimw
    device="cuda" if args.cuda else "cpu"
    train_data=an.read_h5ad(data_dir+"GEX_train_data.h5ad")
    test_data=an.read_h5ad(data_dir+"GEX_test_data.h5ad")
    num_batches=len(train_data.obs["batch"].unique())
    print("number of unique batches: {}".format(num_batches))
    mean=np.mean(train_data.layers["counts"].toarray())
    std=np.std(train_data.layers["counts"].toarray())
    print("mean: {0:.2f},std: {1:.2f}".format(mean,std))
    print("shape for train data: {}".format(train_data.X.shape))
    print("shape for test data: {}".format(test_data.X.shape))
    num_f=train_data.X.shape[1]
    n=train_data.X.shape[0]
    n_test=test_data.X.shape[0]
    data_loader_train=AnnLoader(train_data,args.batch_size,True,use_cuda=args.cuda)
    data_loader_test=AnnLoader(test_data,args.batch_size,True,use_cuda=args.cuda)
    encoder=Encoder([num_f,300,200],dim_x,dim_w,preproces=norm_layer(mean,std))
    decoder=Decoder([dim_x+num_batches,200,300,num_f])
    model=GMVAE(encoder,decoder,dim_x,dim_w,args.K,300,lamb=torch.Tensor([args.L])).to(device)
    if args.speed_compile:
        model=torch.compile(model, mode="default", backend="inductor")
    loss_arr=[]
    loss_arr_test=[]
    optim=Adam(model.parameters(),args.learning_rate)
    scheduler=ExponentialLR(optim,gamma=0.7,verbose=True)
    start=time.time()
    loss_z_arr=[]
    for i in range(1,args.num_epoche+1):
        loss_ep=0
        loss_ep_test=0
        model.train()
        loss_z=0
        for data_batch in tqdm(data_loader_train,"training"):
            optim.zero_grad()
            codes=torch.tensor(data_batch.obs["batch"].cat.codes,device=device)
            batches=F.one_hot(codes.long(),num_classes=num_batches)
            X=data_batch.layers["counts"]
            rec,klw,klz,con=model(X,args.M,batches)
            loss=torch.sum(rec+klw+klz+con)
            loss.backward()
            loss_ep+=loss.item()/n
            optim.step()
        loss_arr.append(loss_ep)
        model.eval()
        for data_batch in tqdm(data_loader_test,"evaluation"):
            optim.zero_grad()
            codes=torch.tensor(data_batch.obs["batch"].cat.codes,device=device)
            batches=F.one_hot(codes.long(),num_classes=num_batches)
            X=data_batch.layers["counts"]
            rec,klw,klz,con=model(X,args.M,batches)
            loss=torch.sum(rec+klw+klz+con)
            loss_ep_test+=loss.item()/n_test
            loss_z+=torch.sum(klz).item()/n_test
        loss_arr_test.append(loss_ep_test)
        print("epoch: {0:.0f}, loss: {1:.3f}, test loss: {2:.3f}".format(i,loss_ep,loss_ep_test))
        print("z prior loss",loss_z)
        loss_z_arr.append(loss_z)
        if i%25==0:
            scheduler.step()
    end=time.time()
    print("training for {}".format(end-start))
    fig=plt.figure(figsize=(10,6))
    ax=plt.axes()
    sns.scatterplot(ax=ax,x=np.arange(0,len(loss_arr)),y=loss_arr,label="train error")
    sns.scatterplot(ax=ax,x=np.arange(0,len(loss_arr_test)),y=loss_arr_test,label="test error")
    ax.set_ylabel("-ELBO loss")
    ax.set_xlabel("epoch")
    plt.savefig("loss_curve_{}_{}_{}_{}.png".format(dim_x,dim_w,args.K,args.L))
    plt.close()

    fig=plt.figure(figsize=(10,6))
    ax=plt.axes()
    sns.scatterplot(ax=ax,x=np.arange(0,len(loss_arr)),y=loss_z_arr)
    ax.set_ylabel("$z$ prior loss")
    ax.set_xlabel("epoch")
    ax.set_ylim(0,4)
    if args.L>0:
        ax.hlines(y=args.L,xmin=0,xmax=len(loss_z_arr),color="black")
    plt.savefig("z_prior_{}_{}_{}_{}.png".format(dim_x,dim_w,args.K,args.L))
    
    
    sample_arr=np.zeros([0,dim_x])
    obs_arr=[]
    batch_arr=[]
    data_loader_test=AnnLoader(test_data,args.batch_size,False,use_cuda=args.cuda)
    for data_batch in data_loader_test:
        samples=model.latent(data_batch.layers["counts"]).cpu().detach().numpy()
        sample_arr=np.concatenate([sample_arr,samples],axis=0)
        obs_arr=[*obs_arr,*data_batch.obs["cell_type"].values]
        batch_arr=[*batch_arr,*data_batch.obs["batch"].values]

    test_data.obsm["embed"]=sample_arr
    torch.save(model.state_dict(),"GMVAE_{}_{}_{}_{}.tc".format(dim_x,dim_w,args.K,args.L))
    return plot(test_data,"umap_latent_{}_{}_{}_{}.png".format(dim_x,dim_w,args.K,args.L),dim_x,
         "UMAP embedding of latent space, $K={}$, $n_x={}$".format(args.K,dim_x))



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="GMM VAE for microseq clustering")
    parser.add_argument("-b","--batch-size",default=128,type=int)
    parser.add_argument("-n","--num-epoche",default=40,type=int)
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float)
    parser.add_argument("-c","--cuda",default=True,type=bool)
    parser.add_argument("-x","--dimx",default=80,type=int)
    parser.add_argument("-w","--dimw",default=60,type=int)
    parser.add_argument("-k","--K",default=22,type=int)
    parser.add_argument("-m","--M",default=10,type=int)
    parser.add_argument("-l","--L",default=0,type=float)
    parser.add_argument("-s","--speed-compile",default=True,type=bool)
    train(parser.parse_args())