import torch
from torch import nn
from torch.nn import functional as F
from typing import List,Optional,Tuple
import numpy as np
from sklearn.metrics import silhouette_score,silhouette_samples

class log1p_layer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return torch.log1p(x)

class norm_layer(nn.Module):
    """
    Normalizing layer to preprocess data using known mean and std
    """
    def __init__(self,mean,std):
        super().__init__()
        self.mean=mean
        self.std=std
    def forward(self,x):
        return (x-self.mean)/self.std

class Encoder(nn.Module):
    """
    General class for encoder
    """
    def __init__(self,sizes:List,
                 dimx:int,
                 dimw:int,
                 preproces=None,
                 dropout:Optional[float]=0.05) -> None:
        """
        sizes
        """
        super().__init__()
        self.latent_size = sizes[-1]
        self.network = []
        if preproces is not None:
            self.network.append(preproces)
        for i in range(len(sizes)-1):
            self.network.append(nn.Linear(sizes[i],sizes[i+1]))
            self.network.append(nn.LayerNorm(sizes[i+1]))
            self.network.append(nn.GELU())
            self.network.append(nn.Dropout(dropout))
        self.infer_x = nn.Linear(sizes[-1],dimx*2)
        self.infer_w = nn.Linear(sizes[-1],dimw*2)
        self.network = nn.Sequential(*self.network)

    def forward(self,inp:torch.Tensor)-> Tuple[Tuple[torch.Tensor]]:
        inp = self.network(inp)
        return (torch.chunk(self.infer_w(inp),2,dim=-1),
                torch.chunk(self.infer_x(inp),2,dim=-1))

class Decoder(nn.Module):
    def __init__(self,sizes:List[int]) -> None:
        super().__init__()
        self.network = []
        for i in range(len(sizes)-2):
            self.network.append(nn.Linear(sizes[i],sizes[i+1]))
            self.network.append(nn.LayerNorm(sizes[i+1]))
            self.network.append(nn.GELU())
            self.network.append(nn.Dropout(p=0.05))
        self.network = nn.Sequential(*self.network)
        self.logits = nn.Linear(sizes[-2],sizes[-1])
        self.tot_count = nn.Linear(sizes[-2],sizes[-1])
    
    def log_prob(self,x:torch.Tensor, y:torch.Tensor):
        """
        -log propability of output
        y - observed value
        x - latent
        """
        x = self.network(x)
        logits = self.logits(x)
        tot_count = F.softplus(self.tot_count(x)) + 1e-3
        unnorm = (tot_count * F.logsigmoid(-logits) + y * F.logsigmoid(logits))
        normalization = (-torch.lgamma(tot_count + y) + torch.lgamma(1. + y) +torch.lgamma(tot_count))
        return -torch.sum(unnorm-normalization, axis=-1)

def log_like_gaussian(mean:torch.Tensor, std,point:torch.Tensor):
    """
    log value of probability for each cluster
    """
    return torch.sum(-(point - mean)**2/(2 * std**2)-torch.log(std),dim=-1)

def KL_gaussian(mu_1:torch.Tensor,sigma_1_log:torch.Tensor,mu_2:torch.Tensor,sigma_2_log:torch.Tensor):
    """
    kl divergence for two gaussians
    mu_1 - first mean
    sigma_1_log - log of first std
    mu_2 - second mean
    sigma_2_log - log of second std
    """
    return torch.sum(-sigma_1_log+sigma_2_log+
                     (torch.exp(2*sigma_1_log)+(mu_1-mu_2)**2)/(2*torch.exp(sigma_2_log*2))
                     -1/2,dim=-1)

def KL_categorical(probs_1,probs_2):
    """
    kl-divergence between two categorical distributions
    probs_1 - first probs
    probs_2 - second probs
    """
    return torch.sum(probs_1*(torch.log(probs_1+1e-8)-torch.log(probs_2+1e-8)),dim=0)

class GMVAE(nn.Module):

    def __init__(self,encoder,
                 decoder,
                 dim_x:int,
                 dim_w:int,
                 K:int,
                 GMM_hidden_dim:int,
                 lamb:Optional[float]=0,
                 device="cuda") -> None:
        """
        General class for gaussian mixture VAE
        encoder - encoder that will output tuple of tuple of tensors
        decoder - class with log_prob method 
        dim_x - dimensionality of x dimension
        dim_w - dimensionality of w dimension
        K - number of clusters
        GMM_hidden_dim - number of neurons in hidden dimension of beta network
        lamb - threshold for z-prior to be turned on
        """
        super().__init__()
        self.encoder=encoder
        self.GMM=nn.Sequential(nn.Linear(dim_w,GMM_hidden_dim),nn.GELU(),nn.LayerNorm(GMM_hidden_dim),nn.Dropout(0.05),nn.Linear(GMM_hidden_dim,2*K*dim_x))
        self.decoder=decoder
        self.K=K
        self.dim_x=dim_x
        self.dim_w=dim_w
        self.lamb=torch.tensor([lamb,],device=device)
        self.device=device

    def forward(self,y:torch.Tensor,M:int,lab_batch=None) -> torch.Tensor:
        """
        y - observation
        M - number of samples used to estimate gradient
        """
        batch = y.shape[0]
        ((w_mu,w_sigma_log), (x_mu,x_sigma_log)) = self.encoder(y)
        x=torch.randn([M,*x_mu.shape],device=self.device) * torch.exp(x_sigma_log) + x_mu
        w=torch.randn([M,*w_mu.shape],device=self.device) * torch.exp(w_sigma_log) + w_mu
        means, log_std = torch.chunk(self.GMM(w).reshape(self.K,M,batch,self.dim_x*2),2,dim=3)
        std = torch.exp(log_std)
        log_probs = log_like_gaussian(means,std,x)
        z_posterior = F.softmax(log_probs,dim=0)
        KL_w_prior = KL_gaussian(w_mu,w_sigma_log,torch.zeros_like(w_mu),torch.zeros_like(w_sigma_log))
        KL_z_prior = torch.maximum(KL_categorical(z_posterior,torch.ones_like(z_posterior)/self.K),
                                   self.lamb)
        conditional_prior = torch.sum(z_posterior*KL_gaussian(x_mu,x_sigma_log,means,log_std),dim=0)
        if lab_batch is None:
            rec_loss = self.decoder.log_prob(x,y.unsqueeze(0).expand(M,*y.shape))
        else:
            lab_batch=lab_batch.unsqueeze(0).expand(M,*lab_batch.shape)
            new_x=torch.cat((x,lab_batch),dim=2)
            rec_loss = self.decoder.log_prob(new_x,y.unsqueeze(0).expand(M,*y.shape))
        return (torch.mean(rec_loss,axis=0),
                torch.mean(KL_w_prior,axis=0),
                torch.mean(KL_z_prior,axis=0),
                torch.mean(conditional_prior,axis=0))
    
    def latent(self,y:torch.Tensor) -> torch.Tensor:
        """
        return latent representation of data
        y - observation
        """
        ((w_mu,w_sigma_log),(x_mu,x_sigma_log))=self.encoder(y)
        return x_mu
    
def score(predictions,width):
    score1=silhouette_score(predictions.iloc[:,0:width].values,labels=predictions["cell_type"].values)
    score_final=0
    for batch in predictions["batch"].unique():
        batch_data=predictions.loc[predictions['batch'] == batch]
        sample_scores=1-np.abs(silhouette_samples(X=batch_data.iloc[:,0:width].values,labels=batch_data["cell_type"].values))
        score_final+=np.mean(sample_scores)
    return (score1+1)/2,score_final/(len(predictions["batch"].unique()))