
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from encoder import Encoder
import pyro
import random
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj as random_dropout_adj

class Generator(nn.Module):
    def __init__(self, encoder, hidden_dim=128, temperature=0.2):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.fc_edge = nn.Linear(in_features = 2 * hidden_dim, out_features = 1)
        self.criteria = nn.L1Loss()
        self.temperature = temperature
        self.tau: float = 0.5


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    
    def _norm_to_interval(self, arr, ub=1,lb=0):
        k = (ub-lb)/(arr.max() - arr.min())
        return lb + k * (arr-arr.min())


    def sample_adj(self, x, sp_adj, edge_remove_rate):
        """ sample an adj from the predicted edge probabilities of ep_net """

        '''计算连边丢弃概率'''
        node_emb = self.encoder(x, sp_adj)
        edge_index = torch.cat((sp_adj.storage.row().reshape(1,-1),sp_adj.storage.col().reshape(1,-1)),dim=0)
        edge_emb = torch.cat((node_emb[edge_index[0]],node_emb[edge_index[1]]),dim=1)
        edge_logits = self.fc_edge(edge_emb)
        #print('edge_score',edge_logits.max().item(),edge_logits.min().item(),edge_logits.mean().item(),edge_logits.median().item()) 
        
        '''控制丢弃概率上限'''
        edge_probs = self._norm_to_interval(edge_logits, ub = edge_remove_rate, lb = max(edge_remove_rate-0.5,0)).reshape(-1)
     
        '''根据概率采样丢弃矩阵'''
        #back2device=sp_adj.device()
        remove_adj_prob = torch.zeros(sp_adj.sizes(),device=sp_adj.device()).index_put((edge_index[0],edge_index[1]), edge_probs)
        remove_adj = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=remove_adj_prob).rsample()
        adj = torch_sparse.SparseTensor.from_dense(torch.mul((1-remove_adj), sp_adj.to_dense()))
        adj_aug_rate = adj.nnz()/sp_adj.nnz()
        return adj, adj_aug_rate
     
    
    def forward(self, x: torch.Tensor, sp_adj: torch.Tensor, edge_remove_rate=0, feature_mask_rate=0):

        feature = random_drop_feature(x, feature_mask_rate) 
        adj, adj_aug_rate = self.sample_adj(x, sp_adj, edge_remove_rate)

        return feature, adj, adj_aug_rate


    def semi_loss_lb(self, zmu1: torch.Tensor, zmu2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(zmu1, zmu1))
        between_sim = f(self.sim(zmu1, zmu2))
        
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def semi_loss_ub(self, mu, logvar):
        num_nodes = mu.size(0)
        KLD = -0.5 / num_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return KLD
    

    def loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        ret = self.semi_loss_ub(mu, logvar)
        return ret


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


    
def random_drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


if __name__ == "__main__":
    import warnings
    from torch_geometric.datasets import Planetoid, CitationFull
    warnings.filterwarnings("ignore")

    dataset = Planetoid('./datasets', name="Cora", transform=T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()]))
    data = dataset[0]
    print('ori adj',data.adj_t,'\n')
    print('ori x shape',torch.nonzero(data.x).shape[0]/data.x.shape[0])

    encoder = Encoder(1433, 128, F.relu, base_model='gcn', n_layers=2)
    model = Generator(encoder,1433, 128)
    x, adj, _ = model(data.x,data.adj_t,0.6,0.6)
    print('gen adj',adj)
    print('gen x', x)
    print('gen x shape',torch.nonzero(x).shape[0]/x.shape[0],'\n')
    
    encoder2 = Encoder(1433, 128, F.relu, base_model='gcn', n_layers=2)
    print('output',encoder2(x,adj))

