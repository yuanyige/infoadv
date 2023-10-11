import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder




class Discriminator(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, dis_lambda = 0, dis_ub=False):
        super(Discriminator, self).__init__()
        self.encoder: Encoder = encoder
        self.dis_lambda = dis_lambda
        self.use_ub = dis_ub
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_ub:
            mu, logvar = self.encoder(x, edge_index)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            return self.encoder(x, edge_index)
            
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss_lb(self, zm1: torch.Tensor, z1: torch.Tensor, zm2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        if self.use_ub:
            refl_sim_1 = f(self.sim(zm1, zm1))
            between_sim_1 = f(self.sim(zm1, zm2))
            between_sim_2 = f(self.sim(z1, zm2))

            return -torch.log(
                between_sim_1.diag()
                / (between_sim_1.diag()+refl_sim_1.sum(1)-refl_sim_1.diag()+between_sim_2.sum(1)-between_sim_2.diag()))#, between_sim_1.diag().mean(), (refl_sim_1.sum(1)-refl_sim_1.diag()+between_sim_2.sum(1)-between_sim_2.diag()).mean()

        else:
            refl_sim = f(self.sim(zm1, zm1))
            between_sim = f(self.sim(zm1, zm2))
            
            return -torch.log(
                between_sim.diag()
                / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))#,between_sim.diag().mean(),(refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - between_sim.diag()).mean()


    def semi_loss_ub(self, mu, logvar):
        num_nodes = mu.size(0)
        KLD = -0.5 / num_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return KLD
    

    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, zmu1 = None, zmu2 = None, mu1 = None, logvar1 = None, mu2 = None, logvar2 = None, mean: bool = True):
        
        if self.use_ub:
            loss_lb_ret1 = self.semi_loss_lb(zmu1, z1, zmu2)
            loss_lb_ret2 = self.semi_loss_lb(zmu2, z2, zmu1)
            loss_ub_ret1 =  self.semi_loss_ub(mu1, logvar1)
            loss_ub_ret2 =  self.semi_loss_ub(mu2, logvar2)

        else:
            loss_lb_ret1 = self.semi_loss_lb(z1, None, z2)
            loss_lb_ret2 = self.semi_loss_lb(z2, None, z1)
            loss_ub_ret1,loss_ub_ret2 =  0,0
        
        ret = (loss_lb_ret1 + loss_lb_ret2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        ret = ret + 0.5*(self.dis_lambda * (loss_ub_ret1 + loss_ub_ret2))
        #print('loss',l1,l2,loss_ub_ret1,loss_ub_ret2)
        return ret#,(-loss_lb_ret1[0].mean(), loss_ub_ret1, -loss_lb_ret2[0].mean(), loss_ub_ret2, loss_lb_ret1[1], loss_lb_ret2[1], loss_lb_ret1[2], loss_lb_ret2[2])

