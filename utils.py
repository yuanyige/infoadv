import torch
import random
import numpy as np 
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, WikiCS, Coauthor
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_sparse import SparseTensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) not in [float,int]:
            val=val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, meters, num_batches=0, prefix=""):
        if num_batches:
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        else:
            self.batch_fmtstr = ''
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, batch=0):
        entries = [self.prefix + str(epoch) + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('   '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class AddFeatureNoise(object):

    def __init__(self, noise_rate):
        self.noise_rate = noise_rate

    def __call__(self, data):
        feature_noise_mask = torch.empty((data.x.size(1), ),dtype=torch.float32,device=data.x.device).uniform_(0, 1) < self.noise_rate
        data.x = data.x.clone()
        data.x [:, feature_noise_mask] = 1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class AddAdjNoise(object):

    def __init__(self, noise_rate):
        self.noise_rate = noise_rate

    def __call__(self, data):
        data.adj_t = data.adj_t.to_dense()
        adj_noise_mask = torch.empty((data.adj_t.size()),dtype=torch.float32,device=data.adj_t.device).uniform_(0, 1) < self.noise_rate
        data.adj_t = data.adj_t.clone()
        data.adj_t[adj_noise_mask] = 1
        data.adj_t = SparseTensor.from_dense(data.adj_t).set_value(None)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def get_dataset(path, name, noise_rate=0, if_noise =False):
    print("Using dataset {}".format(name))
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','AmazonCom','AmazonPhoto','Reddit' ,'CoauthorCS','Coauthorphy','WikiCS']

    if name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']:
        name = 'dblp' if name == 'DBLP' else name
        return (CitationFull if name == 'dblp' else Planetoid)(
            root=path,
            name=name,
            transform= T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()]) if not if_noise else T.Compose([AddFeatureNoise(noise_rate),T.NormalizeFeatures(),T.ToSparseTensor(),AddAdjNoise(noise_rate)]))

    elif name in['AmazonCom','AmazonPhoto']:
        return Amazon(
            root=path,
            name='Computers' if name=='AmazonCom' else 'Photo',
            transform= T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()]) if not if_noise else T.Compose([AddFeatureNoise(noise_rate),T.NormalizeFeatures(),T.ToSparseTensor(),AddAdjNoise(noise_rate)]))



    elif name == 'WikiCS':
        return WikiCS(root=path+'/WikiCS', transform=T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()]) if not if_noise else T.Compose([AddFeatureNoise(noise_rate),T.NormalizeFeatures(),T.ToSparseTensor(),AddAdjNoise(noise_rate)]))


    elif name in ['CoauthorCS','Coauthorphy']:
        return Coauthor(
            root=path,
            name='CS' if name=='CoauthorCS' else 'Physics',
            transform= T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()]) if not if_noise else T.Compose([AddFeatureNoise(noise_rate),T.NormalizeFeatures(),T.ToSparseTensor(),AddAdjNoise(noise_rate)]))


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path=None, patience=100, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.best_epoch = None
        self.best_view = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path


    def __call__(self, val_acc, epoch, model, view):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            if self.path:
                self.save_checkpoint(val_acc, epoch, model, view, self.path)
            else:
                self.memory_best_model(val_acc, epoch, model, view)

        elif score <= self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Best score in epoch {self.best_epoch} is {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            if self.path:
                self.save_checkpoint(val_acc, epoch, model, view, self.path)
            else:
                self.memory_best_model(val_acc, epoch, model, view)
            self.counter = 0

    def memory_best_model(self, val_acc, epoch, model, view):
        if self.verbose:
            print(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).')
        self.val_acc_max = val_acc
        self.best_epoch = epoch
        self.best_model = {"dis_model":model[0],"gen_model":model[1]}
        self.best_view = view
            

    def save_checkpoint(self, val_acc, epoch, model, view, path):
        '''Saves model when validation loss decrease.'''
        self.memory_best_model(val_acc, epoch, model, view)
        if model[1]:
            torch.save(
                {
                "val_acc":val_acc,
                "epoch":epoch,
                "dis_model":model[0].state_dict(),
                "gen_model":model[1].state_dict(),
                "aug_edge":view,
                },path)
        else:
            torch.save(
                {
                "val_acc":val_acc,
                "epoch":epoch,
                "dis_model":model[0].state_dict(),
                "gen_model":None,
                "aug_edge":view,
                },path)

        #self.val_acc_max = val_acc

