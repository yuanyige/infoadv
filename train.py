import argparse
import time
import copy
import os.path as osp
import random
import warnings
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
from torch_geometric.utils import dropout_adj as random_dropout_adj
from torch_geometric.utils import subgraph
import torch_sparse 

from torch.utils.tensorboard import SummaryWriter

from encoder import Encoder
from discriminator import Discriminator
from generator import Generator, CLUB, random_drop_feature
from eval import label_classification_cv, mean_classifier
from utils import ProgressMeter, AverageMeter, EarlyStopping, get_dataset, set_requires_grad, setup_seed, get_activation



def train(dis: Discriminator, gen:Generator, mi_estimator, optimizer_d, optimizer_g, optimizer_mi,
            x, sp_adj, gen_enable, dis_ub=False, gen_ub=False,
            drop_edge_rate=0, drop_feature_rate=0, device=None):
    
    # print("======Start======")

    #nodes_to_extract = [i for i in range(data.x.shape[0])]
    nodes_to_extract = random.sample(range(x.shape[0]), 1000)
    sp_adj = subgraph(nodes_to_extract, sp_adj, relabel_nodes=True)[0]
    x = x[nodes_to_extract]
    # print(x, x.shape)
    # print(sp_adj)
    sp_adj = torch_sparse.SparseTensor.from_edge_index(sp_adj, sparse_sizes=(x.shape[0],x.shape[0]))

    dis.train()
    if gen_enable:
        gen.train()
    
    if gen_enable:

        '''gen aug view'''
        aug_feature_1, aug_sp_adj_1, adj_aug_rate = gen(
            x.to(device[0]), sp_adj.to(device[0]), drop_edge_rate[0], drop_feature_rate[0])
        
        '''gen random view'''
        rdm_feature_2 = random_drop_feature(x, drop_feature_rate[1])
        # print('rdm_feature_2',rdm_feature_2.sum())
        rdm_edge_index_2 = random_dropout_adj(torch.nonzero(sp_adj.to_dense()).t(), p=drop_edge_rate[1])[0]
        # print('rdm_edge_index_2',rdm_edge_index_2.shape)
        
        '''views into dis'''
        h1, mu1, logvar1 = dis(aug_feature_1.to(device[1]), aug_sp_adj_1.to(device[1]))
        h2, mu2, logvar2 = dis(rdm_feature_2.to(device[1]), rdm_edge_index_2.to(device[1]))
        z1 ,zmu1 = dis.projection(h1), dis.projection(mu1)
        z2 ,zmu2 = dis.projection(h2), dis.projection(mu2)

        '''gen backwards'''
        set_requires_grad([dis], False)  # Ds require no gradients when optimizing Gs
        optimizer_g.zero_grad()  # set generator's gradients to zero
        if gen_ub:
            mi_estimator.eval()
            loss_g = mi_estimator(zmu1.to(device[0]), zmu2.to(device[0]))
        else:
            loss_g = gen.loss(mu=mu1.to(device[0]), logvar=logvar1.to(device[0]))
        loss_g.backward(retain_graph=True)

        '''mi training'''
        if gen_ub:
            for _ in range(10):
                mi_estimator.train()
                mi_loss = mi_estimator.learning_loss(zmu1.detach(), zmu2.detach())
                optimizer_mi.zero_grad()
                mi_loss.backward()
                optimizer_mi.step()

        '''dis backwards'''
        set_requires_grad([dis], True)
        optimizer_d.zero_grad()  # set discriminator's gradients to zeros
        if dis_ub:
            loss_d = dis.loss(z1=z1.to(device[1]), z2=z2.to(device[1]), zmu1=zmu1.to(device[1]), zmu2=zmu2.to(device[1]), mu1=mu1.to(device[1]), logvar1=logvar1.to(device[1]), mu2=mu2.to(device[1]), logvar2=logvar2.to(device[1]), mean=True)
        else:
            loss_d = dis.loss(z1=zmu1.to(device[1]), z2=zmu2.to(device[1]), mean=True)
        loss_d.backward()
        
        optimizer_g.step()
        optimizer_d.step()

        return loss_g.item(), loss_d.item(), adj_aug_rate, aug_sp_adj_1

    else:
        optimizer_d.zero_grad()
        rdm_edge_index_1 = random_dropout_adj(torch.nonzero(sp_adj.to_dense()).t(), p=drop_edge_rate[0])[0]
        rdm_edge_index_2 = random_dropout_adj(torch.nonzero(sp_adj.to_dense()).t(), p=drop_edge_rate[1])[0]
        
        rdm_feature_1 = random_drop_feature(x, drop_feature_rate[0])
        rdm_feature_2 = random_drop_feature(x, drop_feature_rate[1])

        if dis_ub:
            h1, mu1, logvar1 = dis(rdm_feature_1.to(device[1]), rdm_edge_index_1.to(device[1]))
            h2, mu2, logvar2 = dis(rdm_feature_2.to(device[1]), rdm_edge_index_2.to(device[1]))

            z1, zmu1 = dis.projection(h1), dis.projection(mu1)
            z2, zmu2 = dis.projection(h2), dis.projection(mu2)

            loss = dis.loss(z1=z1, z2=z2, zmu1=zmu1, zmu2=zmu2, mu1=mu1, logvar1=logvar1, mu2=mu2, logvar2=logvar2, mean=True)

        else:
            h1 = dis(rdm_feature_1.to(device[1]), rdm_edge_index_1.to(device[1]))
            h2 = dis(rdm_feature_2.to(device[1]), rdm_edge_index_2.to(device[1]))

            z1 = dis.projection(h1)
            z2 = dis.projection(h2)

            loss = dis.loss(z1=z1, z2=z2, mean=True)
        
        loss.backward()
        optimizer_d.step()
        return 0, loss.item(), 0, rdm_edge_index_1



def test(model: Discriminator, data):
    model.eval()
    z = model(data.x, data.adj_t)
    mean, std = label_classification_cv(z[0] if type(z) is tuple else z , data.y, ratio=0.1)
    return mean, std



def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        choices=['Cora', 'CiteSeer', 'PubMed', 'AmazonCom','AmazonPhoto','CoauthorCS','Coauthorphy','WikiCS','ogbn-arxiv'],
                        default='Cora') 
    parser.add_argument('--gpu', nargs='+', default=[0,0])
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./log/')
    parser.add_argument('--earlystop_round', type=int, default=50000)
    parser.add_argument('--if_resume', type=bool, default=False)
    parser.add_argument('--if_save', type=bool, default=True)
    parser.add_argument('--noise_rate', type=float, default=0)
    args = parser.parse_args()

    '''define devices'''
    if (args.gpu) and (torch.cuda.is_available()):
        device = [torch.device('cuda:{}'.format(i)) for i in args.gpu]
        print('Using GPU:'+str(args.gpu)+' for training')
    else:
        device = [torch.device("cpu"),torch.device("cpu")]
        print("Using CPU for training")
    #device = [torch.device("cpu"),torch.device('cpu')]

    '''define hyperparam'''
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    setup_seed(config['seed'])
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = get_activation(config['activation'])
    base_model = config['base_model']
    gen_in_epoch = config['gen_in_epoch']
    drop_feature_rate= (config['feature_drop_ub_aug'],config['feature_drop_rate_rdm'])
    drop_edge_rate = (config['edge_drop_ub_aug'],config['edge_drop_rate_rdm'])
    gen_ub = config['gen_ub']
    dis_lambda = config['dis_lambda']
    tau = config['tau']
    

    '''define datasets'''
    dataset = get_dataset('./datasets', args.dataset, args.noise_rate, if_noise = False)
    data = dataset[0]
    print(data.x.sum())
    print(data.edge_index)
    data_eval = copy.deepcopy(data)
    data_eval.adj_t = torch_sparse.SparseTensor.from_edge_index(data_eval.edge_index, sparse_sizes=(data_eval.x.shape[0],data_eval.x.shape[0]))


    # # nodes_to_extract 是你希望抽取的节点 ID 列表
    # #nodes_to_extract = [i for i in range(data.x.shape[0])]
    # nodes_to_extract = random.sample(range(data.x.shape[0]), 1000)

    # # 使用 subgraph 方法来抽取子图
    # data.edge_index = subgraph(nodes_to_extract, data.edge_index, relabel_nodes=True)[0]
    
    # data.x = data.x[nodes_to_extract]
    # data.y = data.y[nodes_to_extract]

    # print(data.x, data.x.shape)
    # print(data.edge_index)
    # print('asasasasa',data_eval.x.shape)

    # data.adj_t = torch_sparse.SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.x.shape[0],data.x.shape[0]))

    # print('aaaax',data.adj_t)

    '''define encoder'''
    dis_ub=None
    if gen_in_epoch:
        encoder_g = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model,use_ub = False).to(device[0])
        dis_ub=True
    elif dis_lambda:
        dis_ub=True
    else:
        dis_ub=False
    encoder_d = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, use_ub = dis_ub).to(device[1])

    '''define gen and dis'''
    if gen_in_epoch :
        gen_model = Generator(encoder=encoder_g, hidden_dim=num_hidden).to(device[0])
    dis_model = Discriminator(encoder=encoder_d, num_hidden=num_hidden, num_proj_hidden=num_proj_hidden, tau=tau, dis_lambda = dis_lambda, dis_ub = dis_ub).to(device[1])


    '''define optimizer''' 
    if gen_in_epoch:
        optimizer_g = torch.optim.Adam(
            gen_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_d = torch.optim.Adam(
        dis_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    '''MI est and opt'''
    if gen_in_epoch and gen_ub:
        mi_estimator = CLUB(num_hidden, num_hidden, num_hidden).to(device[0])
        optimizer_mi = torch.optim.Adam(mi_estimator.parameters(), lr = 1e-4)


    '''define model name'''
    train_time = time.strftime('%Y-%m-%dT%H%M%S',time.localtime(time.time()))
    if gen_in_epoch is False: # Baseline 模型
        model_name=f"{dis_lambda}_{args.noise_rate}_{drop_feature_rate}_{drop_edge_rate}_{train_time}"
        print('''Model discription
            dis_lambda: {arg_list[0]}
            noise_rate: {arg_list[1]}
            drop_feature_rate: {arg_list[2]}
            drop_edge_rate: {arg_list[3]}'''.format(
            arg_list=model_name.split('_')))
    else:
        model_name=f"{(gen_ub,dis_lambda)}_{args.noise_rate}_{gen_in_epoch}_{drop_feature_rate}_{drop_edge_rate}_{train_time}"
        print('''Model discription
            eta_lambda: {arg_list[0]}
            noise_rate: {arg_list[1]}
            gen_in_epoch: {arg_list[2]}
            drop_feature_rate: {arg_list[3]}
            drop_edge_rate: {arg_list[4]}'''.format(
            arg_list=model_name.split('_')))
    
    '''define Tensorboard'''
    if args.if_save:
        log_dir=osp.join(args.save_path,model_name)
        writer = SummaryWriter(log_dir)
    else:
        writer=None
    
    '''define Earlystopping'''
    early_stopping = EarlyStopping(path=osp.join(log_dir,'best_model.pt') if args.if_save else None, patience=args.earlystop_round, verbose=True)
    #early_stopping = EarlyStopping(path = None, patience=args.earlystop_round, verbose=True)
 

    '''define Meters'''
    total_time_meter = AverageMeter('Total_time', ':.3f')
    adj_aug_rate_meter = AverageMeter('Edge_aug_rate', ':.4f')
    loss_g_meter = AverageMeter('Loss_G', ':.4f')
    loss_d_meter = AverageMeter('Loss_D', ':.4f')
    f1mi_mean_meter = AverageMeter('F1mi_mean', ':.4f')
    f1mi_std_meter = AverageMeter('F1mi_std', ':.4f')
    progress_train = ProgressMeter([loss_g_meter, loss_d_meter, adj_aug_rate_meter, total_time_meter],prefix="Epoch: ")


    '''resume prev learning'''
    start_epoch = -1
    if args.if_resume:
        checkpoint = torch.load(args.resume_path) 
        print("Loading from: {}".format(args.resume_path))
        gen_model.load_state_dict(checkpoint['gen_model']) 
        dis_model.load_state_dict(checkpoint['dis_model']) 
        optimizer_g.load_state_dict(checkpoint['optimizer_g']) 
        optimizer_d.load_state_dict(checkpoint['optimizer_d']) 
        start_epoch = checkpoint['epoch']
    

    '''go into epoch'''
    start = t()
    new_start = 0 if start_epoch == -1 else start_epoch
    for epoch in range(start_epoch+1, new_start+num_epochs+1):
               
        '''train and test'''
        loss_g, loss_d,  edge_rate,  aug_edge = train(
            dis_model, gen_model if gen_in_epoch else None, mi_estimator if (gen_in_epoch and gen_ub) else None,
            optimizer_d, optimizer_g if gen_in_epoch else None, optimizer_mi if (gen_in_epoch and gen_ub) else None,
            data.x, data.edge_index, 
            gen_enable = gen_in_epoch, #False if ((epoch < gen_in_epoch) or (gen_in_epoch is False)) else True, 
            dis_ub = dis_ub, #True if dis_lambda else False,
            gen_ub = gen_ub,
            drop_edge_rate=drop_edge_rate,
            drop_feature_rate=drop_feature_rate,
            device=device) 
        if epoch % 50 == 0:
            print("start eval...")
            eval_mean, eval_std = test(dis_model, data_eval.to(device[1]))
            #eval_mean, eval_std = np.array([0,0]), np.array([0,0])

        '''update meters'''
        now = t()
        f1mi_mean_meter.update(eval_mean[0])
        f1mi_std_meter.update(eval_std[0])
        loss_g_meter.update(loss_g)
        loss_d_meter.update(loss_d)
        adj_aug_rate_meter.update(edge_rate)
        total_time_meter.update(now - start)
        
        
        '''print ang plot'''
        progress_train.display(epoch)
        print('Eval | F1mi={:.4f}+-{:.4f}, F1ma={:.4f}+-{:.4f}'.format(eval_mean[0], eval_std[0], eval_mean[1], eval_std[1]))
        if writer is not None:
            writer.add_scalar('train/loss/loss_d', loss_d_meter.val, epoch)
            writer.add_scalar('train/loss/loss_g', loss_g_meter.val, epoch)
            writer.add_scalar('train/aug/edge', adj_aug_rate_meter.val, epoch)
            writer.add_scalar('eval/mean', f1mi_mean_meter.val, epoch)
            writer.add_scalar('eval/std', f1mi_std_meter.val, epoch)
            


        '''earlystopping saving'''
        early_stopping(eval_mean[0], epoch, model=(dis_model, gen_model if gen_in_epoch else None), view=aug_edge)
        if early_stopping.early_stop:
            print("==> Early stopping...")
            print("Best downstream score in epoch {}, is {}".format(early_stopping.best_epoch,early_stopping.best_score))
            break
        
        '''regular saving'''
        if epoch+1 % 2000 ==0 and args.if_save:
            torch.save(
            {
            "val_acc":eval_mean[0],
            "epoch":epoch,
            "dis_model":dis_model.state_dict(),
            "gen_model":gen_model.state_dict() if gen_in_epoch else None,
            'optimizer_d': optimizer_d.state_dict(), 
            'optimizer_g': optimizer_g.state_dict() if gen_in_epoch else None, 
            "aug_edge":aug_edge,
            },osp.join(log_dir,'model_{}.pt'.format(epoch))) 

if __name__ == '__main__':
    main()

