
import argparse
import os
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

from AL_run import AL_Run


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='SimpleSVFEND_tvva', help='assign model name')
parser.add_argument('--dataset_type', default='simpleSVFEND_tvva_gen', help='assign dataset function')
parser.add_argument('--dataset_name', default='fakesv', help='fakett/fakesv')
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--batch_size', type = int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epoch_stop', type=int, default=5) 
parser.add_argument('--seed', type=int)
parser.add_argument('--gpu', default='7')
parser.add_argument('--lr', type=float)
parser.add_argument('--dropout', type=float) 
parser.add_argument('--weight_decay', type=float)


parser.add_argument('--init_ckp_path',default='path to inited models (trained on human annotated data)')
parser.add_argument('--al_itteration',type=int)
parser.add_argument('--al_pool_size',type=int)

parser.add_argument('--path_param', default= './checkpoints/')
parser.add_argument('--path_tensorboard', default= './tb/')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING']='1'
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



print (args)

config = {
        'model_name': args.model_name,

        'dataset_type':args.dataset_type,
        'epoches': args.epoches,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epoch_stop': args.epoch_stop,
        
        'device': args.gpu,
        'lr': args.lr,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        
        'path_param': args.path_param,
        'path_tensorboard': args.path_tensorboard,

        'dataset_name':args.dataset_name,
        'init_ckp_path':args.init_ckp_path,
        'al_itteration':args.al_itteration,
        'al_pool_size':args.al_pool_size,
        }



if __name__ == '__main__':
    # mp.set_start_method("spawn")

    AL_Run(config = config
        ).main()
