import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='.\data', help='Datasets root')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj'])
    parser.add_argument('--num_workers', default=4, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    # encoder/decoder - number of layers and channels
    parser.add_argument('--N_S', default=4, type=int, help='number of encoder and decoder layers')
    parser.add_argument('--en_de_c', default=64, type=int,  help='encoder / decoder channel number')

    # h and h-1 - - number of layers and channels
    parser.add_argument('--N_h', default=4, type=int,  help='number of h layers')
    parser.add_argument('--h_c', default=512, type=int,  help='h channel number')

    # T and T-1 - - number of layers and channels64
    parser.add_argument('--N_T', default=4, type=int,   help='number of T layers')
    parser.add_argument('--T_c', default=512, type=int,  help='T channel number')

    parser.add_argument('--groups', default=8, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--lamda1', default=0., type=float, help='mae')
    parser.add_argument('--lamda2', default=0., type=float, help='x and x_pred')
    parser.add_argument('--lamda3', default=0., type=float, help='z and z_pred')
    parser.add_argument('--lamda4', default=0., type=float, help='xi and xi_pred')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
    exp.calculate_paras_flops_memory()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  finish <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')