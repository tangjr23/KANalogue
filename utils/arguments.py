import argparse

def train_parse_arguments():
    parser = argparse.ArgumentParser(description='KANanlog Hyper-Param Search Trainning')

    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate of training progress.')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Integer. 256 by default.')
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        choices=['MNIST', 'FMNIST', 'cifar10'], 
                        help='Choose a dataset: MNIST, FMNIST, cifar10')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Train Device: cuda or cpu')
    parser.add_argument('--acti', type=str, default='None', 
                        choices=['None', 'PosHC', 'NegHC', 'sigmoid', 'tanh'], 
                        help='Choose the activate function: PosHC, NegHC, sigmoid, tanh, or None')
    parser.add_argument('--max_epochs', type=int, default=10_000, 
                        help='Integer. 10_000 by default.')
    parser.add_argument('--patience', type=int, default=15, 
                        help='Integer. 10 by default.')
    parser.add_argument('--neg_input', action='store_true', default=False, 
                        help='Range of input. False for [0, 1] and True for [-1, 1]')
    parser.add_argument('--basis_type', type=str, default='pos-norm', 
                        choices=['pos-norm', 'pos-ori', 'neg-norm', 'neg-ori', 'odd-sym', 'pos-larger', 'neg-larger'], 
                        help='Type of basis functions.')
    parser.add_argument('--basis_combine', type=int, default=3, 
                        choices=[2, 3], 
                        help='Combination type of basis functions. 2 for two basis, [A21, Z21]. 3 for three basis, [A21, Z21, Z15]')
    parser.add_argument('--exp_name', type=str, default=None, 
                        help='Name this exp. Then all the results will be in results/exp-name.')
    parser.add_argument('--search_mode', type=str, default='random', 
                        choices=['grid', 'random', 'ablation', 'best', 'same-param', 'custom', 'even_early'], 
                        help='Searching mode. Random search by default.')
    parser.add_argument('--norm_layer', type=str, default='layer', 
                        choices=['layer', 'batch', 'None'], 
                        help='Type of the norm layer. "layer" for LayerNorm, "batch" for "BatchNorm", and "None" for no norm layer.')
    parser.add_argument('--add_noise', action='store_true', default=False, 
                        help='Add noise or not. Write in cmmd for True.')
    parser.add_argument('--hidden', type=int, nargs='+', default=[64], 
                        help='Hidden dims. [64] by default. Entering "1024 256" for [1024, 256].')
    parser.add_argument('--noise_num', type=int, default=[1], 
                        help='Hidden dims. [64] by default')
    parser.add_argument('--distribution', type=str, default='binary', 
                        choices=['binary', 'uniform', 'gauss'], 
                        help='Noise distribution. "binary" by default.')

    return parser.parse_args()

def data_fit_parse_arguments():
    parser = argparse.ArgumentParser(description='Fit Data By Diff Methods')
    
    parser.add_argument('--fit_mode', type=str, default='line', 
                        choices=['line', 'spline', 'poly'], 
                        help='Fit mode of basis functins. Piecewise linear fit by default.')
    parser.add_argument('--poly_deg', type=int, default=10, 
                        help='Integer. Degree of poly-fit.')
    parser.add_argument('--spline_type', type=str, default='cubic', 
                        choices=['cubic', 'univariate', 'b'], 
                        help='Fit mode of basis functins. Piecewise linear fit by default.')

    return parser.parse_args()

def data_analysis_parse_arguments():
    parser = argparse.ArgumentParser(description='Options of data analysis.')
    
    parser.add_argument('--exp_name', type=str, default='rough_search', 
                        help='Name of the exp. "rough-search" by default.')
    parser.add_argument('--perform_thre', type=float, default=0.2, 
                        help='Float. Threshold used to evaluate if model is well-performed. 0 by default.')
    parser.add_argument('--use_all', action='store_true', default=False, 
                        help='Wether use all the data. "True" for using all and "False" for using only physically feasible models\' data. "False" by dufault.')
    parser.add_argument('--compare', type=str, default='rough,batch', 
                        help='Name of the exp. "rough-search" by default.')

    return parser.parse_args()

def noise_analysis_parse_arguments():
    parser = argparse.ArgumentParser(description='Options for noise-model analysis.')
    
    parser.add_argument('--exp_name', type=str, default='new_structure_exps/same-param', 
                        help='Name of the exp. "rough-search" by default.')
    parser.add_argument('--distribution', type=str, default='binary', 
                        choices=['binary', 'uniform', 'gauss'], 
                        help='Noise distribution. "binary" by default.')

    return parser.parse_args()

def MLP_KAN_train_parse_arguments():
    parser = argparse.ArgumentParser(description='Comparison table Param Search Trainning')

    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='Learning rate of training progress.')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Integer. 256 by default.')
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        choices=['MNIST', 'FMNIST', 'cifar10'], 
                        help='Choose a dataset: MNIST, FMNIST, cifar10')
    parser.add_argument('--device', type=str, default='cuda', 
                        help="use 'CUDA_VISIBLE_DEVICES =' to set device id.")
    
    parser.add_argument('--max_epochs', type=int, default=10_000, 
                        help='Integer. 10_000 by default.')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Integer. 10 by default.')
    parser.add_argument("--model_name", type=str, default="MLP_CMTDAF",
                        choices=["MLP_RTDAF", "MLP_CMTDAF", "MLP_MOSFETac", "BSplineKAN", "GottliebKAN"],
                        help="Model name, e.g. MLP_RTDAF / BSplineKAN.")
    parser.add_argument('--exp_name', type=str, default=None, 
                        help='Name this exp. Then all the results will be in results/exp-name.')
    parser.add_argument('--search_mode', type=str, default='random', 
                        choices=['grid', 'random'], 
                        help='Searching mode. Random search by default.')
    
    parser.add_argument('--hidden', type=int, nargs='+', default=[64], 
                        help='Hidden dims. [64] by default. Entering "1024 256" for [1024, 256].')
    

    return parser