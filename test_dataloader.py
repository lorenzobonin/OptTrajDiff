from argparse import ArgumentParser, Namespace

from datamodules import ArgoverseV2DataModule
from datasets import ArgoverseV2Dataset
from predictors import DiffNet
from omegaconf import OmegaConf

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--config_path', type=str, required=True)

    #DiffNet.add_model_specific_args(parser)
    args = parser.parse_args()

    print(type(args))
    

    file_args = OmegaConf.load(args.config_path)
    _args = {**vars(args), **OmegaConf.to_container(file_args, resolve=True)}
    args = Namespace(**_args)

    test_dataset = ArgoverseV2Dataset(args.root, 'test', args.test_raw_dir, args.test_processed_dir,
                                               args.test_transform)

    # datamodule = {
    #         'argoverse_v2': ArgoverseV2DataModule,
    #     }[args.dataset](**vars(args)) # if the dataset is argoverse_v2, an ArgoverseV2DataModule is instantiated with the argumentsin args