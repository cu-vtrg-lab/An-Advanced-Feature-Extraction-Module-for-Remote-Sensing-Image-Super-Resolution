
from torch.utils.data import DataLoader

def create_dataloaders(args):
    """create dataloader"""
    if args.dataset == 'RSCNN7':
        from data.aid import RSCNN7
        training_set = RSCNN7(args, root_dir='./dataset/RSCNN7/train',  # RSCNN7_dataset
                                 train=True)
        val_set = RSCNN7(args, root_dir='./dataset/RSCNN7/val',
                            train=False)

    elif args.dataset == 'UCMerced':
        from data.ucmerced import UCMercedDataset
        training_set = UCMercedDataset(args, root_dir='./dataset/UCMerced-dataset/train',
                                 train=True)
        val_set = UCMercedDataset(args, root_dir='./dataset/UCMerced-dataset/val',
                            train=False)
        
    elif args.dataset == 'DIV2K':
        from data.div2k import DIV2KDataset
        training_set = DIV2KDataset(args, root_dir='./dataset/DIV2K-dataset/train',
                                    train=True)
        val_set = DIV2KDataset(args, root_dir='./dataset/UCMerced-dataset/val',
                               train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s ' % args.dataset)

    dataloaders = {'train': DataLoader(training_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0),  # args.n_threads
                   'val': DataLoader(val_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0),  # args.n_threads
                    }

    return dataloaders



















