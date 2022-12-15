import argparse
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader
from data_preprocessing import Predictor1_Dataset
from discharge_model import init_weights, Predictor_1
from SeversonDataset_preprocess import pytorch_dataset_preprocessing
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('Predictor1 training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--seed', default=44, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='Feature_Selector_2', type=str) 
    parser.add_argument('--finetune', default=False, type=bool)   
    parser.add_argument('--load_checkpoint', default='Predictor_1_s44.pth', type=str)                  

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--lr_schedule', type=bool, default=True, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR')
    parser.add_argument('--anneal_period', type=int, default=10, metavar='LR')
    parser.add_argument('--delta', type=int, default=1)
    return parser


def main(args):
    if torch.cuda.is_available():
        print(" -- GPU is available -- ")

    # pytorch_dataset_preprocessing(seed=args.seed, folder='Severson_Dataset/')
    trn_set = Predictor1_Dataset(train=True, last_padding=False)
    trn_loader = DataLoader(trn_set, batch_size=95, num_workers=0, drop_last=False, shuffle=False)
    val_set = Predictor1_Dataset(train=False, last_padding=False)
    val_loader = DataLoader(val_set, batch_size=25, num_workers=0, drop_last=False, shuffle=False)

    if args.finetune:
        model = torch.load(args.load_checkpoint).cuda()
    else:
        model = Predictor_1(8, 1).apply(init_weights).cuda()
    summary(model, (8, 100)) # architecture visualization

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=args.delta)

    best_rmse = 1000
    trn_loss_record, val_loss_record = [], []
    for epoch in range(args.epochs):
        trn_set_rand = Predictor1_Dataset(train=True, last_padding=True)
        trn_loader_rand = DataLoader(trn_set_rand, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=True)
        model.train()
        batch = 0
        n_minibatch = (len(trn_set_rand)//args.batch_size)
        for inputs, targets in trn_loader_rand:
            batch += 1
            optimizer.zero_grad()
            output = model(inputs.cuda().float())
            loss = criterion(output , targets.reshape(-1, args.target_ch).cuda().float())
            loss.backward()
            optimizer.step()
            if batch%50==1:
                print('epoch:[%d / %d] batch:[%d / %d] loss= %.3f' % 
                    (epoch + 1, args.epochs, batch, n_minibatch, loss.mean()))

        if args.lr_schedule:
            adjust_learning_rate(optimizer, args.epochs, epoch, warmup_ep=10, base_lr=args.lr, min_lr=args.min_lr)

        # model evaluation per epoch
        model.eval()
        with torch.no_grad():
            trn_loss, val_loss = 0, 0
            for inputs, targets in trn_loader:
                output = model(inputs.cuda().float())
                loss = criterion(output , targets.reshape(-1, args.target_ch).cuda().float())
                trn_loss += loss.mean()
            for inputs, targets in val_loader:
                output = model(inputs.cuda().float())
                loss = criterion(output , targets.reshape(-1, args.target_ch).cuda().float())
                val_loss += loss.mean()
            trn_loss_record.append(trn_loss)
            val_loss_record.append(val_loss)
            print('trn_loss: %.3f, val_loss: %.3f' % ((trn_loss), (val_loss)))

        # inverse transform to real EOL
        trn_rmse, test_rmse = predictor1_model_evaluation(model, epoch, eval_length=[0, 9, 99])
        print('training set RMSE 1 cycle: %.3f, 10 cycle: %.3f, 100 cycle: %.3f' %
                (trn_rmse[0], trn_rmse[1], trn_rmse[2]))
        print('testing set RMSE 1 cycle: %.3f, 10 cycle: %.3f, 100 cycle: %.3f' %
                (test_rmse[0], test_rmse[1], test_rmse[2]))
        if test_rmse[2]<best_rmse:
            best_rmse = test_rmse[2]
            if args.fine_tuning:
                torch.save(model, 'predictorv0_finetuned.pth')
            else:
                torch.save(model, 'predictorv0_best_model_.pth')
    torch.save(model, 'predictorv0_last_epoch.pth')

    # training finished
    loss_profile(trn_loss_record, val_loss_record)
    print('best RMSE: %.3f' % (best_rmse))


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)