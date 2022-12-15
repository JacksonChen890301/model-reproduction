import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from data_preprocessing import get_scaler, Predictor1_Dataset


def pred_result(model, loader, dataset, pred_target):
    """
    計算每個epoch下model對training set及testing set的誤差(normalized)
    """
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.cuda().float())
            pred.append(outputs.detach().cpu().numpy())
            gt.append(targets)

    gt, pred = np.concatenate(gt, axis=0).reshape(-1, 1), np.concatenate(pred, axis=0).reshape(-1, 1)

    _, scaler_y = get_scaler(pred_target=pred_target)
    gt, pred = scaler_y.inverse_transform(gt), scaler_y.inverse_transform(pred)
    plt.scatter(gt[:, 0], pred[:, 0], c='blue', s=6)
    plt.plot([np.min(gt), np.max(gt)], [np.min(gt), np.max(gt)], ls='--', c='black')
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('prediction', fontsize=14)
    plt.title(dataset+'set result', fontsize=16)
    plt.savefig(dataset+'_real_result.png')
    plt.close()


def real_RMSE_and_MAPE(model, loader, pred_target):
    """
    計算dimreduct1, dimreduct2的真實誤差(RMSE, MAPE)
    """
    _, scaler_y = get_scaler(pred_target=pred_target)
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.cuda().float())
            pred.append(outputs.detach().cpu().numpy())
            gt.append(targets)

    gt, pred = np.concatenate(gt, axis=0).reshape(-1, 1), np.concatenate(pred, axis=0).reshape(-1, 1)
    gt, pred = scaler_y.inverse_transform(gt), scaler_y.inverse_transform(pred)
    n = len(gt)
    rmse = np.sqrt(np.sum(np.square(gt[:, 0]-pred[:, 0]))/n)
    mape = np.sum(np.abs(pred[:, 0]-gt[:, 0])/gt[:, 0])/n
    return rmse, mape


def loss_profile(trn_loss, val_loss):
    """
    plot loss v.s. epoch curve
    """
    plt.plot(np.arange(len(trn_loss)), trn_loss, c='blue', label='trn_loss', ls='--')
    plt.plot(np.arange(len(val_loss)), val_loss, c='red', label='val_loss', ls='--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend()
    plt.savefig('loss_profile.png')
    plt.close()


def adjust_learning_rate(optimizer, full_ep, epoch, warmup_ep, base_lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_ep:
        lr = base_lr * epoch / warmup_ep
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_ep) / (full_ep - warmup_ep)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



def predictor1_model_evaluation(model, iter, eval_length=[0, 9, 99]):
    """
    根據不同input length評估Predictor1之預測誤差
    """
    _, scaler_y = get_scaler()

    model.eval()
    trn_rmse, test_rmse = [], []
    for cycles in eval_length:
        trn_set = Predictor1_Dataset(train=True, last_padding=False, fix_length=cycles)
        test_set = Predictor1_Dataset(train=False, last_padding=False, fix_length=cycles)
        trn_loader = DataLoader(trn_set, batch_size=95, num_workers=0, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=25, num_workers=0, drop_last=False, shuffle=False)
        with torch.no_grad():
            for inputs, targets in trn_loader:
                outputs = model(inputs.cuda().float()).reshape(-1, 2)
                pred = outputs.detach().cpu().numpy()
                gt = targets.reshape(-1, 2)
                gt, pred = scaler_y.inverse_transform(gt), scaler_y.inverse_transform(pred)
                n = len(gt)
                rmse = np.sqrt(np.sum(np.square(gt[:, 0]-pred[:, 0]))/n)
                # mape = np.sum(np.abs(pred-gt)/gt)/n
                trn_rmse.append(rmse)

            # if iter%100==0:
            #     plt.plot([np.min(gt), np.max(gt)], [np.min(gt), np.max(gt)], ls='--', c='black')
            #     plt.scatter(gt, pred, c='blue', s=6, label='training')

            for inputs, targets in test_loader:
                outputs = model(inputs.cuda().float()).reshape(-1, 2)
                pred = outputs.detach().cpu().numpy()
                gt = targets.reshape(-1, 2)
                gt, pred = scaler_y.inverse_transform(gt), scaler_y.inverse_transform(pred)
                n = len(gt)
                rmse = np.sqrt(np.sum(np.square(gt[:, 0]-pred[:, 0]))/n)
                test_rmse.append(rmse)

            # if iter%100==0:
            #     plt.scatter(gt, pred, c='red', s=6, label='testing')
            #     plt.legend()
            #     plt.savefig(str(cycles)+'-cycle prediction.png')
            #     plt.close()
    return trn_rmse, test_rmse
