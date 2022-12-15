import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import torch
import random
from tqdm import tqdm
from discharge_model import *
from data_preprocessing import get_scaler

"""
於'Data-driven prediction of battery cycle life before capacity degradation'中使用的dataset
由124顆商用LFP電池(APR18650M1A)組成 以快充及4C放電循環至EoL
其額定電容量為1.11Ah 額定電壓為3.3V
資料被分為三個bath
"""
def mat_to_pkl():
    """
    將mat檔中需要的資料截取至pkl檔
    """
    filename = ['Severson_Dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat',
                'Severson_Dataset/2017-06-30_batchdata_updated_struct_errorcorrect.mat',
                'Severson_Dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat']
    # 各batch有問題的電池 要加以清理
    b1_err = [0, 1, 2, 3, 4, 8, 10, 12, 13, 18, 22]
    b2_err = [1, 6, 9, 10, 21, 25]
    b3_err = [23, 32, 37]
    bat_dict = {}  # dict. for all batch imformation
    for b in range(len(filename)): # batch數
        f = h5py.File(filename[b], 'r')
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        for i in range(num_cells): # 該batch下的電池cell數量
            """
            以下資料在ITRI資料上沒有紀錄或不必要 暫時不使用
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            """
            if b==0 and i in b1_err:
                continue
            if b==1 and i in b2_err:
                continue
            if b==2 and i in b3_err:
                continue
            cl = f[batch['cycle_life'][i, 0]][()]
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary = {'QC': summary_QC, 'QD': summary_QD, 'chargetime': summary_CT, 
                        'TMIN': summary_TM, 'TMAX': summary_TX, 'TAVG': summary_TA}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(1, cycles['I'].shape[0]): # 該cell實驗運行的cycle數
                """
                以下資料在ITRI資料上沒有紀錄或不必要 暫時不使用
                T = np.hstack((f[cycles['T'][j, 0]][()]))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
                t = np.hstack((f[cycles['t'][j, 0]][()]))
                """
                T = np.hstack((f[cycles['T'][j, 0]]))
                I = np.hstack((f[cycles['I'][j, 0]]))
                V = np.hstack((f[cycles['V'][j, 0]]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]]))
                dd = np.diff(np.diff(Qd))
                dis_s = np.where(np.diff(Qd)>=1e-3)[0][0] # 放電開始
                dis_e = np.where(dd>1e-4)[0][-1]+1 # 放電結束
                if dis_e-dis_s >= 700:
                    print(dis_e-dis_s, b, i, j)
                cd = {'I': I[dis_s:dis_e], 'Qd': Qd[dis_s:dis_e], 'V': V[dis_s:dis_e], 'T': T[dis_s:dis_e]}
                cycle_dict[str(j)] = cd

            cell_dict = {'cycle_life': cl, 'summary': summary, 'cycles': cycle_dict}
            if b == 0:
                key = 'b1c' + str(i).zfill(2)
            elif b == 1:
                key = 'b2c' + str(i).zfill(2)
            else:
                key = 'b3c' + str(i).zfill(2)
            print(key)
            bat_dict[key] = cell_dict

    print(len(bat_dict.keys()), bat_dict.keys())
    with open('Severson_Dataset/all_batch_dc.pkl', 'wb') as fp:
        pickle.dump(bat_dict, fp)


def pkl_preprocessing(filename='Severson_Dataset/all_batch_dc.pkl', index=[0, 100]):
    path = 'Severson_Dataset/Severson_each_cell/'
    dataset = pickle.load(open(filename, 'rb'))
    chargetime = []
    for cell_id in tqdm(dataset.keys()):
        curve, RUL = [], []
        cl = dataset[cell_id]['cycle_life']
        chargetime.append(dataset[cell_id]['summary']['chargetime'][-1])
        n_cycles = len(dataset[cell_id]['cycles'].keys())
        for i in range(n_cycles):
            cycle_id = i+1
            if cl-cycle_id>=1 and cycle_id>1:
                V = dataset[cell_id]['cycles'][str(i)]['V']
                Qd = dataset[cell_id]['cycles'][str(i)]['Qd']
                I = dataset[cell_id]['cycles'][str(i)]['I']
                T = dataset[cell_id]['cycles'][str(i)]['T']
                if len(V) < 1:  # wrong data points number
                    print(cell_id)
                    continue
                # 把資料一律線性插值到500個POINT
                interp_id = np.linspace(0, len(V)-1, 500)
                V_interp = np.interp(interp_id, np.arange(len(V)), V).reshape(1, -1)
                Qd_interp = np.interp(interp_id, np.arange(len(Qd)), Qd).reshape(1, -1)
                I_interp = np.interp(interp_id, np.arange(len(I)), I).reshape(1, -1)
                T_interp = np.interp(interp_id, np.arange(len(T)), T).reshape(1, -1)
                if np.max(I_interp) > 0.5:
                    # 去掉沒把charge資料清乾淨的discharge curve
                    continue
                VQIT = np.concatenate([V_interp, Qd_interp, I_interp, T_interp], axis=0)
                curve.append(np.expand_dims(VQIT, axis=0))
                RUL.append(cl-cycle_id)
        curve = np.concatenate(curve, axis=0)
        RUL = np.array(RUL)
        np.save(path+cell_id+"_Curve", curve[index[0]:index[1]])    
        np.save(path+cell_id+"_RUL", RUL[index[0]:index[1]])   
    np.save(path+"chargetime", np.array(chargetime))


def pkl_preprocessing_for_predictor(length=100):
    """
    將所有cell分別存成:
    (8, 100)的特徵 -> QC, QD, TAVG, TMIN, TMAX, chargetime, eol_feature, chargetime_feature
    (2,)的預測目標 -> eol, eol_chargetime
    """
    pklfile = 'Severson_Dataset/all_batch_dc.pkl'
    savepath = 'Severson_Dataset/discharge_model_each_cell/'
    feature_selector = ['models/Feature_Selector_1_s44.pth', 'models/Feature_Selector_1_s44.pth']
    dataset = pickle.load(open(pklfile, 'rb'))
    scaler_X, _ = get_scaler('both')
    for cell_id in dataset.keys(): 
        # 循環間特徵整理
        qc = dataset[cell_id]['summary']['QC'][1:length+1].reshape(1, -1)
        qd = dataset[cell_id]['summary']['QD'][1:length+1].reshape(1, -1)
        tavg = dataset[cell_id]['summary']['TAVG'][1:length+1].reshape(1, -1)
        tmin = dataset[cell_id]['summary']['TMIN'][1:length+1].reshape(1, -1)
        tmax = dataset[cell_id]['summary']['TMAX'][1:length+1].reshape(1, -1)
        chargetime = dataset[cell_id]['summary']['chargetime'][1:length+1].reshape(1, -1)
        feature = np.concatenate([qc, qd, tavg, tmin, tmax, chargetime], axis=0) # (6, 100)
        # 預測目標 EOL/chargetime
        eol = dataset[cell_id]['cycle_life'][0, 0]
        eol_chargetime = dataset[cell_id]['summary']['chargetime'][-1]
        n_cycles, curve = len(dataset[cell_id]['cycles'].keys()), []
        for i in range(n_cycles):
            cycle_id = i+1
            if eol-cycle_id>=1 and cycle_id>1:
                V = dataset[cell_id]['cycles'][str(i)]['V']
                Qd = dataset[cell_id]['cycles'][str(i)]['Qd']
                I = dataset[cell_id]['cycles'][str(i)]['I']
                T = dataset[cell_id]['cycles'][str(i)]['T']
                # 把資料一律線性插值到500個POINT
                interp_id = np.linspace(0, len(V)-1, 500)
                V_interp = np.interp(interp_id, np.arange(len(V)), V).reshape(1, -1)
                Qd_interp = np.interp(interp_id, np.arange(len(Qd)), Qd).reshape(1, -1)
                I_interp = np.interp(interp_id, np.arange(len(I)), I).reshape(1, -1)
                T_interp = np.interp(interp_id, np.arange(len(T)), T).reshape(1, -1)
                if np.max(I_interp) > 0.5: # 去掉沒把charge資料清乾淨的discharge curve
                    continue
                VQIT = np.concatenate([V_interp, Qd_interp, I_interp, T_interp], axis=0)
                curve.append(np.expand_dims(VQIT, axis=0))
                if len(curve)==length:
                    break
        curve = np.concatenate(curve, axis=0) # shape(100, 4, 500)
        # load feature selector & calculate feature 7, 8
        dimreduct1 = torch.load(feature_selector[0]).cuda()
        dimreduct2 = torch.load(feature_selector[1]).cuda()
        dimreduct1.eval()
        dimreduct2.eval()
        curve = scaler_X.transform(curve.transpose((0, 2, 1)).reshape(-1, 4)).reshape(100, 500, 4).transpose((0, 2, 1))
        with torch.no_grad():
            feature_EOL = dimreduct1(torch.tensor(curve).cuda().float()).detach().cpu().numpy()
            feature_chargetime = dimreduct2(torch.tensor(curve).cuda().float()).detach().cpu().numpy()
        feature = np.concatenate([feature, feature_EOL.reshape(1, -1), feature_chargetime.reshape(1, -1)], axis=0) # shape (8, 100)
        np.save(savepath+cell_id+"_predictor1_feature", feature)
        np.save(savepath+cell_id+"_predictor1_target", np.array([eol, eol_chargetime]))    


def pytorch_dataset_preprocessing(train_val_split=0.8, seed=0, folder=''):
    """
    分成三組: EoL<600, 600<=EoL<=1200, EoL>1200
    training, validation和testing要平均分配樣本
    high:mid:low = 46:67:8
    """
    def np_concat(array):
        return np.concatenate(array, axis=0)
    path = 'Severson_Dataset/Severson_each_cell/'
    filename = sorted(os.listdir(path))
    chargetime = np.load(path+'chargetime.npy')
    low_x, mid_x, high_x = [], [], []
    low_t, mid_t, high_t = [], [], []
    for i in range((len(filename)-1)//2):
        curve = np.load(path+filename[2*i])
        EOL = np.load(path+filename[2*i+1])
        target = np.concatenate([np.full((len(EOL),1), EOL[0]), np.full((len(EOL),1), chargetime[i])], axis=1)
        if EOL[0] < 600:
            low_x.append(curve)
            low_t.append(target)
        elif EOL[0] > 1200:
            high_x.append(curve)
            high_t.append(target)
        else:
            mid_x.append(curve)
            mid_t.append(target)

    low_split, mid_split, high_split = int(len(low_x)*train_val_split), int(len(mid_x)*train_val_split), int(len(high_x)*train_val_split)
    # print(low_split, mid_split, high_split)
    # 隨機調整順序
    random.seed(seed)
    low, mid, high = list(zip(low_x, low_t)), list(zip(mid_x, mid_t)), list(zip(high_x, high_t))
    random.shuffle(low), random.shuffle(mid), random.shuffle(high)
    low_x, low_t = zip(*low)
    mid_x, mid_t = zip(*mid)
    high_x, high_t = zip(*high)
    np.save(folder+'Severson_TrnSet_input', np_concat([np_concat(low_x[:low_split]), np_concat(mid_x[:mid_split]), np_concat(high_x[:high_split])]))
    np.save(folder+'Severson_ValSet_input', np_concat([np_concat(low_x[low_split:]), np_concat(mid_x[mid_split:]), np_concat(high_x[high_split:])]))
    np.save(folder+'Severson_TrnSet_target', np_concat([np_concat(low_t[:low_split]), np_concat(mid_t[:mid_split]), np_concat(high_t[:high_split])]))
    np.save(folder+'Severson_ValSet_target', np_concat([np_concat(low_t[low_split:]), np_concat(mid_t[mid_split:]), np_concat(high_t[high_split:])]))

    # save for EOL dataset
    trn_eol_x = [element[0] for element in low_x[:low_split]]+[element[0] for element in mid_x[:mid_split]]+[element[0] for element in high_x[:high_split]]
    trn_eol_t = [element[0] for element in low_t[:low_split]]+[element[0] for element in mid_t[:mid_split]]+[element[0] for element in high_t[:high_split]]
    val_eol_x = [element[0] for element in low_x[low_split:]]+[element[0] for element in mid_x[mid_split:]]+[element[0] for element in high_x[high_split:]]
    val_eol_t = [element[0] for element in low_t[low_split:]]+[element[0] for element in mid_t[mid_split:]]+[element[0] for element in high_t[high_split:]]
    np.save(folder+'Severson_TrnSet_input_EOL', np.stack(trn_eol_x, axis=0))
    np.save(folder+'Severson_ValSet_input_EOL', np.stack(val_eol_x, axis=0))
    np.save(folder+'Severson_TrnSet_target_EOL', np.stack(trn_eol_t, axis=0))
    np.save(folder+'Severson_ValSet_target_EOL', np.stack(val_eol_t, axis=0))


pkl_preprocessing_for_predictor()