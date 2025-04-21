import numpy as np
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from watermark.assess import *
from watermark.insight import *
from utils.config import parse_args, device
from utils.utils import train, test
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_wgle(tau, args, output_file='WGLE.json'):

    datasets = ['Cora', 'DBLP', 'Photo', 'Computers', 'CS',
                'Physics']  # 'Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics'
    models = ['GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF']  # 'GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF'

    # Accuracy & FPR for watermark detection
    for dataset in datasets:
        save_json = {}
        args.dataset = dataset
        acc_fpr_metrics = []
        for setting in range(1, 4):
            save_json["Dataset, Setting"] = args.dataset, setting
            mi_hms_path = os.path.join(args.results_path, dataset, f'setting{setting}', f'setting{setting}.csv')
            mi_hms = pd.read_csv(mi_hms_path)['Mi hms'].iloc[1:].to_numpy()
            mw_hms = pd.read_csv(mi_hms_path)['Mw hms'].iloc[1:].to_numpy()
            labels = np.concatenate([np.zeros_like(mi_hms), np.ones_like(mw_hms)])
            preds = np.where(np.concatenate([mi_hms, mw_hms]) < tau, 0, 1 )
            ova = accuracy_score(labels, preds)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            fpr = fp / (fp + tn)
            save_json["OVA"] = ova
            save_json["FPR"] = fpr

            tac = pd.read_csv(mi_hms_path)['Mw test acc'].iloc[1:].to_numpy()
            save_json["TAC mean"] = np.mean(tac)
            save_json["TAC std"] = np.std(tac, ddof=1)

            overhead = pd.read_csv(mi_hms_path)['Time'].iloc[1:].to_numpy()
            save_json["Time"] = np.mean(overhead)

            mi_hms_path = os.path.join(args.results_path, dataset, f'setting{setting}', f'pruning_i.csv')
            mi_hms_list = pd.read_csv(mi_hms_path, usecols=range(11, 22)).iloc[1:].to_numpy()
            mw_hms_path = os.path.join(args.results_path, dataset, f'setting{setting}', 'pruning_w.csv')
            mw_hms_list = pd.read_csv(mw_hms_path, usecols=range(11, 22)).iloc[1:].to_numpy()

            for i in range(11):
                mi_hms = mi_hms_list[:,i]
                mw_hms = mw_hms_list[:,i]
                preds = np.where(np.concatenate([mi_hms, mw_hms]) < tau, 0, 1 )
                labels = np.concatenate([np.zeros_like(mi_hms), np.ones_like(mw_hms)])
                ova = accuracy_score(labels, preds)
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                fpr = fp / (fp + tn)
                hms = np.mean(mw_hms)
                save_json["Pruning"+str(i*10)+'%'+"OVA,FPR,HMS"] = [ova, fpr, hms]

            mi_hms_path = os.path.join(args.results_path, dataset, f'setting{setting}', f'fine_tuning_i.csv')
            mi_hms = pd.read_csv(mi_hms_path)['200hms'].iloc[1:].to_numpy()
            mw_hms_path = os.path.join(args.results_path, dataset, f'setting{setting}', 'fine_tuning_w.csv')
            mw_hms = pd.read_csv(mw_hms_path)['200hms'].iloc[1:].to_numpy()

            preds = np.where(np.concatenate([mi_hms, mw_hms]) < tau, 0, 1)
            labels = np.concatenate([np.zeros_like(mi_hms), np.ones_like(mw_hms)])
            ova = accuracy_score(labels, preds)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            fpr = fp / (fp + tn)
            hms = np.mean(mw_hms)
            save_json["Fine_tuning200" + "OVA,FPR,HMS"] = [ova, fpr, hms]

            # Save as JSON
            with open("{}_{}.json".format(args.dataset, setting), "w") as f:
                f.write(json.dumps(save_json))


if __name__ == '__main__':
    # evaluation
    tau = 0.75
    args = parse_args()
    evaluate_wgle(tau, args)