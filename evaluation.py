import numpy as np
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from watermark.assess import *
from watermark.insight import *
from utils.config import parse_args
from utils.utils import train, test
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_wgle(tau, args, output_file='WGLE.json'):

    datasets = ['Cora', 'DBLP', 'Photo', 'CS',
                'Physics', 'Blog']  # 'Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics'
    models = ['GAT', 'GTF', 'SAGE', 'SSG', 'GCNv2', 'ARMA']  # 'GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF'

    # Accuracy & FPR for watermark detection
    for dataset in datasets:
        save_json = {}
        args.dataset = dataset
        acc_fpr_metrics = []
        for setting in range(1, 3):
            save_json["Dataset, Setting"] = args.dataset, setting
            file_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', f'setting{setting}.csv')
            mi_hms = pd.read_csv(file_path)['Mi hms'].iloc[1:].to_numpy()
            mw_hms = pd.read_csv(file_path)['Mw hms'].iloc[1:].to_numpy()
            labels = np.concatenate([np.zeros_like(mi_hms), np.ones_like(mw_hms)])
            q1, q2, q3 = np.quantile(mw_hms, [0.25, 0.5, 0.75])
            preds = np.where(np.concatenate([mi_hms, mw_hms]) <= tau, 0, 1 )
            ova = accuracy_score(labels, preds)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            fpr = fp / (fp + tn)
            save_json["OVA"] = ova
            save_json["FPR"] = fpr
            save_json["M_w Q1/2/3"] = [q1, q2, q3]
            q1, q2, q3 = np.quantile(mi_hms, [0.25, 0.5, 0.75])
            save_json["M_i Q1/2/3"] = [q1, q2, q3]

            tac = pd.read_csv(file_path)['Mw test acc'].iloc[1:].to_numpy()
            save_json["TAC mean"] = np.mean(tac)
            save_json["TAC std"] = np.std(tac, ddof=1)

            overhead = pd.read_csv(file_path)['Time'].iloc[1:].to_numpy()
            save_json["Time"] = np.mean(overhead)

            mi_hms_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', f'pruning_i.csv')
            mi_hms_list = pd.read_csv(mi_hms_path, usecols=range(11, 22)).iloc[1:].to_numpy()
            mw_hms_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', 'pruning_w.csv')
            mw_hms_list = pd.read_csv(mw_hms_path, usecols=range(11, 22)).iloc[1:].to_numpy()

            for i in range(7, 8):
                mi_hms = mi_hms_list[:,i]
                mw_hms = mw_hms_list[:,i]
                preds = np.where(np.concatenate([mi_hms, mw_hms]) <= tau, 0, 1 )
                labels = np.concatenate([np.zeros_like(mi_hms), np.ones_like(mw_hms)])
                ova = accuracy_score(labels, preds)
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                fpr = fp / (fp + tn)
                mw_q = np.quantile(mw_hms, [0.25, 0.5, 0.75])
                mi_q = np.quantile(mi_hms, [0.25, 0.5, 0.75])
                save_json["Pruning"+str(i*10)+'%'+"OVA,FPR,Mw Q1/2/3, Mi Q1/2/3"] = [ova, fpr, mw_q[0], mw_q[1], mw_q[2], mi_q[0], mi_q[1], mi_q[2]]

            mi_hms_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', f'fine_tuning_i.csv')
            mi_hms = pd.read_csv(mi_hms_path)['200hms'].iloc[1:].to_numpy()
            mw_hms_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', 'fine_tuning_w.csv')
            mw_hms = pd.read_csv(mw_hms_path)['200hms'].iloc[1:].to_numpy()

            preds = np.where(np.concatenate([mi_hms, mw_hms]) <= tau, 0, 1)
            labels = np.concatenate([np.zeros_like(mi_hms), np.ones_like(mw_hms)])
            ova = accuracy_score(labels, preds)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            fpr = fp / (fp + tn)
            mw_q = np.quantile(mw_hms, [0.25, 0.5, 0.75])
            mi_q = np.quantile(mi_hms, [0.25, 0.5, 0.75])
            save_json["Fine_tuning200" + "OVA,FPR,Mw Q1/2/3, Mi Q1/2/3"] = [ova, fpr, mw_q[0], mw_q[1], mw_q[2], mi_q[0], mi_q[1], mi_q[2]]

            file_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', f'overwriting.csv')
            original_tac = pd.read_csv(file_path)['original test_acc'].iloc[1:].to_numpy()
            tac = pd.read_csv(file_path)['overwriting test_acc'].iloc[1:].to_numpy()
            original_hms = pd.read_csv(file_path)['original HMS'].iloc[1:].to_numpy()
            hms = pd.read_csv(file_path)['overwriting HMS'].iloc[1:].to_numpy()
            q1, q2, q3 = np.quantile(hms, [0.25, 0.5, 0.75])
            save_json["Overwriting" + "TAC mean, TAC std, HMS Q1, HMS Q2, HMS Q3"] = [np.mean(tac), np.std(tac, ddof=1), q1, q2, q3]
            q1, q2, q3 = np.quantile(original_hms, [0.25, 0.5, 0.75]) - [q1, q2, q3]
            save_json["Overwriting drop" + "TAC, HMS Q1, HMS Q2, HMS Q3"]= [np.mean(original_tac)-np.mean(tac), q1, q2, q3]

            file_path = os.path.join(args.results_path, dataset, args.paradigm, f'setting{setting}', f'model_extract.csv')
            original_tac = pd.read_csv(file_path)['original test_acc'].iloc[1:].to_numpy()
            tac = pd.read_csv(file_path)['SAGE_a test_acc'].iloc[1:].to_numpy()
            original_hms = pd.read_csv(file_path)['original HMS'].iloc[1:].to_numpy()
            hms = pd.read_csv(file_path)['SAGE_a HMS'].iloc[1:].to_numpy()
            q1, q2, q3 = np.quantile(hms, [0.25, 0.5, 0.75])
            save_json["MEA" + "TAC mean, TAC std, HMS Q1, HMS Q2, HMS Q3"] = [np.mean(tac), np.std(tac, ddof=1), q1, q2, q3]
            q1, q2, q3 = np.quantile(original_hms, [0.25, 0.5, 0.75]) - [q1, q2, q3]
            save_json["MEA drop" + "TAC, HMS Q1, HMS Q2, HMS Q3"] = [np.mean(original_tac) - np.mean(tac), q1, q2, q3]

            # Save as JSON
            with open("{}_{}.json".format(args.dataset, setting), "w") as f:
                f.write(json.dumps(save_json))


if __name__ == '__main__':
    # evaluation
    tau = 0.75
    args = parse_args()
    evaluate_wgle(tau, args)