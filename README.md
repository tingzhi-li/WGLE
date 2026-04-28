# WGLE: Backdoor-free and Multi-bit Black-box Watermarking for Graph Neural Networks


## Environment Setup

Opearting system: Ubuntu 22.04.4 LTS

CPU: Intel Xeon Gold 6248R

GPU: A100 with 80GB 

CUDA version: 12.4

You need to install some third-party libraries with the following commands:

```
pip install torch torchvision torchaudio
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install numpy
pip install scikit-learn
pip install pandas
```



## Introduction

We use RBOVG and WGB as baselines and EaaW as a strawman. Their titles and corresponding links are listed below. Note that since the source code of WGB is not publicly available, we reimplemented it based on the descriptions in the paper.

| Method | Paper | Code |
|---------|---------|---------|
| WGB   | Watermarking Graph Neural Networks based on Backdoor Attacks   | -   |
| RBOVG   | Revisiting Black-box Ownership Verification for Graph Neural Networks   | https://github.com/rkzhou/GNN_OwnVer.git   |
| EaaW | Explanation as a Watermark: Towards Harmless and Multi-bit Model Ownership Verification via Watermarking Feature Attribution | https://github.com/shaoshuo-ss/EaaW.git |

## Repo Contents
Below is the directory structure of our project.

```
WGLE/
├── main.py
├── utils/                     
│   ├── config.py                       # hyper-parameters      
│   ├── dataload.py                     # dataset split
│   ├── models.py                       # models
│   └── utils.py                        # train, test function
├── watermark/                     
│   ├── assess.py                       # evaluate effectiveness,fidelity, robust
│   ├── insight.py                      # t-SNE of insight2 and  insight3; impact of watermark lenth; watermark collision
│   ├── robust.py                       # implements of attacks
│   └── watermark.py                    # implements of watermark generation, embedding and extraction.   
├── datasets/
├── results/
├── models/
└── README.md
```

The core codes of main.py are present as follows.

```
if __name__ == '__main__':
    # our methods
    datasets = ['Cora', 'DBLP', 'CS', 'Physics', 'Blog', 'Photo']  # 'Cora', 'DBLP', 'CS', 'Physics', 'Blog', 'Photo'
    models = ['GAT', 'GTF', 'SSG', 'GCNv2', 'ARMA', 'SAGE']  # 'GAT', 'GTF', 'SSG', 'GCNv2', 'ARMA', 'SAGE'

    args.paradigm = 'inductive'
    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1, 3):
            args.setting = ii # setting 1 is STA; setting 2 is STM
            assess_experiments()
            

    args.paradigm = 'transductive'
    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1, 3):
            args.setting = ii
            assess_experiments()
            

```

## Key hyper-parameter introductions
We list the key hyper-parameters below, including their explanations and available options.

- --dataset: load dataset; the option is ['Cora', 'DBLP', 'CS', 'Physics', 'Blog', 'Photo'] 
- --train_val_test: split the dataset into the training graph, the val graph, the test graph; default = [0.4, 0.3, 0.3]
- --model: load model; the option is ['GAT', 'GTF', 'SSG', 'GCNv2', 'ARMA', 'SAGE']
- --hidden_channels: number of models; default = [800, 200, 50]
- --setting: setting STA,STM; the option is [1,2]
- --n_wm: number of bits of the watermark string; default = 64
- --model_num: the number of generated watermark models

## How to run
```
python main.py
```

We run all models, datasets, STA and STM  in a single pass within `main.py`.  
You can modify `main.py` as needed.

`model.py` contains all the model architectures used in our work, and `dataload.py` stores all the datasets we used.  
If you wish to try other datasets or models, please modify these files accordingly.

```
model.py
dataload.py
```

You can run evaluation.py to compute the ownership verification accuracy (OVA) and false positive rate (FPR).
Note: Make sure that at least 5 watermarked models are generated; otherwise, the evaluation may produce incorrect results.
The default decision threshold is set to 0.75, but you may try other thresholds as well.

```
python evaluation.py
```

You can run `WGB.py` to test the performance of the baseline WGB.  
Please note that this is our reimplementation based on the authors’ paper, and it may differ from the original results reported in their publication.

```
python WGB.py
```


After running, we can found the folder:

```
results/
├── multibit/ # fidelity, robust against pruning and fine_tuning across various lenth of the watermark string
├── collision/ # HMS between the target watermark and the extracted watermark from independently trained models
├── insight2/ # the coordinate values of the t-SNE projection for insight2
├── insight3/ # the coordinate values of the t-SNE projection for insight3
├── Cora/
│   ├── setting1/
│   │   ├── setting1.csv
│   │   ├── pruning_w.csv # pruning of watermarked models
│   │   ├── pruning_i.csv # pruning of independently trained models
│   │   ├── fine_tuning_w.csv
│   │   ├── fine_tuning_i.csv
│   │   ├── overwriting.csv
│   │   └── model_extract.csv
│   ├── setting2/
│   └── setting3/
├── DBLP/
...
```

## Results Viewing
- HMS: hamming similarity
- Mo: the original model without any watermarks
- Mi: the independently trained model
- Mw: the watermarked model
- OVA: Ownership Verification accuracy
- FPR: Ownership Verification false positive rate
- Dimension 1, Dimension 2,	Label, and ARI in Insight3: Each line corresponds each node. "Dimension 1" and "Dimension 2" are the first and second coordinates of each node's 2D t-SNE projection. "Label" is its ground-truth label. "ARI" is Adjusted Rand Index for all node predictions.
- LDDE, Label in Insight2: 'LDDE' is the LDDE value for each selected edge. "Label" denotes the assigned value: "0" for (-), "1" for (+).

## Our results
results.zip is the original results of our paper

## Datasets
All datasets are available for automatic download at runtime, or can be downloaded manually via the following address.

| Dataset | Address |
|---------|---------|
| Cora   | https://github.com/abojchevski/graph2gauss/raw/master/data/ |
| DBLP   | https://github.com/abojchevski/graph2gauss/raw/master/data/ |
| Photo  | https://github.com/shchur/gnn-benchmark/raw/master/data/npz/|
| CS| https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ |
|Physics| https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ |
|Blog | https://drive.usercontent.google.com/download?id={178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS}&confirm=t |
| CiteSeer| https://github.com/abojchevski/graph2gauss/raw/master/data/ |
