# WGLE: Backdoor-free and Multi-bit Black-box Watermarking for Graph Neural Networks


## Environment Setup

Opearting system: Ubuntu 22.04.4 LTS

CPU: Intel Xeon Gold 6248R

GPU: A100 with 40GB 

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
    args = parse_args()
    datasets = ['Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics']  # datasets
    models = ['GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF']  # 'GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF' # models

    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1, 4):
            args.setting = ii # setting I, II, III
            assess_experiments(args)

    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1,4):
            args.setting = ii
            assess_insight(args) # insight2, insight3, impact of watermark lenth, watermark collision

    for iii in range(args.model_num): # generate more watermarked models 
        for i in range(6):
            args.dataset = datasets[i]
            args.model = models[i]
            for ii in range(1, 4):
                args.setting = ii
                assess_experiments(args)
```

## Key hyper-parameter introductions
We list the key hyper-parameters below, including their explanations and available options.

- --dataset: load dataset; the option is ['Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics'] 
- --train_val_test: split the dataset into the training graph, the val graph, the test graph; default = [0.7, 0.2, 0.1]
- --model: load model; the option is ['GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF']
- --hidden_channels: number of models; default = [600, 400, 200]
- --setting: setting I,II,III; the option is [1,2,3]
- --n_wm: number of bits of the watermark string; default = 200
- --model_num: the number of generated watermark models

## How to run
```
python main.py
```

We run all models, datasets, and Settings I, II, and III in a single pass within `main.py`.  
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
