
from watermark.assess import *
from watermark.insight import *
from utils.config import parse_args
from utils.utils import train, test

def assess_experiments(args):
    train_data, val_data, test_data, num_features, num_labels = load_data(args)

    model_name = args.model_path + args.dataset + '/' + args.model
    args.random_seed = int(time.time()*1000)
    torch.manual_seed(args.random_seed)
    if os.path.exists(model_name):
        model = torch.load(model_name, weights_only=False)
    else:
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        model = load_model(num_features, num_labels, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            loss = train(model, train_data, optimizer)
            if epoch % 100 == 0:
                train_acc = test(model, train_data)
                val_acc = test(model, val_data)
                test_acc = test(model, test_data)
                print(f'Original model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
            torch.save(model, model_name)
    print('-----------------------------------------------------------')
    model_w, wm, wmk, trigger, model_i = setting(copy.deepcopy(model), copy.deepcopy(model), train_data, val_data, test_data, args)

    # robust
    assess_pruning(model_w, model_i, train_data, val_data, test_data, wm, wmk, trigger, args)
    assess_fine_tuning(model_w, model_i, train_data, val_data, test_data, wm, wmk, trigger, args)
    assess_overwriting(model_w, model_i, train_data, val_data, test_data, wm, wmk, trigger, args)
    assess_model_extract(model_w, model_i, train_data, val_data, test_data, wm, wmk, trigger, args)
    torch.cuda.empty_cache()



if __name__ == '__main__':
    # our methods
    args = parse_args()
    datasets = ['Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics']  # 'Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics'
    models = ['GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF']  # 'GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF'

    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1, 4):
            args.setting = ii
            assess_experiments(args)

    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1,4):
            args.setting = ii
            assess_insight(args)

    for iii in range(args.model_num):
        for i in range(6):
            args.dataset = datasets[i]
            args.model = models[i]
            for ii in range(1, 4):
                args.setting = ii
                assess_experiments(args)
