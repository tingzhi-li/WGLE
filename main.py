from watermark.assess import *
from watermark.insight import *
from utils.config import parse_args
from utils.utils import train, test

args = parse_args()

def assess_experiments():
    data = load_data(args)
    if args.paradigm == 'transductive':
        num_features = data.num_features
        num_classes = data.y.max().item() + 1
    elif args.paradigm == 'inductive':
        num_features = data[0].num_features
        num_classes = data[0].y.max().item() + 1
    else:
        raise ValueError('Error: Wrong paradigm!')

    model_name = args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm
    if os.path.exists(model_name):
        model_o = torch.load(model_name, weights_only=False)
    else:
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        model_o = load_model(num_features, num_classes, args)
        optimizer = torch.optim.Adam(model_o.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
        for epoch in range(args.epochs):
            loss = train(model_o, data, optimizer, args)
            scheduler.step()
            if epoch % 100 == 0:
                train_acc, test_acc = test(model_o, data, args)
                print(f'Original model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        torch.save(model_o, model_name)
    print('-----------------------------------------------------------')

    model_i = load_model(num_features, num_classes, args)
    optimizer = torch.optim.Adam(model_i.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
    for epoch in range(args.epochs):
        loss = train(model_i, data, optimizer, args)
        scheduler.step()
        train_acc, test_acc = test(model_i, data, args)
        if epoch % 100 == 0:
            print(f'Independently model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    
    model_w, wm, wmk, trigger, model_i = setting(copy.deepcopy(model_o), copy.deepcopy(model_i), data, args)

    # robust
    assess_pruning(model_w, model_i, data, wm, wmk, trigger, args)
    assess_fine_tuning(model_w, model_i, data, wm, wmk, trigger, args)
    del model_i

    #assess_unlearning(model_w, data, wm, wmk, trigger, args)
    assess_overwriting(model_w, data, wm, wmk, trigger, args)
    assess_model_extract(model_w, data, wm, wmk, trigger, args)
    torch.cuda.empty_cache()



if __name__ == '__main__':
    # our methods
    datasets = ['Cora', 'DBLP', 'CS', 'Physics', 'Blog', 'Photo']  # 'Cora', 'DBLP', 'CS', 'Physics', 'Blog', 'Flickr'
    models = ['GAT', 'GTF', 'SSG', 'GCNv2', 'ARMA', 'SAGE']  # 'GAT', 'GTF', 'SSG', 'GCNv2', 'ARMA', 'SAGE'

    args.paradigm = 'inductive'
    for i in range(1,2):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1, 3):
            args.setting = ii
            # assess_experiments()
            multibit(args)

    args.paradigm = 'transductive'
    for i in range(1, 2):
        args.dataset = datasets[i]
        args.model = models[i]
        for ii in range(1, 3):
            args.setting = ii
            #assess_experiments()
            multibit(args)


    # for i in range(6):
    #     args.dataset = datasets[i]
    #     args.model = models[i]
    #     for ii in range(1,4):
    #         args.setting = ii
    #         assess_insight(args)
    #
    # for iii in range(args.model_num):
    #     for i in range(6):
    #         args.dataset = datasets[i]
    #         args.model = models[i]
    #         for ii in range(1, 4):
    #             args.setting = ii
    #             assess_experiments(args)
