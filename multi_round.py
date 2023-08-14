import argparse
import copy
import torch
from tqdm import tqdm
from helper.dataset import partition_data, get_dataset, get_dataset_domainnet, DatasetSplit
from helper.utils import setup_seed, test_accuracy, average_weights, local_update
from helper.models import get_model
import numpy as np
import wandb
import os
import warnings
warnings.filterwarnings('ignore')


def get_FL_dataset(args):
    if args.data_type == 'imagenet100':
        num_classes = 100
    elif args.data_type == 'imagenet1000' or args.data_type == 'imagenet1000_syn':
        num_classes = 1000
    else:
        num_classes = 10
        
    # for feature distribution skew 
    if args.data_type == "domainnet":
        trainset_list, testset_list = [], []
        testloader_dict = {}
        trainloader_dict = {}
        for domain in args.domains:
            trainset = get_dataset_domainnet(data_path=args.data_path_train, domain_name=domain, if_train=True, labels=args.labels)
            trainset_list.append(trainset)
            trainloader_dict[domain] = torch.utils.data.DataLoader(
                                    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        for domain in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
            testset = get_dataset_domainnet(data_path=args.data_path_test, domain_name=domain, if_train=False, labels=args.labels)
            testset_list.append(testset)
            testloader_dict[domain] = torch.utils.data.DataLoader(
                                    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return trainloader_dict, testloader_dict, None, None, num_classes
    # for label distribution skew
    elif "image" in args.data_type:
        train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(args.data_type, args.partition, 
            beta=args.beta, num_users=args.num_users, train_dir=args.data_path_train, test_dir=args.data_path_test)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                shuffle=False, num_workers=args.num_workers)

        server_trainloader = None
        if args.sever_ep is not None:
            #todo: get server_trainloader
            # server_trainloader = get_server_trainloader()
            trainset = get_dataset(data_type=args.data_type, if_syn=True, if_train=True, data_path=args.data_path_server, 
                        sample_data_nums=args.sample_data_nums, seed=args.seed-2021)
            server_trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return train_dataset, test_loader, user_groups, server_trainloader, num_classes
    
    else:
        raise ValueError("Not Implemented!")
    
def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Training with federation learning')
    # Training model hyperparameter settings
    parser.add_argument('--com_round', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--sever_ep', type=int, default=None,
                    help="the number of sever epoches: E, if None, not training in server")
    parser.add_argument('--batch_size', type=int, default=128,
                        help=" batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for data loading")

    # federation learning settings
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--domains', nargs='+', type=str, 
                        default=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                        help='in domainnet, number of user: len(domains)')
    # dataset settings
    parser.add_argument('--data_type', type=str, 
                        choices=["domainnet", "imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "imagenet1000"],
                        default="imagenette",
                        help='data set type')
    parser.add_argument('--data_path_train', default=None, type=str, help='data path for train')
    parser.add_argument('--data_path_test', default=None, type=str, help='data path for test')
    parser.add_argument('--data_path_server', default=None, type=str, help='data path for test')
    parser.add_argument('--sample_data_nums', type=int, default=None, 
                    help='if server training, sample number of syn images if None samples all data')
    parser.add_argument('--labels', nargs='+', type=int, 
                        default=[1, 73, 11, 19, 29, 31, 290, 121, 225, 39], #['airplane', 'clock', 'axe', 'basketball', 'bicycle', 'bird', 'strawberry', 'flower', 'pizza', 'bracelet'],
                        help='domainnet subdataset labels')
    parser.add_argument('--partition', default='dirichlet', type=str,
                        help='dirichlet or iid')
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                            'then the partition is more unbalanced')
    # model settings
    parser.add_argument('--net', type=str, 
                        choices=['holocron_resnet18', 'holocron_resnet34', "holocron_resnet50", "resnet50"],
                        default="holocron_resnet18",
                        help='model name to train')
    parser.add_argument('--net_path', type=str, default=None,
                        help='load model weight path')
    # others settings
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--wandb', type=int, default=1,
                    help="set 1 for wandb logging")
    parser.add_argument('--exp_name', type=str, default="exp_1",
                        help="the name of this run")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_parser()
    setup_seed(args.seed)
    experiment_name = "fl-{}-{}-{}".format(args.data_type, args.net, args.exp_name)
    save_model_dir = os.path.join("checkpoints", experiment_name)

    # set wandb
    if args.wandb == 1:
        wandb.init(config=args, project="FGL", group="multi-round FL", name=experiment_name)

    # get dataset
    # import pdb; pdb.set_trace()
    train_dataset, test_loader, user_groups, server_trainloader, num_classes = get_FL_dataset(args)
    global_model = get_model(net_type=args.net, net_weight_path=args.net_path, num_classes=num_classes) 
    global_model.to("cuda")
    if args.data_type == 'domainnet':
        bst_acc_domain = dict()
    bst_acc = -1
    global_model.train()

    # =========== Federated Learning ===============
    for com in tqdm(range(args.com_round)):
        local_weights = []
        if args.data_type == 'domainnet':
            for domain in args.domains:
                w = local_update(copy.deepcopy(global_model), train_dataset[domain], n_epochs=args.local_ep, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, if_log=True)
                local_weights.append(copy.deepcopy(w))
        elif "image" in args.data_type:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                trainloader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                w = local_update(copy.deepcopy(global_model), trainloader, n_epochs=args.local_ep, lr=args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay, if_log=True)
                local_weights.append(copy.deepcopy(w))
        else:
            raise ValueError("Not Implemented!")
    
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        # test model accuracy
        if args.data_type == 'domainnet':
            correct_all, total_all = 0, 0
            test_acc_domain = dict()
            for domain in args.domains:
                correct, total, test_loss = test_accuracy(global_model, test_loader[domain], if_log=True)
                acc = round(100.0 * correct/total, 2)
                correct_all += correct
                total_all += total
                test_acc_domain[domain] = acc
                if args.wandb == 1:
                    wandb.log({"test_{}_loss".format(domain): test_loss/args.batch_size})
                    wandb.log({"test_{}_acc".format(domain):acc})
            test_acc = 100.0 * correct_all / total_all
            if args.wandb == 1:
                wandb.log({"test_acc":test_acc})
            bst_acc = max(bst_acc, test_acc)
            if test_acc > bst_acc:
                bst_acc = test_acc
                bst_acc_domain = test_acc_domain
            print("The {}-th communication round, test acc:{}, bst_acc={}, domain_acc={}".format(com, test_acc, bst_acc, test_acc_domain))
        elif "image" in args.data_type:
            correct, total, test_loss = test_accuracy(global_model, test_loader, if_log=True)
            test_acc = round(100.0 * correct/total, 2)
            is_best = test_acc > bst_acc
            bst_acc = max(bst_acc, test_acc)
            if args.wandb == 1:
                wandb.log({"test_loss": test_loss})
                wandb.log({"test_acc":test_acc})
                wandb.log({"bst_acc":bst_acc})
            print("The {}-th communication round, test acc:{}, bst_acc={}".format(com, test_acc, bst_acc))
            if args.sever_ep is not None:

                local_update(global_model, server_trainloader, n_epochs=args.sever_ep, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, if_log=True)
                correct, total, test_loss = test_accuracy(global_model, test_loader, if_log=True)
                test_acc = round(100.0 * correct/total, 2)
                is_best = test_acc > bst_acc
                bst_acc = max(bst_acc, test_acc)
                if args.wandb == 1:
                    wandb.log({"test_loss": test_loss})
                    wandb.log({"test_acc":test_acc})
                    wandb.log({"bst_acc":bst_acc})
                print("The {}-th communication round, finetune at sever-side, test acc:{}, bst_acc={}".format(com, test_acc, bst_acc))
        else:
            raise ValueError("Not Implemented!")
