import os
import argparse
import wandb
import torch
import torch.optim as optim
from helper.utils import setup_seed, train, test_accuracy
from helper.dataset import get_dataset
from helper.models import get_model
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='One-shot training')
    # Training model hyperparameter settings
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, default=120, help="number of training epochs")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--weight_decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for data loading")
    # model setting
    parser.add_argument('--net', type=str, 
                        choices=['holocron_resnet18', 'holocron_resnet34', 'holocron_resnet50', "resnet50"],
                        default="holocron_resnet18",
                        help='model name to train')
    parser.add_argument('--net_path', type=str, 
                        default=None,
                        help='load model weight path')
    # dataset setting 
    parser.add_argument('--data_type', type=str, 
                        choices=["imagenet1000", "domainnet", "imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk"],
                        default="imagenette",
                        help='data set type')
    parser.add_argument('--data_path_train', default=None, type=str, help='data path for train')
    parser.add_argument('--data_path_test', default=None, type=str, help='data path for test')
    parser.add_argument('--sample_data_nums', default=None, type=int, help='sample number of syn images if None samples all data')
    parser.add_argument('--syn', type=int, choices=[0, 1], default=0, help='if syn dataset')
    parser.add_argument('--if_blip', type=int, choices=[0, 1], default=0, help='if use instance-level syn data')
    # domainnet dataset setting
    parser.add_argument('--labels', nargs='+', type=int, 
                        default=[1, 73, 11, 19, 29, 31, 290, 121, 225, 39], #['airplane', 'clock', 'axe', 'basketball', 'bicycle', 'bird', 'strawberry', 'flower', 'pizza', 'bracelet'],
                        help='domainnet subdataset labels')
    parser.add_argument('--domains', nargs='+', type=str, 
                        default=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                        help='domainent domain')
    # others setting
    parser.add_argument('--seed', type=int, default=0, help="random seed for reproducibility")
    parser.add_argument('--exp_name', type=str, default="exp_1",
                        help="the name of this run")
    parser.add_argument('--wandb', type=int, default=1,
                        help="set 1 for wandb logging")
    args = parser.parse_args()
    # post processing
    args.syn = (args.syn==1)
    args.if_blip = (args.if_blip==1)
    return args


if __name__ == '__main__':
    # get args
    args = get_arguments()
    # mkdir to save models
    save_tmp = "{}-{}-{}".format(args.net, args.data_type, args.exp_name)
    if args.syn:
        save_tmp = save_tmp + "-syn"
    save_model_dir = os.path.join("checkpoints", save_tmp)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    setup_seed(args.seed)

    # wandb
    if args.wandb == 1:
        wandb.init(config=args, project="FGL", group="one-shot FL", name=save_tmp)

    # getdataset
    trainset = get_dataset(data_type=args.data_type, 
                            if_syn=args.syn, 
                            if_train=True, 
                            data_path=args.data_path_train, 
                            sample_data_nums=args.sample_data_nums, 
                            seed=args.seed, 
                            if_blip=args.if_blip, 
                            labels=args.labels,
                            domains=args.domains)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = get_dataset(data_type=args.data_type, 
                            if_syn=False,
                            if_train=False, 
                            data_path=args.data_path_test,
                            sample_data_nums=None,
                            seed=args.seed,
                            if_blip=False,
                            labels=args.labels,
                            domains=args.domains)

    if args.data_type == "domainnet":
        testloader = {}
        for domain in testset:
            testloader[domain] = torch.utils.data.DataLoader(testset[domain], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # get model
    if "imagenet1000" in args.data_type:
        net = get_model(args.net, args.net_path, num_classes=1000)
    elif "imagenet100" in args.data_type:
        net = get_model(args.net, args.net_path, num_classes=100)
    else:   
        net = get_model(args.net, args.net_path, num_classes=10)    
    net.to('cuda')
    
    # train
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0
    best_acc_domain = {}
    for epoch in tqdm(range(args.epochs)):
        # train
        train(net, trainloader, optimizer, if_log=True)

        # test
        if args.data_type == "domainnet":
            correct_all, total_all = 0, 0
            acc_domain = dict()
            for domain in args.domains:
                correct, total, test_loss = test_accuracy(net, testloader[domain])
                correct_all += correct
                total_all += total
                acc_domain[domain] = round(100.0 * correct / total, 2)
                if args.wandb == 1:
                    wandb.log({"test_{}_loss".format(domain): test_loss/args.batch_size})
                    wandb.log({"test_{}_acc".format(domain): acc_domain[domain]})
                    
            acc = 100.0 * correct_all / total_all
            if args.wandb == 1:
                wandb.log({"test_acc":acc})
                
            if acc > best_acc:
                best_acc = acc
                best_acc_domain = acc_domain
                torch.save(net.state_dict(),
                    os.path.join(save_model_dir, 'model-best-epoch-best.pt'))

            print("epoch/max_epoch:{}/{}  test_loss:{}  acc_domain/best_acc_domain:{}/{} \n test_acc/best_acc:{}/{}"
                    .format(epoch, args.epochs, round(test_loss, 2), acc_domain, best_acc_domain, round(acc, 2), round(best_acc,2 )))
        else:
            correct, total, test_loss = test_accuracy(net, testloader, if_log=True)
            acc = 100.0 * correct / total
            if args.wandb == 1:
                wandb.log({"test_loss": test_loss/args.batch_size})
                wandb.log({"test_acc":acc})

            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(),
                    os.path.join(save_model_dir, 'model-best-epoch-best.pt'))
            print("epoch/max_epoch:{}/{} test_loss:{} \n test_acc/best_acc:{}/{}"
                    .format(epoch, args.epochs, round(test_loss, 2), round(acc, 2), round(best_acc, 2)))
        