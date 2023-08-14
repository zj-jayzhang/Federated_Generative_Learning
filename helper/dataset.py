import random
from helper.utils import setup_seed
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision
import torch

def read_domainnet_data(data_path, domain_name, split="train", labels = None):
    data_paths = []
    data_labels = []

    split_file = os.path.join(data_path, "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            relative_data_path, label = line.split(' ')
            absolute_data_path = os.path.join(data_path, relative_data_path)
            label = int(label)
            if labels is not None:
                if label in labels: 
                    data_paths.append(absolute_data_path)
                    data_labels.append(labels.index(label))
            elif labels is None:
                data_paths.append(absolute_data_path)
                data_labels.append(label)
                
    return data_paths, data_labels

class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)

def get_dataset_domainnet(data_path=None, domain_name=None, if_train=True, labels=None):
    if if_train:
        split = "train"
    else:
        split = "test"
    data_paths, data_labels = read_domainnet_data(data_path, domain_name, split=split, labels=labels)
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    if if_train:
        dataset = DomainNet(data_paths, data_labels, transforms_train, domain_name)
    else:
        dataset = DomainNet(data_paths, data_labels, transforms_test, domain_name)

    return dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # return image, label
        return torch.from_numpy(image).float(), torch.tensor(label)


def load_data(dataset, train_dir, test_dir):
    if 'image' in dataset:
        train_dataset = get_dataset(data_path=train_dir, data_type=dataset, if_syn=False, if_train=True)
        test_dataset = get_dataset(data_path=test_dir, data_type=dataset, if_syn=False, if_train=False)
        if dataset == "imagenet1000":
            y_train = np.array(train_dataset.targets)
            y_test = np.array(test_dataset.targets)
            return None, y_train, None, y_test, train_dataset, test_dataset
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
            X_train, y_train = [], []
            # for data in trainloader:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                X_train.append(inputs.numpy())
                y_train.append(targets.numpy())
            
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            X_test, y_test = [], []
            # for data in trainloader:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                X_test.append(inputs.numpy())
                y_test.append(targets.numpy())
            
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            train_dataset = CustomDataset(X_train, y_train)
            test_dataset = CustomDataset(X_test, y_test)
    else:
        raise NotImplementedError
    return X_train, y_train, X_test, y_test, train_dataset, test_dataset


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)  # 去除数组中的重复数字，并进行排序之后输出。
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(10):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 0

        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, partition, beta=0.4, num_users=5, train_dir=None, test_dir=None):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset, train_dir, test_dir)
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts

class ImageFolderDataset(Dataset):
    # sample ration， number of samples. 
    def __init__(self, root, transform=None, num_samples=None, seed=0):
        setup_seed(seed)
        self.root = root
        self.transform = transform
        images = os.listdir(root)
        images.sort()
        if num_samples is not None:
            self.images = random.sample(images, num_samples)
            print("load syn dataset {} from {}.".format(num_samples, self.root))
        else:
            self.images = images
            print("load syn dataset {} from {}.".format(len(images), self.root))
        # print(self.images[:10])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        label = int(img_path.split("_")[-1].split(".")[0])


        if self.transform:
            img = self.transform(img)

        return img, label

class DatasetSplitMap(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, config_map):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.config_map = config_map

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        id_dict = {value: index for index, value in enumerate(self.config_map)}
        label = id_dict[label]
        return image, label


class Config:
    # "tench", "English springer", "cassette player", "chain saw",
    #                     "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    imagenet100 = [117, 70, 88, 133, 5, 97, 42, 60, 14, 3, 130, 57, 26, 0, 89, 127, 36, 67, 110, 65, 123, 55, 22, 21, 1, 71, 
                    99, 16, 19, 108, 18, 35, 124, 90, 74, 129, 125, 2, 64, 92, 138, 48, 54, 39, 56, 96, 84, 73, 77, 52, 20, 
                    118, 111, 59, 106, 75, 143, 80, 140, 11, 113, 4, 28, 50, 38, 104, 24, 107, 100, 81, 94, 41, 68, 8, 66, 
                    146, 29, 32, 137, 33, 141, 134, 78, 150, 76, 61, 112, 83, 144, 91, 135, 116, 72, 34, 6, 119, 46, 115, 93, 7]
    
    dict = {
        "imagenette" : imagenette,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagesquawk": imagesquawk,
        "imagenet100": imagenet100,
    }

config = Config()

def get_dataset(data_type='imagenete', if_syn=True, if_train=True, 
                data_path=None, sample_data_nums=None, seed=0, if_blip=False, 
                labels = [1, 73, 11, 19, 29, 31, 290, 121, 225, 39],
                domains=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']):
    # get real domainnet dataset
    if data_type == "domainnet" and not if_syn:
        if if_train:
            # get train dataset
            trainset_list = []
            for domain in domains:
                trainset = get_dataset_domainnet(data_path=data_path, domain_name=domain, if_train=True, labels=labels)
                trainset_list.append(trainset)
            dataset = torch.utils.data.ConcatDataset(trainset_list)    
        else:
            # get test dataset
            dataset = {}
            for domain in domains:
                testset = get_dataset_domainnet(data_path=data_path, domain_name=domain, if_train=False, labels=labels)
                dataset[domain] = testset
    # for imagenet-like dataset and domainnet syn_data
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # get instance-level syn_data 
        if if_syn and not if_blip:
            if 'domainnet' in data_type:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
                transform_test = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                    ])
            
            dataset = ImageFolderDataset(
                data_path,
                transform_train,
                sample_data_nums,
                seed=seed)    
        # get real imagenet-like dataset or class-level syn_data
        else:
            if if_train:
                dataset_all =torchvision.datasets.ImageFolder(root=data_path,transform=transform_train)
                print("loading raw imagenet1000 train dataset from {}".format(data_path))
            else:
                dataset_all =torchvision.datasets.ImageFolder(root=data_path,transform=transform_test)
                print("loading raw imagenet1000 test dataset from {}".format(data_path))
            if data_type == "imagenet1000":
                dataset = dataset_all
            else:
                config.img_net_classes = config.dict[data_type]
                indexs = np.squeeze(np.argwhere(np.isin(dataset_all.targets, config.img_net_classes)))
                dataset = DatasetSplitMap(dataset_all, indexs, config.img_net_classes)
    return dataset
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

