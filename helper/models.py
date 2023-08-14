
import torch
from holocron.models.classification import resnet34 as holocron_resnet34
from holocron.models.classification import resnet18 as holocron_resnet18
from holocron.models.classification import resnet18 as holocron_resnet50
import torchvision

def get_model(net_type, net_weight_path=None, num_classes=10):
    # Model
    print('==> Building model from {}..'.format(net_type))
    if "resnet18" in net_type:
        net = holocron_resnet18(num_classes=num_classes)
    elif "resnet34" in net_type:
        net = holocron_resnet34(num_classes=num_classes)
    elif "resnet50" in net_type:
        net = torchvision.models.get_model("resnet50", weights=None, num_classes=num_classes)
        # net = holocron_resnet50(num_classes=num_classes)
    else:
        raise ValueError("Unknown model value of : {}".format(net_type))

    if net_weight_path is not None and "resnet50" in net_type:
        print("loading  weight from {}".format(net_weight_path))
        checkpoint = torch.load(net_weight_path, map_location="cpu")
        net.load_state_dict(checkpoint["model"])
    elif net_weight_path is not None:
        print("loading  weight from {}".format(net_weight_path))
        net.load_state_dict(torch.load(net_weight_path))   
    else:
        print("Random initialization of model weights.")

    return net