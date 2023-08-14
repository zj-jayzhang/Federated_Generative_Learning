import os
from tqdm import tqdm
from helper.utils import setup_seed
import argparse
from diffusers import StableDiffusionPipeline
import torch


def args_parser():
    parser = argparse.ArgumentParser(description='generation syn data')
    parser.add_argument('--data_type', type=str, 
                            choices=["imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "domainnet"],
                            default="domainnet", help="generation dataset type")
    parser.add_argument('--per_class_nums', type=int, default=400, help="number of training epochs")
    parser.add_argument('--seed', type=int, default=0, help="random seed for reproducibility")
    parser.add_argument('--exp_name', type=str, default="",
                        help="the name of this experiment")
    parser.add_argument('--start_number', type=int, default=0,
                        help="start number of generated images")
    args = parser.parse_args()
    return args


def gen_data_imagenet(pipe, syn_data_dir, start_num, per_class_nums, data_type):
    if data_type == "imagenet100":
        label_list = ["chambered nautilus, pearly nautilus, nautilus",
                        "harvestman, daddy longlegs, Phalangium opilio",
                        "macaw",
                        "bittern",
                        "electric ray, crampfish, numbfish, torpedo",
                        "birds:drake",
                        "agama",
                        "night snake, Hypsiglena torquata",
                        "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
                        "tiger shark, Galeocerdo cuvieri",
                        "flamingo",
                        "garter snake, grass snake",
                        "common newt, Triturus vulgaris",
                        "tench, Tinca tinca",
                        "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
                        "white stork, Ciconia ciconia",
                        "terrapin",
                        "diamondback, diamondback rattlesnake, Crotalus adamanteus",
                        "flatworm, platyhelminth",
                        "sea snake",
                        "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
                        "green snake, grass snake",
                        "bald eagle, American eagle, Haliaeetus leucocephalus",
                        "kite, bird of prey in the family Accipitridae",
                        "goldfish, Carassius auratus",
                        "scorpion",
                        "goose",
                        "bulbul",
                        "chickadee",
                        "sea anemone, anemone",
                        "magpie",
                        "mud turtle",
                        "crayfish, crawfish, crawdad, crawdaddy",
                        "lorikeet",
                        "garden spider, Aranea diademata",
                        "spoonbill",
                        "hermit crab",
                        "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
                        "green mamba",
                        "bee eater",
                        "bustard",
                        "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
                        "hognose snake, puff adder, sand viper",
                        "common iguana, iguana, Iguana iguana",
                        "king snake, kingsnake",
                        "toucan",
                        "peacock",
                        "barn spider, Araneus cavaticus",
                        "wolf spider, hunting spider",
                        "thunder snake, worm snake, Carphophis amoenus",
                        "water ouzel, dipper",
                        "Dungeness crab, Cancer magister",
                        "nematode, nematode worm, roundworm",
                        "vine snake",
                        "wombat",
                        "black widow, Latrodectus mactans",
                        "oystercatcher, oyster catcher",
                        "black grouse",
                        "red-backed sandpiper, dunlin, Erolia alpina",
                        "goldfinch, Carduelis carduelis",
                        "snail",
                        "hammerhead, hammerhead shark",
                        "spotted salamander, Ambystoma maculatum",
                        "American alligator, Alligator mississipiensis",
                        "banded gecko",
                        "wallaby, brush kangaroo",
                        "great grey owl, great gray owl, Strix nebulosa",
                        "jellyfish",
                        "black swan, Cygnus atratus",
                        "ptarmigan",
                        "hummingbird",
                        "whiptail, whiptail lizard",
                        "sidewinder, horned rattlesnake, Crotalus cerastes",
                        "hen",
                        "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
                        "albatross, mollymawk",
                        "axolotl, mud puppy, Ambystoma mexicanum",
                        "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
                        "American coot, marsh hen, mud hen, water hen, Fulica americana",
                        "loggerhead, loggerhead turtle, Caretta caretta",
                        "redshank, Tringa totanus",
                        "crane",
                        "tick",
                        "sea lion",
                        "tarantula",
                        "boa constrictor, Constrictor constrictor",
                        "conch",
                        "prairie chicken, prairie grouse, prairie fowl",
                        "pelican",
                        "coucal",
                        "limpkin, Aramus pictus",
                        "chiton, coat-of-mail shell, sea cradle, polyplacophore",
                        "black and gold garden spider, Argiope aurantia",
                        "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
                        "stingray",
                        "rock crab, Cancer irroratus",
                        "green lizard, Lacerta viridis",
                        "sea slug, nudibranch",
                        "hornbill",
                        "cock"]
        labels = [i for i in range(100)]
    elif data_type == "imagenette":
        label_list = ["tench", "English springer", "cassette player", "chain saw",
                        "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"] 
        labels = [i for i in range(10)]
    elif data_type == "imagefruit":
        label_list = ["pineapple, ananas",
                        "banana",
                        "strawberry",
                        "fruit: orange",
                        "lemon",
                        "pomegranate",
                        "fig",
                        "bell pepper",
                        "cucumber, cuke",
                        "Granny Smith"]
        labels = [i for i in range(10)]
    elif data_type == "imagesquawk":
        label_list = ["peacock", 
                        "flamingo",
                        "macaw",
                        "pelican",
                        "king penguin, Aptenodytes patagonica", 
                        "bald eagle, American eagle, Haliaeetus leucocephalus",
                        "toucan",
                        "ostrich",
                        "black swan, Cygnus atratus",
                        "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita"
                        ]
        labels = [i for i in range(10)]
    elif data_type == "imageyellow":
        label_list = ["bee", 
                        "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 
                        "banana",
                        "lemon", 
                        "corn", 
                        "school bus", 
                        "honeycomb", 
                        "lion, king of beasts, Panthera leo", 
                        "black and gold garden spider, Argiope aurantia", 
                        "goldfinch, Carduelis carduelis"]
        labels = [i for i in range(10)]
    else: 
        raise ValueError(f"Unknown data_type {data_type}")
    
    prompts = [label + ", real world images, high resolution " for label in label_list]
    img_size = 512
    negative_prompt = ["low quality, low resolution" for i in range(len(label_list))]

    for epoch in tqdm(range(per_class_nums)):
        num = 0
        # generate 10 images per iteration
        for i in range(int(len(label_list)/10)):   
            images = pipe(
                prompt = prompts[num:num+10],
                height = img_size,
                width = img_size,
                num_inference_steps = 50,
                guidance_scale = 2,
                negative_prompt = negative_prompt[num:num+10], 
                num_images_per_prompt = 1,
                generator = torch.Generator().manual_seed(start_num+num),
            )
            # save images
            for i, image in enumerate(images.images):
                image = image.resize((256, 256))
                num_id = '{:09d}_{}'.format(num+i+start_num, labels[num+i])
                image.save(f"{syn_data_dir}/{num_id}.png")
            num += 10
        start_num += num
    return start_num

def gen_data_domainnet(pipe, syn_data_dir, start_num, per_class_nums):
    domain_list = ['Sketch drawing with only one object in the picture', 
                    'real world images, high resolution, only one object in the picture', 
                    "a black and white drawing of ",
                    'painting with only one object in the picture, high resolution',
                    "infograph images with only one object in the picture, including text descriptions of the object's components.",
                    'clippart, cartoon style, single object.']
    img_size = 512
    for epoch in tqdm(range(per_class_nums)):
        for style in domain_list:
            # generation quickdraw domain
            if style[0]== 'a':
                label_list = ['an airplane', 'a clock', 'an axe', 'a basketball', 'a bicycle', 'a bird', 'a strawberry', 'a flower', 'a pizza', 'a bracelet']
                prompts = ["{}{}".format(style, label) for label in label_list]
                negative_prompt = ["colorful, painting" for i in range(len(label_list))]
            # generation others domain
            else:
                label_list = ['airplane', 'clock', 'axe', 'basketball', 'bicycle', 'bird', 'strawberry', 'flower', 'pizza', 'bracelet']
                prompts = ["{}, {}".format(label, style) for label in label_list]
                negative_prompt = ["low quality, low resolution" for i in range(len(label_list))]
            images = pipe(
                prompt=prompts,
                height=img_size,
                width=img_size,
                num_inference_steps=50,
                guidance_scale=2,
                negative_prompt = negative_prompt, 
                num_images_per_prompt = 1,
                generator = torch.Generator().manual_seed(start_num)
            )

            # save images
            for i, image in enumerate(images.images):
                num_id = '{:09d}_{}_{}'.format(i+start_num, domain_list.index(style), i)
                image.save(f"{syn_data_dir}/{num_id}.png")
            start_num += len(label_list)
            print(f"dataset size: {start_num}")

if __name__ == "__main__":
    args = args_parser()
    setup_seed(int(args.seed))
    model_id = "stabilityai/stable-diffusion-2-1-base"
    syn_data_dir = os.path.join("syn_images", "syn_" + args.data_type + "_" + args.exp_name)
    if not os.path.exists(syn_data_dir):
        os.makedirs(syn_data_dir)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, 
                                                    requires_safety_checker = False, safety_checker=None,
                                                    local_files_only=True,
                                                    )
    pipe.to("cuda")
    if args.data_type == "domainnet":
        gen_data_domainnet(pipe=pipe, 
                            syn_data_dir=syn_data_dir, 
                            start_num=args.start_number,
                            per_class_nums=args.per_class_nums)
        
    elif "image" in args.data_type:
        gen_data_imagenet(pipe=pipe, 
                        syn_data_dir=syn_data_dir, 
                        start_num=args.start_number, 
                        per_class_nums=args.per_class_nums,
                        data_type=args.data_type)
    else:
        raise ValueError("Not Implemented!")
        