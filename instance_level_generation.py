import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def args_parser():
    parser = argparse.ArgumentParser(description='instance level generation')
    parser.add_argument('--raw_data_dir', type=str, default="./raw_data_demo", help="to/your/raw_data_dir/path")
    parser.add_argument('--syn_data_dir', type=str, default="./syn_data_demo", help="to/your/syn_data_dir/path")
    parser.add_argument('--blip_batch_size', type=int, default=256, help="the batch size used for generating captions")
    parser.add_argument('--batch_size', type=int, default=8, help="the batch size used for generating syn img data.")
    parser.add_argument('--prompt_dir', type=str, default=None, help='If None, captions are not saved; if not empty, captions are saved to a text file.')
    parser.add_argument('--save_prompt_dir', type=str, default="./captions_demo", help='If None, captions are not saved; if not empty, captions are saved to a text file.')
    parser.add_argument('--img1k_label', type=str, default="./helper/img1k_label.txt", help='to/your/img1k_label_file/path')
    args = parser.parse_args()
    return args

def get_captions(args):
    captions = {} 
    # if already saved, just return
    if args.prompt_dir is not None:
        prompt_file_names = os.listdir(args.prompt_dir)
        prompt_file_names.sort()
        for prompt_file_name in prompt_file_names:
            prompt_file_key = prompt_file_name.split(".")[0]
            captions[prompt_file_key] = []
            with open(os.path.join(args.prompt_dir, prompt_file_name), 'r') as fr:
                lines = fr.readlines()
                for line in lines:
                    line = line.strip("\n")
                    img_name, caption = line.split("\t")
                    captions[prompt_file_key].append((img_name, caption)) 
    # else use the blip model to generate captions
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # get blip-v2 model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model_blipv2 = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
                    )
        model_blipv2.to(device)
        # get captions
        raw_data_dir_names = os.listdir(args.raw_data_dir)
        raw_data_dir_names.sort()
        for raw_data_dir_name in raw_data_dir_names:
            captions[raw_data_dir_name] = []
            raw_data_dir_path = os.path.join(args.raw_data_dir, raw_data_dir_name)
            img_names = os.listdir(raw_data_dir_path)
            img_names.sort()
            # generate captions
            for batch_left_index in range(0, len(img_names), args.blip_batch_size):
                batch_right_index = min(batch_left_index + args.blip_batch_size, len(img_names))
                batch_imgs = []
                for img_index in range(batch_left_index, batch_right_index):
                    img_path = os.path.join(args.raw_data_dir, raw_data_dir_name, img_names[img_index])
                    img = Image.open(img_path)
                    batch_imgs.append(img)
                batch_inputs = processor(images=batch_imgs, return_tensors="pt").to(device, torch.float16)
                batch_generated_ids = model_blipv2.generate(**batch_inputs)
                batch_generated_texts = processor.batch_decode(batch_generated_ids, skip_special_tokens=True)
                for tmp_index in range(len(batch_generated_texts)):
                    captions[raw_data_dir_name].append((img_names[batch_left_index + tmp_index], batch_generated_texts[tmp_index].strip()))
    # save captions
    if args.prompt_dir is None and args.save_prompt_dir is not None:
        if not os.path.exists(args.save_prompt_dir):
            os.makedirs(args.save_prompt_dir)
            
        for raw_data_dir_name in captions:
            img_caption_pairs = captions[raw_data_dir_name]
            img_caption_pairs_file_path = os.path.join(args.save_prompt_dir, raw_data_dir_name + ".txt")
            print("write caption for dir name :{}, to {}.".format(raw_data_dir_name, img_caption_pairs_file_path))
            with open(img_caption_pairs_file_path, 'w') as fw:
                for (img_name, caption) in img_caption_pairs:
                    fw.write(img_name + "\t" + caption + "\n")
                    fw.flush()
                    
    return captions

def generation_syn_data(args, captions, label):
    gen_data_seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get stable-diffusion model
    model_id = "stabilityai/stable-diffusion-2-1-base"
    model_sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, 
                                                    requires_safety_checker = False, safety_checker=None,
                                                    local_files_only=True,
                                                    )
    model_sd.to(device)

    if not os.path.exists(args.syn_data_dir):
        os.makedirs(args.syn_data_dir)
    # generation data
    for raw_data_dir_name in captions:
        img_caption_pairs = captions[raw_data_dir_name]
        if not os.path.exists(os.path.join(args.syn_data_dir, raw_data_dir_name)):
            os.makedirs(os.path.join(args.syn_data_dir, raw_data_dir_name))
        for batch_left_index in range(0, len(img_caption_pairs), args.batch_size):
            batch_right_index = min(batch_left_index + args.batch_size, len(img_caption_pairs))
            
            positive_prompts = [label[raw_data_dir_name] + ", " + img_caption_pairs[index][1]  + ", real world images, high resolution." for index in range(batch_left_index, batch_right_index)]
            negative_prompt = ["low quality, low resolution" for i in range(len(positive_prompts))]
            images = model_sd(
                prompt = positive_prompts,
                height = 512,
                width = 512,
                num_inference_steps = 50,
                guidance_scale = 2,
                negative_prompt = negative_prompt, 
                num_images_per_prompt = 1,
                generator = torch.Generator().manual_seed(gen_data_seed),
            )
            gen_data_seed += 1
            img_names = [img_caption_pairs[index][0] for index in range(batch_left_index, batch_right_index)]
            # save images
            for i, image in enumerate(images.images):
                syn_img_path = os.path.join(args.syn_data_dir, raw_data_dir_name, img_names[i])
                image.save(syn_img_path)
    
def get_img1k_label(args):
    label = {}
    with open(args.img1k_label, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip("\n")
            label_id, label_name = line[:9], line[10:]
            label[label_id] = label_name
    return label


if __name__ == '__main__':
    args = args_parser()
    
    # get imagenet 1k label
    label = get_img1k_label(args)

    # get instance-level caption
    captions = get_captions(args)
    # import pdb;pdb.set_trace()
    # generate synthetic data via saved captions
    generation_syn_data(args, captions, label)
    
