import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import socket

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


from PIL import Image
import requests
import time

# Configuration
POLLING_INTERVAL = 60  # in seconds
MIDDLEMAN_SERVER_API_URL = 'https://middleman.server/api/data'
AUTH_TOKEN = 'your_auth_token_here'  # Replace with your actual token

# Headers for the HTTP request
HEADERS = {
    'Authorization': f'Bearer {AUTH_TOKEN}',
    'Content-Type': 'application/json'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True,
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(
    vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(
    device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList(
    [StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(
    args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')
chat_state = CONV_VISION.copy()

print(chat_state)

# ========================================
#             Gradio Setting
# ========================================


def reset(img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return img_list


def upload_img(img):
    if img is None:
        return chat_state, None

    img_list = []
    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)
    return img_list


def ask(user_message):
    if len(user_message) == 0:
        return
    chat.ask(user_message, chat_state)
    return


def answer(img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    return img_list, llm_message


def menu():
    while True:
        print("1. Upload Image")
        print("2. Ask")
        print("3. Reset")
        ui = input("enter your choice")
        if (int(ui) == 1):
            img = get_image()
            img_list = upload_img(img)
            print(type(img_list[0]))
        elif (int(ui) == 2):
            text = input("enter your text")
            ask(text)
            img_list, llm_message = answer(
                img_list, num_beams=1, temperature=1.0)
            print(llm_message)
        elif (int(ui) == 3):
            img_list = reset(img_list)
        else:
            return


def get_image():
    print(os.curdir)
    img_path = input(
        "Please enter the name of the image you want to talk about (in the test_image folder)")

    for k in os.listdir('image_folder'):
        if k == img_path:
            return Image.open('./image_folder/'+k)


def server():
    host = '158.132.255.33'
    port = 320
    s = socket.socket()
    s.bind((host, port))

    s.listen(1)
    c, addr = s.accept()
    while True:
        data = c.recv(1024)
        if not data:
            break
        name = str(data.name).upper()
        if name == "RESET":
            img_list = reset(img_list)
        elif name == "ASK":
            img_list, llm_message = answer(
                img_list, num_beams=1, temperature=1.0)
            c.send(llm_message)
        elif name == "UPLOAD":
            img_list = upload_img(data)
            print(type(img_list[0]))
    c.close()


if __name__ == '__main__':
    server()

menu()



def poll_middleman_server(api_url):
    try:
        # Perform the GET request to the middleman server's API
        response = requests.get(api_url, headers=HEADERS)

        # Check if the response is successful
        if response.status_code == 200:
            # Assuming the server returns JSON data
            data = response.json()
            return data

        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def process_data(data):
    # Placeholder for processing logic
    # You would add your processing code here
    print("Processing received data:", data)
    while True:
        if not data:
            break
        name = str(data.name).upper()
        if name == "RESET":
            img_list = reset(img_list)
        elif name == "ASK":
            img_list, llm_message = answer(
                img_list, num_beams=1, temperature=1.0)
            #send(llm_message)
        elif name == "UPLOAD":
            img_list = upload_img(data)
            print(type(img_list[0]))

def main():
    while True:
        print("Polling the middleman server for new data...")
        data = poll_middleman_server(MIDDLEMAN_SERVER_API_URL)

        if data:
            process_data(data)

        # Wait for the specified polling interval before checking again
        time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    main()