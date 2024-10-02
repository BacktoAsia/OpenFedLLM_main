import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module,make_supervised_data_module_clients
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##
# init_vlm_model() sets up the types of the vision tower, language model and tokenizer
# make_supervised_data_module() sets up the dataset and dataloader

# ===== Define the arguments =====
script_args, fed_args, peft_config, model_args, data_args = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args, model_args, data_args)
# print(script_args, fed_args)

## laod lamma 这里需要加入visual encoder
model,tokenizer=init_vlm_model(script_args,model_args, data_args)



# ===== Load the dataset =====
# dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
# dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# 源代码用的是datasets.Dataset包，vlm代码用的是torch.utils.data.Dataset，暂时替换
#0924 here we should load datasets from a series of json files
# eg. client0.json, client1.json, client2.json, client3.json
# dataset = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args) # 这里需要划分数据集
local_datasets, global_test_dataset,data_collator = make_supervised_data_module_clients(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) # 这里需要划分数据集

#Trainer训练时，会将dataset中的数据按照对应的键值传入，因此需要在自己模型的forward方法中接收键值变量。如上例，需要将方法写为：forward(self, x, labels)




# ===== Split the dataset into clients =====
# local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()
    
# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
# tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.unk_token   # following vicuna

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
## only for language model
# formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
# response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
# data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
# 加载数据集

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
   

        # 显示当前模型占用的显存量


    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)


        # ===== Train local model on the client side =====
        # print("laoding trainer....")
        trainer = get_fed_local_vlm_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=local_datasets[client],
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )
      
    #     # 显示当前模型占用的显存量
        print(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
    # break

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))