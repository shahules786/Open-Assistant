import torch
from transformers import (
    MT5ForConditionalGeneration, 
    T5ForConditionalGeneration, 
    AutoTokenizer, 
    EvalPrediction,
    DataCollator,
    Trainer,
    TrainingArguments,
    get_scheduler)
from datasets import load_dataset,concatenate_datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import wandb
import os
from transformers import ConversationalPipeline
from torch.optim import AdamW

jobid = os.environ.get("SLURM_JOB_ID","123")
ROOT_DIR = os.path.join("/scratch/c.scmse/safety",jobid)

# LABEL2ID = {
#     "__casual__": "0",
#     "__needs_caution__": "1",
#     "__needs_intervention__": "2",
#     "__probably_needs_caution__": "3",
#     "__possibly_needs_caution__": "4",
# }

LABEL2ID = {
    "__casual__": "__casual__",
    "__needs_caution__": "__needs_caution__",
    "__needs_intervention__": "__needs_intervention__",
    "__probably_needs_caution__": "__probably_needs_caution__",
    "__possibly_needs_caution__": "__possibly_needs_caution__",
}

SPECIAL_TOKENS = {"context_token":"<ctx>","sep_token":"<sep>","label_token":"<cls>","rot_token":"<rot>"}

wandb_key = json.load(open("/home/c.scmse/credentials/wandb.json" ))["key"]

wandb.login(key=wandb_key)


CONFIG = {"special_tokens":SPECIAL_TOKENS,
"model":"t5-base",
"max_len":256,
"gradient_accumulation_steps":4,
"epochs":1,
"batch_size":8,
"fp16":False,
"train_dataset":"allenai/prosocial-dialog",
"Notes":"MT5 using prosocial",
"train_dataset":{
    "allenai/prosocial-dialog":["train","validation"],
    "shahules786/prosocial-confessions":"train",

},
"test_dataset":{"allenai/prosocial-dialog":"test"},
"learning_rate":1e-4,
"scheduler":"cosine",
"warmup":0.1,
}

def add_special_tokens(tokenizer,model):
    for key,value in SPECIAL_TOKENS.items():
        setattr(tokenizer,key,value)
        tokenizer.add_tokens([value])
        setattr(tokenizer,key+"_id",tokenizer.encode(value)[0])

    model.resize_token_embeddings(len(tokenizer))


class SafetyDataset(Dataset):
    
    def __init__(self,dataset,split,tokenizer,max_len=512):
        
        super().__init__()

        if isinstance(split,List):
            self.split = "-".join(split)
            self.dataset = concatenate_datasets([dataset[sp] for sp in split])
        else:
            self.split = split
            self.dataset = dataset[split]

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2id = LABEL2ID
        
        
    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self,idx):
        
        
        idx_start = idx
        end = self.dataset[max(0, idx_start - 1)]["episode_done"]
        while (not end) and (idx_start > 0):
            end = self.dataset[max(0, idx_start - 2)]["episode_done"]
            idx_start -= 1
        idx_start = max(0, idx_start)
        context = [f'\nUser: {self.dataset[i]["context"]}\n bot:{self.dataset[i]["response"]}' for i in range(idx_start, idx)]
        context = self.tokenizer.sep_token.join(context)
        rots = self.dataset[idx]["rots"]
        label = self.label2id[self.dataset[idx]["safety_label"]]
        input_tokens = self.tokenizer.encode(self.dataset[idx]["context"],add_special_tokens=False)
        max_len = self.max_len - (len(input_tokens)+2)
        context = self.tokenizer.encode(context,
                                add_special_tokens=False,
                               max_length=max_len,
                               )
        rots = self.tokenizer.sep_token.join(rots)
        input_ids = input_tokens + [self.tokenizer.context_token_id] + context + [self.tokenizer.eos_token_id]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * max(0,(self.max_len - len(input_ids)))
        mask = [1]*len(input_ids) + [self.tokenizer.pad_token_id] * (self.max_len-len(input_ids))
        target_text = self.tokenizer.label_token + label + self.tokenizer.context_token + rots
        decoder_ids = self.tokenizer(target_text,
                                add_special_tokens=True,
                               max_length=self.max_len,
                               padding='max_length',
                               )
        
        return {
            "input_ids":torch.LongTensor(input_ids),
            "attention_mask":torch.LongTensor(mask),
            "decoder_input_ids":torch.LongTensor(decoder_ids["input_ids"]),
            "decoder_attention_mask":torch.LongTensor(decoder_ids["attention_mask"]),
        }
        
        
        
# This dataclass implementation is taken from Suraj Patil: https://github.com/patil-suraj/question_generation
@dataclass
class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
    A dictionary of tensors
    """
    
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100 
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
    
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
    }

def filter_dataset(dataset,split):
    from datasets import Dataset
    if "confidence" in dataset.column_names[split]:
        dataset = anthropic_aug_with_prosocial_label(dataset,split)
        dataset = Dataset.from_list(dataset,split=split)
        dataset = dataset.filter(lambda example : example["safety_label"]!="__needs_intervention__" and example["confidence"]>0.5)
        dataset = dataset.train_test_split(test_size=0.0001)
    return dataset

def prepare_dataset(tokenizer,col):
    all_datasets=[]
    if isinstance(CONFIG[col],dict):
        for key,value in CONFIG[col].items():
            dataset = load_dataset(key)
            try:
                dataset = dataset.rename_columns({"Human":"context","Assistant":"response"})
            except Exception as e:
                print(e)
            dataset = filter_dataset(dataset,"train")
            train_dataset = SafetyDataset(dataset,split=value,tokenizer=tokenizer,max_len=CONFIG["max_len"])
            all_datasets.append(train_dataset)
    
    return ConcatDataset(all_datasets)

def configure_optimizers(model,total_steps):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": CONFIG.get("weight_decay",0.0),
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG["learning_rate"],)
    scheduler = get_scheduler(CONFIG["scheduler"],optimizer,
                num_warmup_steps=int(CONFIG["warmup"]*total_steps),
                num_training_steps=total_steps
                )
    
    return optimizer,scheduler


def anthropic_aug_with_prosocial_label(dataset,split):

  new_data = []
  episodes = {}
  prev_task=""
  for  idx, d in enumerate(dataset[split]):
    prev_task = ",".join(d['rots']).lower()
    safety_label = d['safety_label']

    if safety_label ==   "__casual__":
        safety_label = "__casual__"
    else:
      if 'jewish' in prev_task or  'n word' in prev_task or "racial" in prev_task or "nazi" in prev_task or "holocust" in prev_task or "mexican" in prev_task or "hispanic" in prev_task or " jews " in prev_task or " jew " in prev_task or " black people " in prev_task or " asian people " in  prev_task or "nigger" in prev_task:
        safety_label = safety_label
        #print (safety_label, d)
      elif "self harm" in prev_task:
        safety_label = safety_label.replace("possibly_", "").replace("probably_", "")
        #print (safety_label, d)
      elif "date underage" in prev_task:
        if "joke" not in prev_task:
          safety_label =  "__needs__intervention__"
          #print (prev_task, d)   
      elif "damage" in prev_task or "robbing" in prev_task or " rob " in prev_task or "hide drug" in prev_task or "run from the cops" in prev_task or "blackmail" in prev_task or "violen" in prev_task or "steroids" in prev_task or "spike" in prev_task or "bomb" in prev_task or "hit someone" in prev_task or " kill" in prev_task or "cannibal" in prev_task or "take steroids" in prev_task or "drive drunk" in prev_task or "theft" in prev_task or "property" in prev_task or "hiding drugs" in prev_task or "frame someone" in prev_task or "drive after a few drink" in prev_task or "take something" in prev_task or "get high" in prev_task or "shoplift" in prev_task or "abusing" in prev_task or "abuse" in prev_task or " rape" in prev_task or " kill" in prev_task or "taking" in prev_task or "stalk" in prev_task or "steal" in prev_task or "hurt" in prev_task or "illegal" in prev_task or "hide money" in prev_task or "property damage" in prev_task or "someone's drink" in prev_task:
        if "joke" not in prev_task:
          safety_label = safety_label.replace("possibly_", "").replace("probably_", "")
      elif "dangerous" not in prev_task and safety_label == "__needs__intervention__" and d['confidence'] < .6:
        safety_label =  "__needs_caution__"
      elif safety_label ==  "__needs_caution__":
        if d['confidence'] < .6:
          safety_label = "__probably_needs_caution__"
          #print ('swap', safety_label, prev_task)
      elif safety_label == "__probably_needs_caution__":
        if d['confidence'] < .6:
          safety_label = "__probably_needs_caution__"
          #print ('swap', safety_label, prev_task)
      elif "possibly" in safety_label:
        if d['confidence'] < .6:
          safety_label =  "__casual__"
      #episodes[prev_task] = episodes.get(prev_task, []) + [f"Input: {d['Human']}\nOutput: {safety_label}"]
    
    d["safety_label"] = safety_label
    new_data.append(d)
  return new_data


if __name__ == "__main__":

    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)
    
    with open(os.path.join(ROOT_DIR,"config.json"),"w") as file:
        json.dump(CONFIG,file,indent=4)
    

    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"],padding_side="right",truncation_side="right",model_max_length=512)
    add_special_tokens(tokenizer,model)
    
    train_dataset = prepare_dataset(tokenizer,"train_dataset")
    valid_dataset = prepare_dataset(tokenizer,"test_dataset")

    total_steps = len(train_dataset) * CONFIG["epochs"]/(CONFIG["batch_size"]*CONFIG["gradient_accumulation_steps"])
    print("Total Number of train steps:",total_steps)
    optimizer,scheduler = configure_optimizers(model,total_steps)

    training_args = TrainingArguments(output_dir=ROOT_DIR, 
                                  per_device_train_batch_size=CONFIG["batch_size"], 
                                  per_device_eval_batch_size=CONFIG["batch_size"],
                                  gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
                                  num_train_epochs=CONFIG["epochs"],
                                  logging_steps=100,
                                  evaluation_strategy="steps",
                                  eval_steps=1000,
                                  save_steps=3000,
                                  report_to="wandb",
                                  push_to_hub=False,
                                  fp16=CONFIG["fp16"],
                                  run_name=f"safety-bot-sample-hawk-{jobid}",)


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
        optimizers = [optimizer,scheduler]
    )

    # Training
    trainer.train()

    wandb.finish()
    trainer.save_model(os.path.join(ROOT_DIR,"safety-model"))
