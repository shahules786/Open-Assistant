from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json
import pandas as pd
import numpy as np

SBERT_MODEL = "all-MiniLM-L6-v2"
MAX_ROWS = 5

def load_vectorizer(model=SBERT_MODEL):
     return SentenceTransformer(model)
    
def vectorize_text(model,texts):
    
    return model.encode(texts, show_progress_bar=True)


def prepare_dataset(dataset, tokenizer, max_seq_len):
     sys,inputs,outputs = [dataset[key] for key in ["instruction","input","output"]]
     sys = list(set(sys))
     assert len(sys) == 1, "expects single instruction type"
     max_seq_len = max_seq_len - len(tokenizer.encode(sys[0])) - 4

     model_inputs = [f"{inp} {out}" for inp,out in zip(inputs, outputs)]
     model_inputs = tokenizer.batch_encode_plus(model_inputs, return_attention_mask=False)["input_ids"]
     dataset_tokens_map = {i:len(tokens) for i,tokens in enumerate(model_inputs)}
     dataset_tokens_map = dict(
         sorted(dataset_tokens_map.items(), key=lambda item: item[1]))

     samples = []
     current_sample_len = 0
     current_sample = []
     num_of_samples = 0
     for idx,len_ in dataset_tokens_map.items():
         if (current_sample_len + len_ <= max_seq_len) and (num_of_samples < MAX_ROWS):
             current_sample.append({"input":inputs[idx],"output":outputs[idx]})
             current_sample_len += len_
             num_of_samples+=1
         else:
             num_of_samples = 0
             samples.append(current_sample)
             if len_ <= max_seq_len:
                 current_sample = [{"input":inputs[idx],"output":outputs[idx]}]
                 current_sample_len = len_
                 num_of_samples += 1

     if len(current_sample) > 0:
         samples.append(current_sample)
     return samples
    


def cluster_indices(dataset, model, tokenizer, max_seq_len, threshold):
    
    
    ## create embeddings
    emmbeddings = vectorize_text(model, [item["input"].strip() for item in dataset])
    
    
    dataset_samples = []
    ## add indices
    d = emmbeddings.shape[-1]
    quantizer = faiss.IndexFlatIP(d)
    nlist = 100 if emmbeddings.shape[0] > 100 else 2
    index = faiss.IndexIVFFlat(quantizer,d,nlist)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    faiss.normalize_L2(emmbeddings)
    index.train(emmbeddings[:5000])
    index.add(emmbeddings)
    
    removed_indices = []
    for i in range(emmbeddings.shape[0]):
        if i not in removed_indices:
            _,_,indices = index.range_search(emmbeddings[i].reshape(1,-1),thresh=threshold)
            
            ## add random samples for 5% data
            if len(indices) < 2 and (np.random.uniform(0,1) < 0.05):
                 _,_,indices = index.range_search(emmbeddings[i].reshape(1,-1),thresh=0.3)
                 
            if len(indices) > 1:
                index.remove_ids(indices) 
                removed_indices.extend(indices.tolist())
                samples = prepare_dataset(dataset.select(indices), tokenizer, max_seq_len)
                dataset_samples.extend(samples)

            else:
                samples = dataset.select([i])
                sys,inputs,outputs = [samples[key] for key in ["instruction","input","output"]]
                samples = [{"input":inputs[0],"output":outputs[0]}]
                dataset_samples.append(samples)
    return dataset_samples


def main(max_seq_len=8000, threshold=0.75):
    
    model = load_vectorizer()
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    dataset = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl")["train"]
    instructions = json.load(open("instructions.json"))["orca"]
    
    orca_df = pd.DataFrame()
    for instr in tqdm(instructions):
        subsample = dataset.filter(lambda example: example["instruction"]==instr)
        dataset_samples = cluster_indices(subsample,model,tokenizer,max_seq_len,threshold)
        df = pd.DataFrame({"conversation":dataset_samples})
        df["source"] = "ehartford/dolphin"
        df["instruction"] = instr
        orca_df = pd.concat([orca_df, df], ignore_index=True)

    with open("orca-chat-gpt4.json","w") as file:
        json.dump(orca_df.to_dict("records"),file,indent=4)
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max_sequence_len", type=int, default=8000, help="max seq len")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold")
    args = parser.parse_args().__dict__
    
    main(args.get("max_sequence_len"), args.get("threshold"))