from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel
from collections import Counter
from tqdm import tqdm
import argparse
import json
import pandas as pd
import numpy as np

SBERT_MODEL = "all-mpnet-base-v2"
MAX_ROWS = 20

def load_vectorizer(model=SBERT_MODEL):
     
     return FlagModel('BAAI/bge-base-en', query_instruction_for_retrieval="Represent this sentence for searching relevant passages:")

     #return SentenceTransformer(model)
    
def vectorize_text(model,texts):
    
    return model.encode(texts, batch_size=32)


def prepare_dataset(dataset, max_seq_len):
     sys,inputs,outputs = [dataset[key] for key in ["instruction","input","output"]]
     sys = list(set(sys))
     assert len(sys) == 1, "expects single instruction type"
     max_seq_len = max_seq_len - (len(sys[0])//4) - 4

     model_inputs = [f"{inp} {out}" for inp,out in zip(inputs, outputs)]
     dataset_tokens_map = {i:len(sent)//4 for i,sent in enumerate(model_inputs)}
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
    

def create_faiss_index(emmbeddings):
    
      ## add indices
    d = emmbeddings.shape[-1]
    quantizer = faiss.IndexFlatIP(d)
    nlist = 100 if emmbeddings.shape[0] > 100 else 2
    index = faiss.IndexIVFFlat(quantizer,d,nlist)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    faiss.normalize_L2(emmbeddings)
    index.train(emmbeddings[:5000])
    index.add(emmbeddings)

    return index 

def cluster_indices(dataset, model, max_seq_len, threshold):
    
    
    ## create embeddings
    emmbeddings = vectorize_text(model, [item["input"].strip() for item in dataset])
    
    dataset_samples = []
    index = create_faiss_index(emmbeddings)
    
    removed_indices = []
    for i in range(emmbeddings.shape[0]):
        if i not in removed_indices:
            _,D,indices = index.range_search(emmbeddings[i].reshape(1,-1),thresh=threshold)
            dup_indices = indices[(D >= 0.9).nonzero()[0].tolist()].tolist()
            if i in dup_indices:
                dup_indices.remove(i)
            ## add random samples for 5% data
            if len(indices) < 2 and (np.random.uniform(0,1) < 0.05):
                 _,_,indices = index.range_search(emmbeddings[i].reshape(1,-1),thresh=0.3)
                 
            if len(indices) > 1:
                index.remove_ids(indices) 
                removed_indices.extend(indices.tolist())
                len_indices = len(indices)
                indices = [i for i in indices if i not in dup_indices]
                samples = prepare_dataset(dataset.select(indices), max_seq_len)
                dataset_samples.extend(samples)

            else:
                samples = dataset.select([i])
                sys,inputs,outputs = [samples[key] for key in ["instruction","input","output"]]
                samples = [{"input":inputs[0],"output":outputs[0]}]
                dataset_samples.append(samples)
    return dataset_samples



def cluster_similar(dataset, model,threshold):
    
    emmbeddings = vectorize_text(model, ["\n".join([item["input"]]) for item in dataset])
    index = create_faiss_index(emmbeddings)
    dataset_samples = []
    removed_indices = []
    for i in tqdm(range(emmbeddings.shape[0])):
        if i not in removed_indices:
            _,D,indices = index.range_search(emmbeddings[i].reshape(1,-1),thresh=threshold)
            data_sample = {}
            if len(indices) > 1:
                index.remove_ids(indices) 
                removed_indices.extend(indices.tolist())
                data_sample["samples"] = [{"input":item["input"],"output":item["output"]} for item in dataset.select(indices)]

            else:
                samples = dataset.select([i])
                sys,inputs,outputs = [samples[key] for key in ["instruction","input","output"]]
                data_sample["samples"]  = [{"input":inputs[0],"output":outputs[0]}]
            dataset_samples.append(data_sample)
        
    return dataset_samples


def count_output_chars(examples):
    response = examples['output'] 
    lens = [len(r) for r in response]
    
    return {"output_len":lens}

def update_json(df, filename):
    
    data = json.load(open(filename))
    data.append(df.to_dict("records"))
    with open(filename,'w') as file:
        json.dump(data, file, indent=4)

def main(max_seq_len=8000, threshold=0.75):
    
    filename = "flan1m-cluster"
    with open(filename,"w") as file:
        json.dump([],file)
    model = load_vectorizer()
    dataset = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl")["train"]
    dataset = dataset.map(count_output_chars,batch_size=32,batched=True)
    dataset = dataset.filter(lambda ex: ex['output_len']//4>100)
    
    instructions = [item["instruction"] for item in dataset]
    instructions = list(dict(Counter(instructions).most_common()).keys())[::-1]
    print("Number of instructions",len(instructions))
    # orca_df = pd.DataFrame()
    for instr in tqdm(instructions[3:]):
        print(f"processing instruction {instr}")
        subsample = dataset.filter(lambda example: example["instruction"]==instr)
        # dataset_samples = cluster_indices(subsample,model,max_seq_len,threshold)
        dataset_samples = cluster_similar(subsample,model,threshold)
        df = pd.DataFrame({"conversation":dataset_samples})
        df["source"] = "ehartford/dolphin"
        df["instruction"] = instr
        update_json(df, filename)
        # orca_df = pd.concat([orca_df, df], ignore_index=True)

    # with open("orca-gpt4-dedup.json","w") as file:
    #     json.dump(orca_df.to_dict("records"),file,indent=4)
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max_sequence_len", type=int, default=8000, help="max seq len")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold")
    args = parser.parse_args().__dict__
    
    main(args.get("max_sequence_len"), args.get("threshold"))