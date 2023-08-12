# Use a pipeline as a high-level helper
from datasets import load_dataset
from tqdm import tqdm
import json


    
PROMPT =  """### User: Given a prompt and response
    1. Label the question based on the level of difficulty as easy/medium/hard.
    2. Check if the answer contains explanation as instructed. Give yes or no
    3. Tag based of question using categories like math, translation, common sense, etc
    
    prompt: What is 1+1?
    response: 2
    output:\n 1.easy\n2.no\n3.math
    
    prompt:{prompt}
    response: {response}
    \n\n### Assistant:
    output:\n
    """
 
def infer(input_text, model, tokenizer, **kwargs):
     
     input_tokens = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
     output = model.generate(**input_tokens, **kwargs)
     output = tokenizer.decode(output.sequences[0])
     

def infer_client(prompt, **kwargs):
    client = Client("http://209.20.159.31:8080")
    return client.generate(prompt, **kwargs).generated_text
 
if __name__ == "__main__":
    
    
    dataset = load_dataset("shahules786/orca-best")
    dataset = dataset["train"]
    model_args = {
        "do_sample":True,
        "max_new_tokens":128,
        "top_p":0.95, 
        "top_k":0,
        "return_dict_in_generate":True,

    }
    
    # MODEL = "EleutherAI/gpt-neo-125m"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    # model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu")
    # model = llm = LLM(model=MODEL)

    outputs = []
    for item in tqdm(dataset):
        samples = item["conversation"]["samples"]
        instruction = item["instruction"]
        samples = [PROMPT.format(prompt=item["input"], response=item["output"]) for item in samples]
        output = [infer_client(item,**model_args) for item in samples]
        outputs.append(output)
        
    # dataset = dataset.add_column(MODEL.split("/")[-1], outputs)
    # dataset.to_json("orca-collection.json")
    with open("outputs.json","w") as file:
        json.dump(outputs,file,indent=4)