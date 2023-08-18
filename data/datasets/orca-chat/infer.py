from datasets import load_dataset
from tqdm import tqdm
import json
from text_generation import Client
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

    
PROMPT = """
### System:\nYou are an AI that follows instructions extremely well. I want you to act as an instructor.\n\n
### User
We would like you to evaluate and rate the difficulty and complexity of the following question. Label the question based on the level of difficulty as easy/medium/hard.
Also generate appropriate tags for the given question.

question : In this task, you are given inputs i,j, and A, where i and j are integers and A is a list. You need to list all elements of A from the ith element to the jth element. i and j will be non-negative, and will always have a value less than the length of A. i will always be less than j.\nInput: 18, 28, ['e', '3481', '981', '8783', '9239', '5303', '3059', '6985', '129', '5915', 'M', '3953', '1053', '8777', 'C', '443', '3013', 'P', 'F', '7697', 'T', '9475', 'w', 'T', '141', '5493', '2631', '4553']
1.Question Difficulty Level: Easy
3.Tag Categories: Math, Programming Logic

question:{inputs}
\n\n### Assistant:\n
"""
 
def infer(input_text, model, tokenizer, **kwargs):
     
     input_tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4000).to(model.device)
     print(input_tokens["input_ids"].shape)
     output = model.generate(**input_tokens, **kwargs)
     output = tokenizer.decode(output.sequences[0])
     

def infer_client(prompt, **kwargs):
    try:
        client = Client("http://127.0.0.1:8080")
        output = client.generate(prompt, **kwargs).generated_text
    except Exception as e:
        print(e)
        output = ["None"]
    return output

def filter_dataset(dataset_name):
    
        
    dataset = load_dataset(dataset_name)
    dataset = dataset["train"].select(range(0,100))
    model_args = {
        "top_p":0.95,
        "max_new_tokens":128,
        "top_k":0

    }

    outputs = []
    for item in tqdm(dataset):
        samples = item["conversation"]["samples"]
        samples = [PROMPT.format(inputs=item["input"]) for item in samples]
        output = [infer_client(item,**model_args) for item in samples]
        outputs.append(output)
        
    return outputs
    
    
def generate_random(num):
    
    outputs = []
    model_args = {
        "top_p":0.95,
        "max_new_tokens":128,
        "top_k":0

    }
    system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
    message = "Generate a random number between 1 and 10."
    prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
    for i in range(num):
        outputs.append(infer_client(prompt, **model_args))
    return outputs
    
 
if __name__ == "__main__":
    
    outputs = generate_random(1000)
    with open("random_numbers.json","w") as file:
        json.dump(outputs,file,indent=4)