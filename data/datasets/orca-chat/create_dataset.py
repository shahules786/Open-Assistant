from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np 

SBERT_MODEL = "all-MiniLM-L6-v2"

def load_vectorizer(model=SBERT_MODEL):
     return SentenceTransformer(model)
    
def vectorize_text(model,texts):
    
    return model.encode(texts, show_progress_bar=True)

def batch_search(input_vectors, index,knn=3,batch_size=32, min_score=0.8):
    
    similar_dict = {}
    for idx in tqdm(range(0,input_vectors.shape[0],batch_size)):
        I,D = index.search(input_vectors[idx:idx+batch_size],knn)
        indices = np.where(I>min_score,D,-1)
        similar_dict.update({idx+i:[k for k in indices[1:] if k!=-1] for i,indices in enumerate(indices)})
        
    return similar_dict

def main():
    
    model = load_vectorizer()
    dataset = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl")["train"]
    # dataset = dataset["train"].select(range(0,1000))
    input_text = [item["input"].strip() for item in dataset]
    input_vectors = vectorize_text(model, input_text)
    
    d = input_vectors.shape[-1]
    
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer,d,5)
    faiss.normalize_L2(input_vectors)
    index.train(input_vectors[:10000])
    index.add(input_vectors)
    similar_dict = batch_search(input_vectors, index,knn=5, min_score=0.7)
    dataset = dataset.add_column("history", similar_dict.values())
    dataset.push_to_hub("shahules786/dolphin-gpt4-chat")
    

if __name__ == "__main__":
    main()