from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SBERT_MODEL = "all-MiniLM-L6-v2"

def load_vectorizer(model=SBERT_MODEL):
     return SentenceTransformer(model)
    
def vectorize_text(model,texts):
    
    return model.encode(texts, show_progress_bar=True)

def batch_search(input_vectors, batch_size=32, min_score=0.8):
    
    similar_dict = {}
    for idx in tqdm(range(0,input_vectors.shape[0],batch_size)):
        I,D = index.search(input_vectors[idx:idx+batch_size],k)
        indices = np.where(I>0.5,D,-1)
        similar_dict.update({idx+i:[k for k in indices[1:] if k!=-1] for i,indices in enumerate(indices)})
        
    return similar_dict

def main():
    
    model = load_vectorizer()
    dataset = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl")
    dataset = dataset["train"].sample(range(0,1000))
    input_text = [item["input"].strip() for item in dataset]
    input_vectors = vectorize_text(model, input_text)
    
    d = input_vectors.shape[-1]
    k = 3
    
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer,d,5)
    faiss.normalize_L2(input_vectors)
    index.train(input_vectors[:10000])
    index.add(input_vectors)
    similar_dict = batch_search(input_vectors, min_score=0.75)
    dataset = dataset.add_column("history", similar_dict.values())
    dataset.push_to_hub("shahules786/dolphin-gpt4-chat")
    
