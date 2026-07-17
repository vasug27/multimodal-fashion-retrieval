import pickle
import numpy as np
import torch
import os
import argparse
from transformers import CLIPProcessor, CLIPModel

GARMENT_MAP = {
    "raincoat": ["coat", "jacket"],
    "coat": ["coat"],
    "jacket": ["jacket", "cardigan"],
    "sweater": ["sweater"],
    "cardigan": ["cardigan"],
    "blazer": ["jacket"],
    "suit": ["jacket", "pants"],
    "hoodie": ["top, t-shirt, sweatshirt"],
    "sweatshirt": ["top, t-shirt, sweatshirt"],
    "t-shirt": ["top, t-shirt, sweatshirt"],
    "tshirt": ["top, t-shirt, sweatshirt"],
    "shirt": ["shirt, blouse", "top, t-shirt, sweatshirt"],
    "blouse": ["shirt, blouse"],
    "pants": ["pants"],
    "trousers": ["pants"],
    "jeans": ["pants"],
    "shorts": ["shorts"],
    "skirt": ["skirt"],
    "dress": ["dress"],
    "gown": ["dress"],
    "tie": ["tie"],
    "shoe": ["shoe"],
    "shoes": ["shoe"],
    "sneakers": ["shoe"],
    "boots": ["shoe"],
    "bag": ["bag, wallet"],
    "handbag": ["bag, wallet"],
    "purse": ["bag, wallet"],
    "wallet": ["bag, wallet"],
    "scarf": ["scarf"],
    "belt": ["belt"],
    "glasses": ["glasses"],
    "sunglasses": ["glasses"],
    "hat": ["hat"],
    "umbrella": ["umbrella"]
}

COLORS = ["red", "blue", "yellow", "green", "black", "white", "grey", "gray", "brown", "pink", "purple", "orange", "beige", "bright yellow"]

def decompose_query(query):
    query_lower = query.lower()
    local_queries = []
    words = query_lower.split()
    
    found_garments = []
    for g_key in GARMENT_MAP.keys():
        if g_key in query_lower:
            found_garments.append(g_key)
            
    found_garments = sorted(found_garments, key=len, reverse=True)
    filtered_garments = []
    for fg in found_garments:
        if not any(fg != other and fg in other for other in filtered_garments):
            filtered_garments.append(fg)
            
    for fg in filtered_garments:
        associated_color = None
        for color in COLORS:
            if f"{color} {fg}" in query_lower:
                associated_color = color
                break
        
        term = f"{associated_color} {fg}" if associated_color else fg
        local_queries.append({
            "garment_key": fg,
            "target_labels": GARMENT_MAP[fg],
            "term": f"a photo of a {term}"
        })
        
    return query, local_queries

def compute_cosine_similarity(query_emb, doc_embs):
    return np.dot(doc_embs, query_emb)

class FashionRetriever:
    def __init__(self, index_path="fashion_index.pkl", model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading retriever using device: {self.device}")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file '{index_path}' not found.")
        
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
            
        self.image_names = list(self.index.keys())
        self.global_embeddings = np.array([self.index[name]["global_embedding"] for name in self.image_names])
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            return text_emb.cpu().numpy().squeeze()

    def search(self, query_str, k=5, w_global=0.4, w_local=0.6):
        global_query, local_queries = decompose_query(query_str)
        
        print(f"\nSearching for: '{query_str}'")
        print(f"Decomposed local sub-queries: {[l['term'] for l in local_queries]}")
        
        global_query_emb = self.get_text_embedding(global_query)
        global_scores = compute_cosine_similarity(global_query_emb, self.global_embeddings)
        
        if not local_queries:
            final_scores = global_scores
        else:
            local_scores_per_query = []
            
            for lq in local_queries:
                term_emb = self.get_text_embedding(lq["term"])
                target_labels = lq["target_labels"]
                
                query_local_scores = []
                
                for img_name in self.image_names:
                    detected_objects = self.index[img_name]["detected_objects"]
                    matching_crops = [
                        obj for obj in detected_objects 
                        if any(lbl in obj["label"].lower() for lbl in target_labels)
                    ]
                    
                    if matching_crops:
                        crop_embs = np.array([obj["embedding"] for obj in matching_crops])
                        crop_sims = compute_cosine_similarity(term_emb, crop_embs)
                        max_sim = float(np.max(crop_sims))
                        query_local_scores.append(max_sim)
                    else:
                        query_local_scores.append(0.15)
                        
                local_scores_per_query.append(query_local_scores)
                
            local_scores_per_query = np.array(local_scores_per_query)
            aggregated_local_scores = np.min(local_scores_per_query, axis=0)
            
            final_scores = (global_scores * w_global) + (aggregated_local_scores * w_local)
            
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "image_name": self.image_names[idx],
                "score": float(final_scores[idx]),
                "global_score": float(global_scores[idx])
            })
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--index", type=str, default="fashion_index.pkl")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    
    retriever = FashionRetriever(args.index)
    results = retriever.search(args.query, k=args.k)
    for i, res in enumerate(results):
        print(f"{i+1}. Image: {res['image_name']} - Score: {res['score']:.4f}")
