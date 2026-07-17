import os
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from transformers import (
    YolosImageProcessor,
    YolosForObjectDetection,
    CLIPProcessor,
    CLIPModel
)

def parse_args():
    parser = argparse.ArgumentParser(description="Index images for Multimodal Fashion Retrieval")
    parser.add_argument("--image_dir", type=str, default="test")
    parser.add_argument("--output", type=str, default="fashion_index.pkl")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' not found.")
        return

    num_threads = min(12, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"Set PyTorch threads to {num_threads} (out of {os.cpu_count()} CPU cores)")

    all_files = sorted(os.listdir(args.image_dir))
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if args.limit > 0:
        image_files = image_files[:args.limit]
    
    print(f"Found {len(image_files)} images to index.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading YOLOS-Fashionpedia...")
    yolos_name = "valentinafevu/yolos-fashionpedia"
    yolos_processor = YolosImageProcessor.from_pretrained(yolos_name)
    yolos_model = YolosForObjectDetection.from_pretrained(yolos_name).to(device)
    yolos_model.eval()

    print("Loading CLIP...")
    clip_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_model.eval()

    index_data = {}

    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Indexing Batches"):
        batch_names = image_files[i:i + args.batch_size]
        batch_images = []
        batch_loaded_names = []
        
        for name in batch_names:
            img_path = os.path.join(args.image_dir, name)
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_loaded_names.append(name)
            except Exception as e:
                print(f"Skipping {name}: {e}")

        if not batch_images:
            continue

        clip_inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            global_embs = clip_model.get_image_features(**clip_inputs)
            global_embs = global_embs / global_embs.norm(p=2, dim=-1, keepdim=True)
            global_embs_np = global_embs.cpu().numpy()

        yolos_inputs = yolos_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = yolos_model(**yolos_inputs)
        
        target_sizes = torch.tensor([img.size[::-1] for img in batch_images], device=device)
        results = yolos_processor.post_process_object_detection(outputs, threshold=args.confidence_threshold, target_sizes=target_sizes)

        for idx, (img_name, image, result) in enumerate(zip(batch_loaded_names, batch_images, results)):
            width, height = image.size
            global_emb_np = global_embs_np[idx]
            
            detected_objects = []
            crop_images = []
            crop_metadata = []
            
            for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                box_coords = [float(coord) for coord in box.tolist()]
                label_idx = int(label.item())
                label_name = yolos_model.config.id2label[label_idx]
                conf = float(score.item())

                xmin = max(0.0, box_coords[0])
                ymin = max(0.0, box_coords[1])
                xmax = min(float(width), box_coords[2])
                ymax = min(float(height), box_coords[3])

                if xmax <= xmin or ymax <= ymin:
                    continue

                cropped_img = image.crop((xmin, ymin, xmax, ymax))
                crop_images.append(cropped_img)
                crop_metadata.append({
                    "label": label_name,
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": conf
                })

            if crop_images:
                crop_inputs = clip_processor(images=crop_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    local_embs = clip_model.get_image_features(**crop_inputs)
                    local_embs = local_embs / local_embs.norm(p=2, dim=-1, keepdim=True)
                    local_embs_np = local_embs.cpu().numpy()
                
                for c_idx, meta in enumerate(crop_metadata):
                    detected_objects.append({
                        "label": meta["label"],
                        "box": meta["box"],
                        "confidence": meta["confidence"],
                        "embedding": local_embs_np[c_idx]
                    })

            index_data[img_name] = {
                "global_embedding": global_emb_np,
                "detected_objects": detected_objects
            }

    with open(args.output, "wb") as f:
        pickle.dump(index_data, f)
    
    print(f"Successfully indexed {len(index_data)} images and saved to {args.output}")

if __name__ == "__main__":
    main()
