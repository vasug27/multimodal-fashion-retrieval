import os
import matplotlib.pyplot as plt
from PIL import Image
from retrieve import FashionRetriever

EVAL_QUERIES = {
    "1_attribute_specific": "A person in a bright yellow raincoat.",
    "2_contextual_place": "Professional business attire inside a modern office.",
    "3_complex_semantic": "Someone wearing a blue shirt sitting on a park bench.",
    "4_style_inference": "Casual weekend outfit for a city walk.",
    "5_compositional": "A red tie and a white shirt in a formal setting."
}

def create_visualization_grid(query_name, query_str, results, image_dir="test", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    fig.suptitle(f"Query: '{query_str}'", fontsize=14, y=0.98)
    
    if len(results) == 1:
        axes = [axes]
        
    for idx, (ax, res) in enumerate(zip(axes, results)):
        img_path = os.path.join(image_dir, res["image_name"])
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"Rank {idx+1}\nScore: {res['score']:.3f}\nFile: {res['image_name']}", fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha="center", va="center")
            ax.set_title(f"Rank {idx+1} (Error)")
        ax.axis("off")
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{query_name}_results.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization for query '{query_name}' to {output_path}")

def main():
    print("Initializing Fashion Retriever...")
    retriever = FashionRetriever("fashion_index.pkl")
    
    print("\nRunning Evaluation Queries...")
    
    report_md = "# Evaluation Results\n\n"
    
    for q_name, q_str in EVAL_QUERIES.items():
        print(f"\nEvaluating Query: {q_str}")
        results = retriever.search(q_str, k=5)
        
        report_md += f"## Query: \"{q_str}\"\n\n"
        report_md += "| Rank | Image Name | Total Score | Global Score |\n"
        report_md += "|------|------------|-------------|--------------|\n"
        
        for idx, res in enumerate(results):
            print(f"Rank {idx+1}: {res['image_name']} (Score: {res['score']:.4f}, Global: {res['global_score']:.4f})")
            report_md += f"| {idx+1} | [{res['image_name']}](file:///c:/Development/glance_assignment/test/{res['image_name']}) | {res['score']:.4f} | {res['global_score']:.4f} |\n"
            
        report_md += "\n"
        
        create_visualization_grid(q_name, q_str, results)
        
    with open("evaluation_report.md", "w") as f:
        f.write(report_md)
        
    print("\nEvaluation complete. Saved report to evaluation_report.md")

if __name__ == "__main__":
    main()
