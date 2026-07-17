# Multimodal Fashion & Context Retrieval Search Engine

This repo implements a zero-shot, compositional search engine for fashion items and contextual image retrieval using local object detection and vision-language features.

## ML Architecture & Logic

A vanilla application of CLIP (Contrastive Language-Image Pre-training) struggles with compositionality (e.g., distinguishing "red shirt with blue pants" from "blue shirt with red pants") because it encodes images globally, causing attribute binding errors.

Our solution implements a **Localized Region Embedding (LRE) Pipeline** to resolve this:

1. **Localization**: We use `yolos-fashionpedia` (a lightweight object detector fine-tuned on Fashionpedia) to segment clothes and identify regions of interest (e.g. `shirt`, `pants`, `tie`, `dress`, `coat`).
2. **Local Feature Binding**: We crop the localized garments and generate regional CLIP embeddings (`clip-vit-base-patch32`) for each crop. This binds attributes (like color or fabric) to specific garment categories.
3. **Query Decomposition & Scoring**: 
   - Global query matching (CLIP vs. full image) captures context and scene background (e.g. "inside a modern office").
   - Local queries match colors/details to the cropped garment regions.
   - We aggregate scores: `final_score = (global_score * 0.4) + (min(local_scores) * 0.6)`. The `min()` operator functions as a soft-AND, ensuring all specified attributes must be matched.

---

## Repository Directory Structure

```
├── results/                    # Top-5 visual grid results for evaluation queries
├── index.py                    # Batched CPU-optimized offline indexing pipeline
├── retrieve.py                 # Real-time search script featuring query decomposition
├── evaluate.py                 # Test evaluator running prompts and plotting grids
├── report.py                   # Automatic generator for FPDF2 submission report
├── fashion_index.pkl           # Saved index file containing metadata and embeddings
├── evaluation_report.md        # Detailed ranking tables for test queries
├── .gitignore                  # Git ignore rules
└── README.md                   # This project manual
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vasug27/multimodal-fashion-retrieval.git
   cd multimodal-fashion-retrieval
   ```

2. **Install dependencies**:
   ```bash
   pip install torch transformers pillow numpy matplotlib fpdf2 tqdm
   ```

3. **Index the images** (Already processed and stored in `fashion_index.pkl`):
   ```bash
   python index.py --image_dir test --limit 600 --batch_size 16
   ```

4. **Query the retriever**:
   ```bash
   python retrieve.py "A red tie and a white shirt in a formal setting"
   ```

---

## Evaluation Queries & Results

We run 5 evaluation categories in `evaluate.py`:

| Category | Query | Top Retrieve File | Total Score |
|---|---|---|---|
| **Attribute Specific** | "A person in a bright yellow raincoat." | `1ae9cdebd762234889e60f2c0d07a768.jpg` | **0.2740** |
| **Contextual/Place** | "Professional business attire inside a modern office." | `0cb2351c74ccc0b38128e91d0629625d.jpg` | **0.2781** |
| **Complex Semantic** | "Someone wearing a blue shirt sitting on a park bench." | `165b0195ff6dc62096d0c49c680407ca.jpg` | **0.2532** |
| **Style Inference** | "Casual weekend outfit for a city walk." | `289236c117e289c12d6c57e2cb4ce427.jpg` | **0.2885** |
| **Compositional** | "A red tie and a white shirt in a formal setting." | `280a76210b66ab30456070b315ac81e7.jpg` | **0.2427** |

Visual grid plots of the top-5 outputs are saved in the [results/](results) directory.

---

## Author

**Vasu Goel**

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:vasugoel2754@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vasugoel503/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vasug27)

---
