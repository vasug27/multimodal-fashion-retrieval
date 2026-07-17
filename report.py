import os
from fpdf import FPDF

class FashionRetrievalReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Glance ML Internship Assignment Report', border=0, ln=1, align='C')
        self.set_font('helvetica', 'I', 10)
        self.cell(0, 5, 'Multimodal Fashion & Context Retrieval Search Engine', border=0, ln=1, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', border=0, ln=0, align='C')

    def chapter_title(self, num, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(220, 230, 242)
        self.cell(0, 8, f'{num}. {title}', border=0, ln=1, align='L', fill=True)
        self.ln(3)

    def chapter_body(self, body_text):
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 5, body_text)
        self.ln(5)
        
    def bullet_point(self, title, desc):
        self.set_font('helvetica', 'B', 10)
        self.write(5, f" - {title}: ")
        self.set_font('helvetica', '', 10)
        self.write(5, f"{desc}\n")
        self.ln(1)

def generate_pdf():
    pdf = FashionRetrievalReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title("1", "Possible Approaches & Trade-offs")
    pdf.chapter_body(
        "To build a multimodal fashion search engine, several architectures can be considered, "
        "ranging from basic zero-shot embedding alignment to complex hybrid pipelines. "
        "Below are the primary candidate approaches along with their trade-offs:"
    )
    pdf.bullet_point(
        "Approach A: Vanilla CLIP Zero-Shot Retrieval",
        "Uses a single vision-language model (e.g. CLIP) to encode both full images and text queries, performing retrieval via cosine similarity. "
        "Trade-off: Fast and zero-shot, but struggles heavily with spatial compositionality (e.g., matching a 'red shirt with blue pants' to a photo containing 'blue shirt with red pants') and fine-grained fashion attribute binding."
    )
    pdf.bullet_point(
        "Approach B: VLM Image Captioning + Text Retrieval",
        "Uses a Visual Language Model (VLM) like BLIP or LLaVA to generate detailed descriptive captions for every image, storing captions in a text vector index (e.g., MiniLM). "
        "Trade-off: Rich descriptions, but captioning on CPU is extremely slow and expensive during indexing. Relies entirely on the captioner capturing all attributes correctly."
    )
    pdf.bullet_point(
        "Approach C: Localized Region Embedding (LRE) Pipeline (Chosen)",
        "A hybrid pipeline combining a fine-tuned fashion object detector (YOLOS-Fashionpedia) and CLIP. "
        "It localizes apparel regions, extracts region-level visual embeddings, and decomposes text queries into global context and crop-level attributes. "
        "Trade-off: Highly accurate attribute binding, solves compositionality, runs efficiently offline, and requires no API keys or complex setups."
    )
    pdf.ln(5)

    pdf.chapter_title("2", "Short Write-up on Chosen Approach (LRE)")
    pdf.chapter_body(
        "Our chosen architecture implements the Localized Region Embedding (LRE) Pipeline. "
        "It operates in two distinct phases:\n\n"
        "1. Part A - The Indexer:\n"
        "   - Extracts a global image embedding using CLIP (ViT-B/32) to capture context and vibe.\n"
        "   - Runs YOLOS-Fashionpedia to detect garment categories (e.g., shirt, pants, coat, tie, dress).\n"
        "   - Crops detected clothing regions and generates crop-level embeddings using CLIP.\n"
        "   - Saves these global and region-level features in a structured, lightweight index.\n\n"
        "2. Part B - The Retriever (Query Decomposition):\n"
        "   - Parses search queries into global terms (e.g. 'formal setting') and local terms (e.g. 'red tie', 'white shirt').\n"
        "   - Computes global visual similarity and crop-level local similarities.\n"
        "   - Combines scores using a weighted aggregation: final_score = (global_score * 0.4) + (min(local_scores) * 0.6).\n"
        "   - The 'min' operator acts as a logical AND, ensuring all compositional attributes are matched."
    )
    
    pdf.chapter_title("3", "Codebase & Directory Structure")
    pdf.chapter_body(
        "The complete assignment implementation is structured cleanly into modules:\n"
        " - index.py: Offline pipeline to index images and generate global + crop-level local embeddings.\n"
        " - retrieve.py: Real-time search script featuring query decomposition and score aggregation.\n"
        " - evaluate.py: Evaluator running test queries, logging metrics, and outputting visualization grids.\n"
        " - fashion_index.pkl: The generated visual index containing metadata and normalized tensor features.\n\n"
        "GitHub Codebase URL: https://github.com/vasug27/multimodal-fashion-retrieval"
    )

    pdf.chapter_title("4", "Approaches for Future Work & Scale")
    pdf.bullet_point(
        "Extending to Cities, Locations & Weather",
        "We can integrate a text-based context parser using a lightweight Named Entity Recognition (NER) model to detect location nouns (e.g., 'Paris', 'office') and weather attributes (e.g., 'raining', 'sunny'). These are used as metadata filters or queried against specialized weather-context text classifiers."
    )
    pdf.bullet_point(
        "Improving Precision",
        "Precision can be further enhanced by incorporating a color histogram extraction or dominant color clustering (K-Means on L*a*b* space) on the cropped garments to act as hard filters. Additionally, fine-tuning CLIP's vision projection layer specifically on fashion datasets like Fashionpedia or DeepFashion will improve fine-grained attribute representation."
    )
    pdf.bullet_point(
        "Scalability to 1 Million Images",
        "To scale to millions of images, we can migrate the local index to a distributed vector database like Milvus or Qdrant. For localized regions, we can store crop coordinates in metadata and use hierarchical indexes (HNSW) to perform sub-millisecond retrieval."
    )

    output_path = "Glance_ML_Assignment_Submission.pdf"
    pdf.output(output_path)
    print(f"Successfully generated submission report PDF at: {output_path}")

if __name__ == "__main__":
    generate_pdf()
