import json
import os
import PyPDF2
import warnings
warnings.filterwarnings('ignore')

def extract_and_chunk_pdf(pdf_path: str, chunk_size=1000) -> str:
    """Extract text from PDF and create chunks for Colab processing"""
    print(f"\nProcessing PDF: {pdf_path}")
    
    # Create output directory
    doc_name = os.path.basename(pdf_path).replace('.pdf', '')
    output_dir = "colab_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        for i, page in enumerate(pdf_reader.pages, 1):
            text += page.extract_text() + "\n\n"
            print(f"\rExtracting text: {i}/{total_pages} pages", end="")
    print("\nText extraction complete")
    
    # Create chunks
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Save chunks for Colab
    chunks_path = os.path.join(output_dir, f"{doc_name}_chunks.json")
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"\nCreated {len(chunks)} chunks")
    print(f"Saved to: {chunks_path}")
    print("\nNow you can:")
    print("1. Upload this JSON file to Google Colab")
    print("2. Use the provided Colab notebook to create embeddings")
    print("3. Download the resulting embeddings.json back for querying")
    
    return chunks_path

if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF file: ")
    extract_and_chunk_pdf(pdf_path)
