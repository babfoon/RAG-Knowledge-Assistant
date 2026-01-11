import os
import PyPDF2
from sentence_transformers import SentenceTransformer

# Load the sentence transformers model for embedding process
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Extracting text through PDF files
def process_pdf(pdf_path):
  text = ''
  with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
      text += page.extract_text()
  return text

# Converts a document's text into embeddings
def generate_embeddings(text):
  sentences = text.split ('\n')
  embeddings = embedder.encode (sentences)
  return sentences, embeddings

# Example
if __name__ == '__main__':
  raw_text = process_pdf('data/sample.pdf') # <--- Replace when ready
  sentences, embeddings = generate_embeddings(raw_text)
  print(f'Processed {len(sentences)} sentences from the document.')
