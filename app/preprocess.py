import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import PyPDF2
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
from pytesseract import image_to_string

# Load Sentence Transformer model for embedding generation
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# Cleans and extracts text from the input PDF
def process_pdf(pdf_path):
    try:
        text = ''
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()

        if text.strip():
            print('Text successfully extracted using PyPDF2.')

            # Cleanup the extracted text
            text = text.replace("\n", " ").replace("\xa0", " ")
            text = " ".join(text.split())  # Normalize multiple spaces
            return text

        print('PDF is scanned. Using OCR...')
        images = convert_from_path(pdf_path)
        text_blocks = [image_to_string(image) for image in images]
        text = " ".join(text_blocks).replace("\n", " ")
        return " ".join(text.split())  # Normalize spaces for OCR text

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return ''


# Generates sentence embeddings and pre-processes sentences
def generate_embeddings(text):
    # Break text into sentences
    sentences = sent_tokenize(text)

    # Filter and clean sentences
    sentences = [' '.join(s.split()) for s in sentences if len(s.split()) > 3]  # Filter meaningful sentences
    print(f'Sentences processed: {len(sentences)}')
    print(f'Sample sentences: {sentences[:5]}')

    # Generate embeddings using the transformer model
    if not sentences:
        return [], np.array([])

    embeddings = embedder.encode(sentences)
    return sentences, embeddings
