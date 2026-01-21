import os
from app.preprocess import process_pdf, generate_embeddings
from app.vectorstore import save_index, load_index
from app.query import query_faiss, generate_response
import numpy as np


DOCUMENT_PATH = 'data/Sherlock Holmes.pdf'
INDEX_PATH = 'embeddings/index.faiss'

# Global variable to store sentences
sentences = []


# Builds the FAISS index and saves the associated sentences
def build_index():
    global sentences
    print('Processing the document...')
    text = process_pdf(DOCUMENT_PATH)

    if text.strip():
        print(f'Extracted text:\n{text[:500]}')
    else:
        print('No text was extracted. Verify PDF content and dependencies.')
        return

    print('Generating embeddings...')
    sentences, embeddings = generate_embeddings(text)

    if sentences:
        print(f'Generated {len(sentences)} sentences for the FAISS index:')
        print(sentences[:5])
    else:
        print('No meaningful sentences were generated from the extracted text.')
        return

    embeddings = np.array(embeddings, dtype='float32')
    if embeddings.size == 0:
        print('No embeddings generated. Check the input text and model.')
        return

    # Save the FAISS index
    save_index(embeddings, INDEX_PATH)
    print('Index has been successfully built.')

    # Save sentences to a file for later retrieval
    with open('embeddings/sentences.txt', 'w') as f:
        f.write('\n'.join(sentences))
    print('Sentences have been saved.')


# Processes a query using the FAISS index and LLM
def query_index(query):
    global sentences
    print('Loading the index...')

    # Load FAISS index
    if not os.path.exists(INDEX_PATH):
        print("FAISS index not found. Please build the index first.")
        return "Index is missing. Build the index before querying."

    index = load_index(INDEX_PATH)

    # Load sentences from the saved sentences file
    if not os.path.exists('embeddings/sentences.txt'):
        print("Sentences file not found. Please build the index first.")
        return "Sentences file is missing. Build the index before querying."

    with open('embeddings/sentences.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(sentences)} sentences from the index.")

    # Generate an embedding for the query
    print('Generating query embedding...')
    embedded_query = generate_embeddings(query)[1][0].reshape(1, -1)

    # Query FAISS index
    print('Querying FAISS...')
    distances, indices = query_faiss(index, embedded_query, top_k=5)

    # Retrieve sentences corresponding to the indices
    if indices is not None and len(sentences) > 0:
        retrieved_sentences = [sentences[idx] for idx in indices[0] if idx < len(sentences)]
        # Filter meaningful sentences only
        retrieved_sentences = [s for s in retrieved_sentences if len(s.split()) > 3]
        context = '\n'.join(retrieved_sentences)
        if not context.strip():  # If context is empty, fallback
            context = 'No relevant information found in the text.'
        print(f"Retrieved context:\n{context}")
    else:
        context = 'No relevant information found in the index.'
        print('No relevant information found.')

    # Generate a response using the LLM
    print('Generating a response...')
    response = generate_response(query, context)
    return response


if __name__ == '__main__':
    # Ensure index and sentences file exist; rebuild if missing
    if not os.path.exists(INDEX_PATH) or not os.path.exists('embeddings/sentences.txt'):
        print("Required files missing. Rebuilding index...")
        build_index()

    # Main query loop for questions
    while True:
        query = input('Enter your question (or type "exit" to quit): ')
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            answer = query_index(query)
            print(f'Answer: {answer}')
