import os
from app.preprocess import process_pdf, generate_embeddings
from app.vectorstore import save_index, load_index
from app.query import query_faiss, generate_response
import numpy as np

DOCUMENT_PATH = 'data/A Course of Pure Mathmatics (1921).pdf'
INDEX_PATH = 'embeddings/index.faiss'

def build_index():
  print('Processing the document...')
  text = process_pdf(DOCUMENT_PATH)
  sentences, embeddings = generate_embeddings(text)

  embeddings = np.array (embeddings, dtype = 'float32')
  save_index(embeddings, INDEX_PATH)
  print('Index has been successfully built.')

def query_index(query):
  print('Loading the index...')
  index = load_index(INDEX_PATH)

  embedded_query = generate_embeddings(query)[1][0].reshape(1,-1)

  print('Querying FAISS...')
  distances, indices = query_faiss(index, embedded_query, top_k=5)

  context = 'Retrieved document context based on your query...'

  print('Generating a response...')
  response = generate_response(query, context)
  return response

if __name__ == '__main__':
  if not os.path.exists(INDEX_PATH):
    build_index()

  query = input ('Enter your question:')
  answer = query_index(query)
  print (f'Answer: {answer}')

  
