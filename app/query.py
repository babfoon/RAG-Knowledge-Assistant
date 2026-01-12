from transformers import pipeline
import faiss 
import numpy as np

# Loading of LLM for generation
llm = pipeline('text-generation', model='gpt2')

# Retrieve top-k results from FAISS
def query_faiss(index, query_embedding, top_k=5):
  distances, indices = index.search(query_embedding, top_k)
  return distances, indices

# Generate a response from the model
def generate_response(query, context):
  input_text = f'Context: \n{context}\n\nQuestion: {query}\n'
  response = llm(input_text, max_length = 150, num_return_sequences = 1)
  return response[0] ['generated_text']

# Example retrieval
if __name__ == '__main__':
  query_embedding = np.random.random((1,384)).astype('float32')
  index = faiss.read_index('embeddings/index.faiss')
  distances, indices = query_faiss(index, query_embedding)

# Example generation
context = 'Document 1 Content: ... (retrieved text)'
print (generate_response('What is the purpose of this document?', context))
