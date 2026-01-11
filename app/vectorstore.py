import faiss
import numpy as np

# Save FAISS index to file
def save_index(embeddings, output_path):
  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)
  index.add(embeddings)
  faiss.write_index(index, output_path)
  print(f'Index saved to {output_path}')

# Load FAISS index from file
def load_index(index_path):
  index = faiss.read_index(index_path)
  return index

# Example
if __name__ == '__main__':
  dummy_embeddings = np.random.random((100, 384)).astype('float32')
  save_index(dummy_embeddings, 'embeddings/index.faiss')
  index = load_index('embeddings/index.faiss')
  print(f'Index contains {index.total} vectors.')
