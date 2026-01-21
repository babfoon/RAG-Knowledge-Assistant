import os
import faiss
import numpy as np

# Save FAISS index to file
def save_index(embeddings, output_path):
  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)

  # Add a debug check for empty embeddings
  if embeddings.shape[0] == 0:
    print('No embeddings found; index creation aborted.')
    return

  index.add(embeddings)
  print(f'Added {embeddings.shape[0]} vectors with dimension {dim} to the index.')
  faiss.write_index(index, output_path)
  print(f'Index saved to {output_path}')

# Load FAISS index from file
def load_index(index_path):
    # Check if the FAISS index file exists
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file at {index_path} does not exist. "
                                "Please run the build_index() function to create the index.")
    index = faiss.read_index(index_path)
    return index

# Example
if __name__ == '__main__':
  dummy_embeddings = np.random.random((100, 384)).astype('float32')
  save_index(dummy_embeddings, 'embeddings/index.faiss')
  index = load_index('embeddings/index.faiss')
  print(f'Index contains {index.ntotal} vectors.')
