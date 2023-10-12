from memory_profiler import profile
import numpy as np
import faiss
import gc

# Testing how the memory usage compares when using FAISS with Float32 versus Float64
# Important testing to reduce production hosting costs/errors
@profile
def run_faiss(d, num_vectors, dtype):
    print(f"Running FAISS with {dtype}")
    
    # Generate random vectors
    contexts = np.random.rand(num_vectors, d).astype(dtype)
    query_embedding = np.random.rand(1, d).astype(dtype)

    # Initialize and populate FAISS index
    index = faiss.IndexFlatIP(d)
    index.add(contexts)

    # Number of nearest neighbors to search for
    k = 10

    # Perform the search
    _, I = index.search(query_embedding, k)

    # Clear variables
    del contexts
    del query_embedding
    del index
    del I
    del _
    gc.collect()

@profile
def main():
    # Dimensions and number of vectors
    d = 1536  
    num_vectors = 160_000  
    
    # Run with float64
    run_faiss(d, num_vectors, 'float64')

    # Run with float32
    run_faiss(d, num_vectors, 'float32')


if __name__ == '__main__':
    main()