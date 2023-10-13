## These functions are taken from OpenAI cookbook
## There are slight changes, but I do not claim as my own work

import pandas as pd
import openai
import numpy as np
import tiktoken
import faiss

EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    """
    Returns the vector embedding for a string.
    """

    result = openai.Embedding.create(
      model=model,
      input=text
    )
    
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame): 
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.text) for idx, r in df.iterrows()
    }

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts):


    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    #TRYING SOMEHTING NEW
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    # document_similarities = sorted([
    #     (vector_similarity(query_embedding, doc_embedding), doc_index) 
    #     if all(isinstance(val, (int, float)) for val in doc_embedding) 
    #     else (float(0), doc_index)
    #     for doc_index, doc_embedding in contexts.items()
    # ], reverse=True)

    # document_similarities = sorted([
    #     (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    # ], reverse=True)
    
    return document_similarities[:10]

def NEW_order_document_sections_by_query_similarity(query: str, contexts):

    d = 1536 #dimensions

    #need this to be the format 
    query_embedding = np.array(get_embedding(query)).reshape(1,d) 

    # index = faiss.IndexFlatL2(d)   # build the index
    index = faiss.IndexFlatIP(d)
    index.add(contexts)            # add vectors to the index
    k = 10                         # we want to see 5 nearest neighbors

    D, I = index.search(query_embedding, k)     # actual search
    
    return I[:10] #return the top 10 indexes that match!




def load_embeddings(fname: str): # -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


MAX_SECTION_LEN = 4000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame):
    """
    Fetch relevant 
    """

    # Base OpenAI function returns array of 2 dimension arrays, [float (similarity), (article, section, subsection, paragraph, index)]
    # EXAMPLE: (0.8572210044023278, ('Abduction', '2. Explicating Abduction', nan, 1, 22))

    #OLD
    # most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    most_relevant_document_sections = NEW_order_document_sections_by_query_similarity(question, context_embeddings)

    # chosen_sections = []
    # chosen_sections_len = 0
    # chosen_sections_indexes = []
    # chosen_sections_array = []
    # for _, section_index in most_relevant_document_sections:
    #     document_section = df.loc[section_index]
        
    #     chosen_sections_len += len(document_section.text) + separator_len
    #     if chosen_sections_len > MAX_SECTION_LEN:
    #         break
            
    #     chosen_sections.append(SEPARATOR + document_section.text.replace("\n", " "))
    #     chosen_sections_indexes.append(str(section_index))
    #     chosen_sections_array.append(section_index)

    #NEW CODE
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    chosen_sections_array = []
    for section_index in most_relevant_document_sections[0]:

        document_section = df.loc[df['index']==section_index] #this should be the row where the magic happens
        # document_section = row["text"]
        
        chosen_sections_len += len(document_section.text.item()) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.text.item().replace("\n", " ").strip())
        chosen_sections_indexes.append(str(section_index))
        chosen_sections_array.append(section_index)
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    print("\n\n")

    # num_selected_sections = len(chosen_sections)
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_array