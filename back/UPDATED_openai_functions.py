import pandas as pd
# import openai
import numpy as np
import tiktoken
import faiss
# from memory_profiler import profile

try:
    from back.openai_functions import get_embedding, load_embeddings, vector_similarity, compute_doc_embeddings
except ImportError:
    from openai_functions import get_embedding, load_embeddings, vector_similarity, compute_doc_embeddings

EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 4000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def UPDATE_order_document_sections_by_query_similarity(query: str, index):

    d = 1536 #dimensions

    #need this to be the format 
    query_embedding = np.array(get_embedding(query)).reshape(1,d)


    # # index = faiss.IndexFlatL2(d)   # build the index
    # index = faiss.IndexFlatIP(d)
    # index.add(contexts)            # add vectors to the index
    k = 10                         # we want to see 10 nearest neighbors

    _, I = index.search(query_embedding, k)     # actual search
    del _, index
    
    return I[:10] #return the top 10 indexes that match!


def UPDATE_construct_prompt(question: str, index, df: pd.DataFrame):
    """
    Fetch relevant 
    """

    # Base OpenAI function returns array of 2 dimension arrays, [float (similarity), (article, section, subsection, paragraph, index)]
    # EXAMPLE: (0.8572210044023278, ('Abduction', '2. Explicating Abduction', nan, 1, 22))

    #OLD
    # most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    most_relevant_document_sections = UPDATE_order_document_sections_by_query_similarity(question, index)

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

        # print("ADDED 1 SECTION TO THE RELEVENT INFO")
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    # print("\n\n")

    # num_selected_sections = len(chosen_sections)
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "This is a guess, but:" then answer the question \n\nContext:\n"""
    
    # print(header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_array)

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_array