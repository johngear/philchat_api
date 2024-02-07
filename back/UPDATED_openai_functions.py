import pandas as pd
import numpy as np
import tiktoken
import openai

# try:
#     from back.openai_functions import get_embedding, load_embeddings, vector_similarity, compute_doc_embeddings
# except ImportError:
#     from openai_functions import get_embedding, load_embeddings, vector_similarity, compute_doc_embeddings

EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 4000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def UPDATE_order_document_sections_by_query_similarity(query: str, index):
    """
    This function takes in the string Question, and returns indexes to the top 10 most 
    related paragraphs (using dot product comparison of the vector embeddings).

    Uses FAISS library to search entire dataset. 
    """

    d = 1536 #length of embedding vector
    number_of_return_paragraphs = 10 

    query_embedding = np.array(get_embedding(query)).reshape(1,d)

    _, I = index.search(query_embedding, number_of_return_paragraphs)
    del _, index
    
    return I[:number_of_return_paragraphs]


def UPDATE_construct_prompt(question: str, index, df: pd.DataFrame):
    """
    This function constructs the entire prompt that will be given to the completion LLMS API. 
    It adds as many paragraphs of context from the sources as can fit as MAX_SECTION_LENGTH allows, 
    and returns a long string and an array that contains the sources used. 

    The string 'header' could be changed to try different types of prompting.
    """
    most_relevant_document_sections = UPDATE_order_document_sections_by_query_similarity(question, index)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    chosen_sections_array = []
    for section_index in most_relevant_document_sections[0]:

        #the unique index of a relevant paragraph found
        document_section = df.loc[df['index']==section_index] 
        
        chosen_sections_len += len(document_section.text.item()) + separator_len
            
        chosen_sections.append(SEPARATOR + document_section.text.item().replace("\n", " ").strip())
        chosen_sections_indexes.append(str(section_index))
        chosen_sections_array.append(section_index)

        #add 1 paragraph of context, even if it is very long, then break after
        if chosen_sections_len > MAX_SECTION_LEN: 
            break
            
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "This is a guess, but:" then answer the question \n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_array


def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    """
    Returns the vector embedding for a string.
    """

    result = openai.Embedding.create(
      model=model,
      input=text
    )
    
    return result["data"][0]["embedding"]