"""
This can be run locally, via command line to test out the functionality. 

However, the data must be local too.

#TODO find a good way to share large dataset
"""
from back.backend import main_backend

import pickle
import time
import pandas as pd
import faiss

path_to_pickle_of_all_embeddings = '/Users/johngearig/Documents/GitHub/phil_gpt_flask_api/back/data/embeddings_full.pickle'
path_to_csv_of_whole_dataset = '/Users/johngearig/Documents/GitHub/phil_gpt_flask_api/back/data/FULL_DATA_new.csv'


def compute(data):
    out, context = main_backend(data['q'], data['temp'], data['model'], index, whole_dataset)
    return {'result': out, 
            'context': context}

index, whole_dataset = None, None
def load_data():
    print("""\n\n\n
            Starting up Philosophy Chat!
            \n\n\n""")
    global index, whole_dataset
    
    start = time.time()

    with open(path_to_pickle_of_all_embeddings, 'rb') as file:
        doc_embeddings = pickle.load(file)
        doc_embeddings = doc_embeddings.astype('float32')

    whole_dataset = pd.read_csv(path_to_csv_of_whole_dataset).rename(columns={'Unnamed: 0': 'index'})

    index = faiss.IndexFlatIP(1536)
    index.add(doc_embeddings)      
    del doc_embeddings

    print(f'Time for Loading Dataset into Faiss Index: {time.time() - start}')

if __name__ == '__main__':
    load_data()

    print("Type Exit to stop\n\n")

    while True:
        input_question = input("Ask a philosophy question:   ")

        if input_question.lower() == "exit":
            print("All done\n\n\n")
            break

        input_data = {'q':input_question, 
                    'temp':0.2, 
                    'model': "chat"}
        out = compute(input_data)

        print("\n\nAnswer:   ",out['result'],"\n")
        # print("\n\nContext:\n\n",out['context'])

