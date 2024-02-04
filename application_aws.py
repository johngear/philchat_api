from flask import Flask, request
from flask_cors import CORS
from back.backend import main_backend
from back.config import EC2_API_URL

import pickle
import time
import pandas as pd
import faiss

#paths to the embeddings and to the full dataset CSV
path_to_pickle = '/home/ec2-user/flask_api/back/data/embeddings_full.pickle' 
path_to_dataset = '/home/ec2-user/flask_api/back/data/FULL_DATA_new.csv'

application = Flask(__name__)
CORS(application, origins=['http://localhost:5000', 'http://localhost:4173', 'https://jrg.netlify.app', 'https://philosophy-chat.com', 'http://localhost:5173', EC2_API_URL])

@application.route('/q_and_a', methods=['POST'])
def compute():
    data = request.get_json()  # Get the data from the request body

    #From the API call, we get the Question, the Temperature and the Model that we want to answer with
    out, context = main_backend(data['q'], data['temp'], data['model'], index, full_dataset)

    #In return, we get the text out, and an array of the context that we used.
    return {'result': out, 
            'context': context}

index, full_dataset = None, None


@application.route('/health', methods=['GET'])
def health_check():
    return "OK", 200


def load_data():
    global index, full_dataset

    with open(path_to_pickle, 'rb') as file:
        doc_embeddings = pickle.load(file)
        doc_embeddings = doc_embeddings.astype('float32')

    full_dataset = pd.read_csv(path_to_dataset) 
    full_dataset = full_dataset.rename(columns={'Unnamed: 0': 'index'})

    index = faiss.IndexFlatIP(1536)
    index.add(doc_embeddings)      
    del doc_embeddings

load_data()

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8000)

