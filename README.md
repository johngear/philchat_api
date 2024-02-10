# Philosophy Chat API
![cool pic of the School of Athens](https://github.com/johngear/philchat_api/blob/main/athens_copy.jpeg)

[![Netlify Status](https://api.netlify.com/api/v1/badges/dbd6b317-8660-4d36-b96d-4245b95b4195/deploy-status)](https://app.netlify.com/sites/jrg/deploys)

## Summary:
Try it here: https://philosophy-chat.com/ 

This is a design document explaining Philosophy Chat, an application created to give answers to academic philosophy questions. The backend is written in Python and hosted on an AWS EC2 instance, and the frontend in Svelte, a Javascript framework, hosted using Netlify. 

## Backend:
The backend is designed as an API that receives: 
1. a question as a text string, 
2. temperature, a parameter for the large language model (LLM),
3. a model (which LLM we want to call), currently ‘chat’ and ‘completion’ are supported

We return a data structure that contains:
1. an answer as a string,
2. a table, containing a variable number of rows, each with 5 columns that give information about the source. 

When the application starts up, the dataset of embedding vectors is loaded into memory as a FAISS Index (Facebook AI Semantic Search), a data structure that allows for quick vector search. This data lives in memory in my EC2 instance to allow for quicker answers, and to keep system design simpler, as it does not require a database.

When a question is asked, and the API is called the following occurs:
1. The question, a string, is embedded via a call to the OpenAI API and is returned as a vector. 
2. The embedding vector is compared against the FAISS index, and the top 10 most similar vectors are returned
3. A prompt which includes the text question and as many additional paragraphs of context as can fit in the context window length (which we set). An example is included at the end of this document.
4. The answer is returned as text, as is a table that includes the sources and information to the source of the API call.

## Hosting:
This runs on an AWS EC2 instance. A Load Balancer connects the Flask application on the EC2 to a URL where our frontend can make API calls. Using the Linux process manager ‘systemd’, if the program fails (and thus the data is no longer in memory), then it automatically restarts. This is ran by the `flask_api.service` file. It is crucial for latency to have the dataset in memory so that when an API call is made, the data is already in the FAISS index and vector search does not take so long. 

## Frontend:
Without much experience, I decided to use Svelte, a JS framework to create a simple frontend. This is hosted via Netlify and syncs to a Github repository. If I push changes, they are reflected in production immediately and if there is a mistake, it is easy to undo. I am not currently sharing this code.

## Data:
This application uses data from the Stanford Encyclopedia of Philosophy, an academic resource that has articles about nearly every relevant topic in philosophy. I originally found someone had already scraped this, but still had to do substantial data cleaning. Details of data cleaning can be seen in the other repo https://github.com/johngear/Encyclopedia-GPT 

There are approximately 160,000 paragraphs (which come from ~1,800 articles), stored as text and as vector embeddings in a Pandas dataframe. This data is used to augment the LLMs as it answers questions. 

It can be downloaded here:
https://huggingface.co/datasets/empiricist/philosophy_chat/tree/main

## To Run it locally:

This repository contains the production backend code (`application_aws.py`), and a version that can be run locally via command line (`application_local.py`).

To run locally, you will need to download my data from https://huggingface.co/datasets/empiricist/philosophy_chat/tree/main and update paths at the top of application_local . Also, you will have to create a config.py file in the /back folder with the following information:

```
OPENAI_API_KEY= "your key here"
COMPLETIONS_MODEL = "gpt-3.5-turbo-instruct"
CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
```

Information about the dependencies can be found in `requirements.txt`
