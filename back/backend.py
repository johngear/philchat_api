import openai
import pandas as pd
import pickle
import time
import numpy as np
from memory_profiler import profile


# @profile
def main_backend(question: str, temp: float, model: str, index, df):

    try:
        from back.config import OPENAI_API_KEY, COMPLETIONS_MODEL, CHAT_MODEL
        from back.UPDATED_openai_functions import UPDATE_construct_prompt
    except ImportError:
        from config import OPENAI_API_KEY, COMPLETIONS_MODEL, CHAT_MODEL
        from UPDATED_openai_functions import UPDATE_construct_prompt


    openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
    openai.api_key = OPENAI_API_KEY
    openai.Model.list()

    start = time.time()
    prompt_sample, context_used_array_ints = UPDATE_construct_prompt(question, index, df)

    print(f'Time for Constructing Prompt: {time.time() - start}')

    filtered_df_contexts = df.loc[context_used_array_ints]
    filtered_df_contexts.drop(['index','pubinfo','text'], axis=1, inplace=True)
    json_df =  filtered_df_contexts.to_json(orient='records')

    if model == "chat":
        response = openai.ChatCompletion.create(
            model = CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a professor of philosophy with access to the Stanford Encyclopedia."},
                {"role": "user", "content": prompt_sample}
                ],
            temperature =temp
        )
        
        out = response['choices'][0]['message']['content'].strip(" \n")
        
    elif model == "completion":
        response = openai.Completion.create(
                    prompt=prompt_sample,
                    temperature=temp,
                    max_tokens=1000,
                    model=COMPLETIONS_MODEL,
                )

        out = response["choices"][0]["text"].strip(" \n")
    else:
        out = "ERROR WITH OPENAI CALL. Must specify if completion or chat, or rework the backend"
    
    # print(out, json_df)
    return out, json_df

if __name__ == "__main__":
    # main_backend()
    # ans, source = main_backend("what is the meaning of life", 0.8, "chat")
    # print(ans)
    pass