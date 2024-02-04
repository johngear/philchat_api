import openai
import pandas as pd

def main_backend(question: str, temp: float, model: str, index, df):
    """
    The main backend, which takes a question and returns the answer and sources
    in a format that the Front End can parse.
    """

    try:
        from back.config import OPENAI_API_KEY, COMPLETIONS_MODEL, CHAT_MODEL
        from back.UPDATED_openai_functions import UPDATE_construct_prompt
    except ImportError:
        from config import OPENAI_API_KEY, COMPLETIONS_MODEL, CHAT_MODEL
        from UPDATED_openai_functions import UPDATE_construct_prompt


    openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
    openai.api_key = OPENAI_API_KEY
    openai.Model.list()

    #get the long string prompt from the input question and our data
    prompt_sample, context_used_array_ints = UPDATE_construct_prompt(question, index, df)

    #format the table of our source paragraphs for readability
    filtered_df_contexts = df.loc[context_used_array_ints]
    filtered_df_contexts.drop(['index','pubinfo','text'], axis=1, inplace=True)
    json_df =  filtered_df_contexts.to_json(orient='records')

    #Call OpenAI API and get answer
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
        """
        gpt-3.5-turbo-instruct is being used instead of text-davinci-003 now
        """
        response = openai.Completion.create(
                    prompt=prompt_sample,
                    temperature=temp,
                    max_tokens=1000,
                    model=COMPLETIONS_MODEL,
                )

        out = response["choices"][0]["text"].strip(" \n")
    else:
        out = "Error. Need Chat or Completion"
    
    return out, json_df