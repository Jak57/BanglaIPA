from openai import OpenAI
import pandas as pd
from config import PATH
from utils import get_path

def get_contextual_rewritten_text(text, model="openai/gpt-4.1-nano"):
    """
    Provides word forms of numerals.
    
    Args:
        - text (str): Bangla text
        - model (str): Name of the model
    Returns:
        - str: Text where numbers are converted to the word forms.
    """
    client = OpenAI(
        base_url=PATH['BASE_URL'],
        api_key=PATH['API_KEY'],
    )
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "none", 
            "X-Title": "none", 
        },
        extra_body={},
        max_tokens=500,
        model= f"{model}", 
        messages= [
            {
                'role': "system",
                'content': 'You are a helpful chatbot who understands Bengali numerals in different contexts.'
            },
            {
                "role": "user", 
                'content': f"Please rewrite the provided text so that no Bengali digits are present. Convert the numbers to word form based on the context. Do not modify any words. Here is the text:\n\"{text}\""
            }
        ],
    )
    text = completion.choices[0].message.content
    return text

def inference(df, model, output_path_jsonl):
    for index, row in df.iterrows():
        text = row['text']
        text = "১টি বাছাই করুন: ১ম/ ২য় পত্র।"
        text_context = get_contextual_rewritten_text(text, model)
        print(f"text={text}")
        print(f"text_context={text_context}")
        break

if __name__ == "__main__":
    folder = PATH['project_folder_path'] + "/" + "data/input/test"
    paths = get_path(folder)
    print(f"Total files={len(paths)}")
    model = "openai/gpt-4.1-nano"
    sum = 0
    for path in paths:
        df = pd.read_json(path)
        df = df[['text', 'ipa']]
        inference(df, model, "")
        sum += 1
        break
    print(f"sum={sum}")
