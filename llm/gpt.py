from openai import OpenAI
from utils import get_path
import pandas as pd
from config import PATH

def get_text2ipa(text, model):
    """
    Provides IPA transcription.
    
    Args:
        - text (str): Bangla text
        - model (str): Name of the model
    Returns:
        - str: IPA transcription
    """
    print(text, model)
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
                'content': 'You are a helpful chatbot who knows how to convert Bangla text to the International Phonetic Alphabet (IPA) transcription. Please be concise and do not add any explanation.'
            },
            {
                "role": "user", 
                'content': f"Please convert the given Bangla text into the International Phonetic Alphabet (IPA) transcription\n\"{text}\""
            }
        ],
    )
    text = completion.choices[0].message.content
    return text

def get_contextual_rewritten_text(text, model):
    """
    Provides word forms of numerals.
    
    Args:
        - text (str): Bangla text
        - model (str): Name of the model
    Returns:
        - str: Text where numbers are converted to the word forms.
    """
    print(text, model)
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
        # ipa = get_text2ipa(text, model)
        ipa = get_contextual_rewritten_text(text, model)
        print(f"text={text}")
        print(f"ipa={ipa}")
        break

if __name__ == "__main__":
    folder = PATH['project_folder_path'] + "/" + "data/input/test"
    paths = get_path(folder)
    print(f"Total files={len(paths)}")
    sum = 0

    print(paths)

    model = "openai/gpt-4.1-mini"
    for path in paths:
        # df = load_dataset_csv(path)
        df = pd.read_json(path)
        df = df[['text', 'ipa']]
        # print(df)
        # print(f"path={path} len={len(df)}")
        # print(df.head(5))

        inference(df, model, "")
        break
        sum += len(df)
    print(f"sum={sum}")
