from openai import OpenAI
from utils import get_path, load_dataset_csv
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
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
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

def inference(df, model, output_path_jsonl):
    # pass
    for index, row in df.iterrows():
        text = row['text']
        ipa = get_text2ipa(text, model)
        print(f"text={text}")
        print(f"ipa={ipa}")
        break

if __name__ == "__main__":
    folder = PATH['project_folder_path'] + "/" + "data/input/test"
    paths = get_path(folder)
    print(f"Total files={len(paths)}")
    sum = 0

    model = "openai/gpt-4.1-mini"
    for path in paths:
        df = load_dataset_csv(path)
        df = df[['text', 'ipa']]
        # print(f"path={path} len={len(df)}")
        # print(df.head(5))

        inference(df, model, "")
        break
        sum += len(df)
    print(f"sum={sum}")
