from utils import is_number_present
from llm.gpt import get_contextual_rewritten_text

def get_contextual_rewritten_passage(passage):
    """
    Convert all numbers to word forms.
    
    Args:
        - passage (str): Input passage
    Returns:
        - str: Passage where numbers are converted to word forms
    """
    texts = passage.strip().split("।")
    text_with_context = []
    for text in texts:
        text = text.strip()
        if len(text) <= 0:
            continue
        text += "।"
        if is_number_present(text):
            text_with_context.append(get_contextual_rewritten_text(text))
        else:
            text_with_context.append(text)
    # passage_rewritten = ("। ".join(text_with_context)).strip() + "।"
    return " ".join(text_with_context)

if __name__ == "__main__":
    passage = "১টি বাছাই করুন: গণিত ১ম/ গণিত ২য় পত্র। স্নাতক বা সমমান শ্রেণিতে প্রথম বর্ষের শিক্ষার্থীদের ভর্তি সহায়তা দেবে অন্তর্বর্তী সরকার। শিক্ষার্থীদের আবেদন চলবে আগামীকাল, ৩০ ডিসেম্বর পর্যন্ত অনলাইনে।"
    print(get_contextual_rewritten_passage(passage))

## python contextual_rewriting.py