import tensorflow as tf
from contextual_rewriting import get_contextual_rewritten_passage
from state_alignment import get_state_aligned_segments
from config import PATH
from inference import get_vectorization, get_subword2ipa

if __name__ == "__main__":
    # Load BanglaIPA model
    path = PATH['MODEL_PATH']
    banglaipa_model=tf.saved_model.load(path)
    bn_vectorization, ipa_vectorization = get_vectorization()
    print("BanglaIPA model loaded.\n")

    # Input passage
    passage = "১টি বাছাই করুন: গণিত ১ম/ গণিত ২য় পত্র। স্নাতক বা সমমান শ্রেণিতে প্রথম বর্ষের শিক্ষার্থীদের ভর্তি সহায়তা দেবে অন্তর্বর্তী সরকার। শিক্ষার্থীদের আবেদন চলবে আগামীকাল, ৩০ ডিসেম্বর পর্যন্ত অনলাইনে।"
    print(f"Original text={passage}\n")

    # Perform contextual rewriting
    passage_context = get_contextual_rewritten_passage(passage)
    # passage_context = "একটি বাছাই করুন: গণিত প্রথম/ গণিত দ্বিতীয় পত্র। স্নাতক বা সমমান শ্রেণিতে প্রথম বর্ষের শিক্ষার্থীদের ভর্তি সহায়তা দেবে অন্তর্বর্তী সরকার। শিক্ষার্থীদের আবেদন চলবে আগামীকাল, ত্রিশ ডিসেম্বর পর্যন্ত অনলাইনে।"
    print(f"Contextually Rewritten text={passage_context}\n")

    # Remove duplicate words
    DICTIONARY = {}
    words = passage_context.split()
    for word in words:
        if word not in DICTIONARY.keys():
            DICTIONARY[word] = ""
    
    # Apply State Alignment (STAT) algorithm, perform model-based IPA generation & merging
    for word in DICTIONARY:
        # State alignment
        states, tokens = get_state_aligned_segments(word)
        ipas = []
        try:
            for i in range(len(states)):
                subword = tokens[i]
                if states[i]:
                    # Model-based IPA generation
                    ipa = get_subword2ipa(subword, bn_vectorization, ipa_vectorization, banglaipa_model)
                    ipas.append(ipa)
                else:
                    ipas.append(subword)
            # Merging
            DICTIONARY[word] = "".join(ipas)
        except:
            pass
    # Rebuilding IPA sequence
    ipas = []
    for word in words:
        ipas.append(DICTIONARY[word])
    print(f"IPA={" ".join(ipas)}")

# python main.py

# # Output:
# BanglaIPA model loaded.
# Original text=১টি বাছাই করুন: গণিত ১ম/ গণিত ২য় পত্র। স্নাতক বা সমমান শ্রেণিতে প্রথম বর্ষের শিক্ষার্থীদের ভর্তি সহায়তা দেবে অন্তর্বর্তী সরকার। শিক্ষার্থীদের আবেদন চলবে আগামীকাল, ৩০ ডিসেম্বর পর্যন্ত অনলাইনে।

# Contextually Rewritten text=একটি বাছাই করুন: গণিত প্রথম/ গণিত দ্বিতীয় পত্র। স্নাতক বা সমমান শ্রেণিতে প্রথম বর্ষের শিক্ষার্থীদের ভর্তি সহায়তা দেবে অন্তর্বর্তী সরকার। শিক্ষার্থীদের আবেদন চলবে আগামীকাল, ত্রিশ ডিসেম্বর পর্যন্ত অনলাইনে।

# IPA=ekti bɐcʰɐ͡i̯ koɾun: gonit̪o pɾot̪ʰom/ gonit̪o d̪it̪iʲo pɔt̪ɾo। snɐt̪ɔk bɐ ʃɔmomɐn sɾenit̪e pɾot̪ʰom bɔɾʃeɾ ʃikkʰɐɾt̪ʰid̪eɾ bʱɔɾt̪i ʃɔhɐʲt̪ɐ d̪ebe ɔnt̪oɾboɾt̪i ʃɔɾkɐɾ। ʃikkʰɐɾt̪ʰid̪eɾ ɐbed̪ɔn colbe ɐgɐmikɐl, t̪ɾiʃ diʃembɔɾ poɾɟont̪o ɔnɔlɐ͡i̯ne।