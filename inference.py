import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np
from config import PATH

def get_vocab():
  """
  Returns sorted list of Bengali characters, IPA characters, special tokens and other characters seen in the training set.
  """
  vb = ['', '[UNK]', '[start]', '[end]', 'া', 'র', '্', 'ে', 'ি', 'ন', 'ক', 'ব', 'স', 'ল', 'ত', 'ম', 'প', 'ু', 'দ', 'ট', 'য়', 'জ', '।', 'ো', 'গ', 'হ', 'য', 'শ', 'ী', 'ই', 'চ', 'ভ', 'আ', 'ও', 'ছ', 'ষ', 'ড', 'ফ', 'অ', 'ধ', 'খ', 'ড়', 'উ', 'ণ', 'এ', 'থ', 'ং', 'ঁ', 'ূ', 'ৃ', 'ঠ', 'ঘ', 'ঞ', 'ঙ', 'ৌ', '‘', 'ৎ', 'ঝ', 'ৈ', '়', 'ঢ', 'ঃ', 'ঈ', '\u200c', 'ৗ', 'a', 'ঐ', 'd', 'w', 'ঋ', 'i', 'e', 't', 's', 'n', 'm', 'b', '“', 'u', 'r', 'œ', 'o', '–', 'ঊ', 'ঢ়', 'Í', 'g', 'p', '\xad', 'h', 'c', 'l', 'ঔ', 'ƒ', '”', 'Ñ', '¡', 'y', 'j', 'f', '→', '—', 'ø', 'è', '¦', '¥', 'x', 'v', 'k']
  vipa = ['', '[UNK]', '[start]', '[end]', 'ɐ', 'ɾ', 'i', 'o', 'e', '̪', 't', 'n', 'k', 'ɔ', 'ʃ', 'b', 'd', 'l', 'u', 'p', 'm', 'ʰ', 'ɟ', '͡', '̯', 'g', 'ʱ', '।', 'c', 'ʲ', 'h', 's', 'ŋ', 'ɛ', 'ɽ', '̃', 'ʷ', '‘', '“', '–', '”', '—', 'w', 'j']
  v = vb + vipa
  s = set()
  for ch in v:
    s.add(ch)
  vocab = sorted(list(s))
  return vocab

def get_vectorization():
  """
  Performs vectorization.
  """
  vocab = get_vocab()
  vocab_size = len(vocab)
  sequence_length = 64
  bn_vectorization = TextVectorization(
      max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
      vocabulary=vocab
  )
  ipa_vectorization = TextVectorization(
      max_tokens=vocab_size,
      output_mode="int",
      output_sequence_length=sequence_length + 1,
      vocabulary=vocab
  )
  return bn_vectorization, ipa_vectorization

def decode_sequence(input_sentence, bn_vectorization, ipa_vectorization, banglaipa_model):
    """
    Generate IPA for subword.
    
    Args:
      - input_sentence (str): Synthetic sentence where every adjacent characters has a space between them.
      - bn_vectorization: TextVectorization
      - en_vectorization: TextVectorization
      - banglaipa_model: Transformer model
    Returns:
      - str: String of IPA characters and special tokens where adjacent characters are separated with a space.
    """
    max_decoded_sentence_length = 64
    spa_vocab = ipa_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    tokenized_input_sentence = bn_vectorization([input_sentence])
    decoded_sentence = '[start]'
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = ipa_vectorization([decoded_sentence])[:, :-1]
        predictions = banglaipa_model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == '[UNK]':
            break
    return decoded_sentence

def sentence_to_word(sentence):
  """
  Generate word from synthetic sentence by removing spaces between adjacent characters.

  Args:
    - sentence (str): Synthetic sentence.
  Returns:
    - str: subword/word
  """
  trg=''
  for ch in sentence:
      if ch != " ":
        trg += ch
  return trg

def word_to_sentence(word):
  """
  Generate synthetic sentence from word by inserting spaces between adjacent characters.

  Args:
    - word (str): subword/word segement
  Returns:
    - str: Synthetic sentence
  """
  sentence = ""
  for ch in word:
    sentence += (ch + " ")
  return sentence

def get_subword2ipa(word, bn_vectorization, ipa_vectorization, banglaipa_model):
    translated = decode_sequence(word_to_sentence(word), bn_vectorization, ipa_vectorization, banglaipa_model)
    trg = sentence_to_word(translated)
    trg = trg[7:]
    trg = trg[:-5]
    return trg

if __name__ == "__main__":
    path = PATH['MODEL_PATH']
    banglaipa_model=tf.saved_model.load(path)
    print("BanglaIPA model loaded.")
    bn_vectorization, ipa_vectorization = get_vectorization()
    text = "একটি বাছাই করুন গণিত প্রথম গণিত দ্বিতীয় পত্র"
    ipa = ""
    words = text.split(" ")
    for word in words:
        trg = get_subword2ipa(word, bn_vectorization, ipa_vectorization, banglaipa_model)
        print(word, trg)
        ipa += (trg + " ")
    print(f"IPA={ipa}")

## python inference.py
# # Output:
# BanglaIPA model loaded.
# একটি ekti
# বাছাই bɐcʰɐ͡i̯
# করুন koɾun
# গণিত gonit̪o
# প্রথম pɾot̪ʰom
# গণিত gonit̪o
# দ্বিতীয় d̪it̪iʲo
# পত্র pɔt̪ɾo
# IPA=ekti bɐcʰɐ͡i̯ koɾun gonit̪o pɾot̪ʰom gonit̪o d̪it̪iʲo pɔt̪ɾo
      
