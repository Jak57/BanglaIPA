import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
import numpy as np
from config import PATH

def get_vocab():
  vb = ['', '[UNK]', '[start]', '[end]', 'া', 'র', '্', 'ে', 'ি', 'ন', 'ক', 'ব', 'স', 'ল', 'ত', 'ম', 'প', 'ু', 'দ', 'ট', 'য়', 'জ', '।', 'ো', 'গ', 'হ', 'য', 'শ', 'ী', 'ই', 'চ', 'ভ', 'আ', 'ও', 'ছ', 'ষ', 'ড', 'ফ', 'অ', 'ধ', 'খ', 'ড়', 'উ', 'ণ', 'এ', 'থ', 'ং', 'ঁ', 'ূ', 'ৃ', 'ঠ', 'ঘ', 'ঞ', 'ঙ', 'ৌ', '‘', 'ৎ', 'ঝ', 'ৈ', '়', 'ঢ', 'ঃ', 'ঈ', '\u200c', 'ৗ', 'a', 'ঐ', 'd', 'w', 'ঋ', 'i', 'e', 't', 's', 'n', 'm', 'b', '“', 'u', 'r', 'œ', 'o', '–', 'ঊ', 'ঢ়', 'Í', 'g', 'p', '\xad', 'h', 'c', 'l', 'ঔ', 'ƒ', '”', 'Ñ', '¡', 'y', 'j', 'f', '→', '—', 'ø', 'è', '¦', '¥', 'x', 'v', 'k']
  vipa = ['', '[UNK]', '[start]', '[end]', 'ɐ', 'ɾ', 'i', 'o', 'e', '̪', 't', 'n', 'k', 'ɔ', 'ʃ', 'b', 'd', 'l', 'u', 'p', 'm', 'ʰ', 'ɟ', '͡', '̯', 'g', 'ʱ', '।', 'c', 'ʲ', 'h', 's', 'ŋ', 'ɛ', 'ɽ', '̃', 'ʷ', '‘', '“', '–', '”', '—', 'w', 'j']
  v = vb + vipa
  s = set()
  for ch in v:
    s.add(ch)
  vocab = sorted(list(s))
  return vocab

def get_vectorization():
  vocab = get_vocab()
  vocab_size = len(vocab)
  sequence_length = 64
  eng_vectorization = TextVectorization(
      max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
      vocabulary=vocab
  )
  spa_vectorization = TextVectorization(
      max_tokens=vocab_size,
      output_mode="int",
      output_sequence_length=sequence_length + 1,
      vocabulary=vocab
  )
  return eng_vectorization, spa_vectorization

def decode_sequence(input_sentence, eng_vectorization, spa_vectorization, new_model):
    max_decoded_sentence_length = 64
    spa_vocab = spa_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = '[start]'
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = new_model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == '[UNK]':
            break
    return decoded_sentence

def sentence_word(sentence):
  trg=''
  for ch in sentence:
      if ch != " ":
        trg += ch
  return trg

def word_sentence(word):
  sentence = ""
  for ch in word:
    sentence += (ch + " ")
  return sentence

def get_subword2ipa(word, eng_vectorization, spa_vectorization, new_model):
    translated = decode_sequence(word_sentence(word), eng_vectorization, spa_vectorization, new_model)
    trg = sentence_word(translated)
    trg = trg[7:]
    trg = trg[:-5]
    return trg

if __name__ == "__main__":
    path = PATH['MODEL_PATH']
    new_model=tf.saved_model.load(path)
    print("BanglaIPA model loaded.")
    eng_vectorization, spa_vectorization = get_vectorization()
    text = "একটি বাছাই করুন গণিত প্রথম গণিত দ্বিতীয় পত্র"
    ipa = ""
    words = text.split(" ")
    for word in words:
        trg = get_subword2ipa(word, eng_vectorization, spa_vectorization, new_model)
        print(word, trg)
        ipa += (trg + " ")
    print(ipa)

## python inference.py
      
