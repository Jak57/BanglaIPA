import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
import numpy as np
from config import PATH

path = PATH['MODEL_PATH']
new_model=tf.saved_model.load(path)

print("Model loaded")

vb = ['', '[UNK]', '[start]', '[end]', 'া', 'র', '্', 'ে', 'ি', 'ন', 'ক', 'ব', 'স', 'ল', 'ত', 'ম', 'প', 'ু', 'দ', 'ট', 'য়', 'জ', '।', 'ো', 'গ', 'হ', 'য', 'শ', 'ী', 'ই', 'চ', 'ভ', 'আ', 'ও', 'ছ', 'ষ', 'ড', 'ফ', 'অ', 'ধ', 'খ', 'ড়', 'উ', 'ণ', 'এ', 'থ', 'ং', 'ঁ', 'ূ', 'ৃ', 'ঠ', 'ঘ', 'ঞ', 'ঙ', 'ৌ', '‘', 'ৎ', 'ঝ', 'ৈ', '়', 'ঢ', 'ঃ', 'ঈ', '\u200c', 'ৗ', 'a', 'ঐ', 'd', 'w', 'ঋ', 'i', 'e', 't', 's', 'n', 'm', 'b', '“', 'u', 'r', 'œ', 'o', '–', 'ঊ', 'ঢ়', 'Í', 'g', 'p', '\xad', 'h', 'c', 'l', 'ঔ', 'ƒ', '”', 'Ñ', '¡', 'y', 'j', 'f', '→', '—', 'ø', 'è', '¦', '¥', 'x', 'v', 'k']
vipa = ['', '[UNK]', '[start]', '[end]', 'ɐ', 'ɾ', 'i', 'o', 'e', '̪', 't', 'n', 'k', 'ɔ', 'ʃ', 'b', 'd', 'l', 'u', 'p', 'm', 'ʰ', 'ɟ', '͡', '̯', 'g', 'ʱ', '।', 'c', 'ʲ', 'h', 's', 'ŋ', 'ɛ', 'ɽ', '̃', 'ʷ', '‘', '“', '–', '”', '—', 'w', 'j']
v = vb + vipa
s = set()
for ch in v:
  s.add(ch)

vocab = sorted(list(s))
print("Length of vocab:", len(s))
print(vocab)
vocab_size = len(vocab)

sequence_length = 64 # 20
batch_size = 64

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

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 64 #20

def decode_sequence(input_sentence):
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

def bangla_vocabulary():
  Vowels = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'ঌ', 'এ', 'ঐ', 'ও', 'ঔ']
  Vowel_signs = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ে', 'ৈ', 'ো', 'ৌ']
  Consonants = ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ']
  Operators = ['=', '+', '-', '*', '/', '%', '<', '>', '×', '÷']
  Punctuation_marks = ['।', ',', ';', ':', '?', '!', "'", '.', '"', '-', '[', ']', '{', '}', '(', ')', '–', '—', '―', '~']
  Others = ['্', '়', 'ৗ', '‘', '’', '“', '”']

  BANGLA_VOCAB = sorted(list(set(Vowels + Vowel_signs + Consonants +  Operators + Punctuation_marks + Others)))
  return BANGLA_VOCAB

def foreign_character_normalization(word):
  BANGLA_VOCAB = bangla_vocabulary()
  normalized_word = ""

  for ch in word:
    if ch not in BANGLA_VOCAB:
      continue
    normalized_word += ch
  return normalized_word

def aligned_stateful_tokenizer(word):
  vocab = [ 'ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', '়', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ', 'ৗ', 'ড়', 'ঢ়', 'য়']
  n = len(word)
  i = 0
  j = n-1

  state = []
  tokens = []

  while i < n:
    subword = ""
    if word[i] in vocab:
      found = True
      while i < n and word[i] in vocab:
        subword += word[i]
        i += 1

    elif not(word[i] in vocab):
      found = False
      while i < n and not(word[i] in vocab):
        subword += word[i]
        i += 1

    state.append(found)
    tokens.append(subword)
  return state, tokens

def preprocess(word):
  preprocessed_word = foreign_character_normalization(word)
  return preprocessed_word

if __name__ == "__main__":
    text = "একটি বাছাই করুন গণিত প্রথম গণিত দ্বিতীয় পত্র"
    # text = "একটি বাছাই" # করুন গণিত প্রথম গণিত দ্বিতীয় পত্র"
    ipa = ""

    words = text.split(" ") #[0]
    for word in words:
        translated = decode_sequence(word_sentence(word))
        trg = sentence_word(translated)
        trg = trg[7:]
        trg = trg[:-5]
        # tokens[i] = trg
        print(word, trg)

        ipa += (trg + " ")
    print(ipa)
      
