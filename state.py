

def get_state_aligned_segments(word, character_set):
    """
    Split a word into subword segments and assign state.    
    
    Args:
        - word (str): Input word
    Returns:
        - list of lists: Subword segments and state information (True/False)
    """
    i = 0
    n = len(word)
    vocab = list(character_set)
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

if __name__ == "__main__":
    character_set = set([ 'ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', '়', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ', 'ৗ', 'ড়', 'ঢ়', 'য়'])
    text = "একটি বাছাই করুন: গণিত প্রথম╱ গণিত দ্বিতীয় পত্র।"
    words = text.split(" ")
    print(words)
    for word in words:
        state, tokens = get_state_aligned_segments(word, character_set)
        print(state)
        print(tokens)
        print()
