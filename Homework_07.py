import nltk

def findAndSortAllByTag(taged_tokens, tag):
    tokens = []
    for taged_token in taged_tokens:
        if taged_token[1] == tag:
            tokens.append(taged_token[0])
    return sorted(list(set(tokens)))

def findAndAllByTagSeq(taged_tokens, tag_sepqence):
    tokens = []
    for i in range(len(taged_tokens) -1 -len(tag_sepqence)):
        for count, searchTag in enumerate(tag_sepqence):
            tokenTag = taged_tokens[i +count][1]
            if not tokenTag == searchTag:
                break
        else:
            tokens.append([taged_token[0] for taged_token in taged_tokens[i:i +len(tag_sepqence)]])
    return tokens

if __name__ == '__main__':
    words = nltk.corpus.brown.words()
    taged_tokens = nltk.pos_tag(words)

    print("\n(a) Produce an alphabetically sorted list of the distinct words tagged as MD.")
    MDtaged_tokens = findAndSortAllByTag(taged_tokens, "MD")
    print(MDtaged_tokens)

    print("\n(c) Identify three-word prepositional phrases of the form ADP + DET + NOUN (eg. 'at the end').")
    sepTaged_tokens = findAndAllByTagSeq(taged_tokens, ["RB", "DT", "NN"])
    print(sepTaged_tokens)
