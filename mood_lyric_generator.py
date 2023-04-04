import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def readMemory():
    song_dataframe = pd.read_csv("songdata.csv")
    artist = np.array(song_dataframe['artist'])
    song = np.array(song_dataframe['song'])
    link = np.array(song_dataframe['link'])
    text = np.array(song_dataframe['text'])
    id = list()

    # artist.size
    for i in range(0, artist.size):
        temp = ''
        for j in artist[i]:
            if j.isupper():
                temp += j.lower()
            elif j == ' ':
                temp += '_'
            else:
                temp += j
        temp += '-'
        for k in song[i]:
            if k.isupper():
                temp += k.lower()
            elif k == ' ':
                temp += '_'
            else:
                temp += k
        id.append(temp)

    id = np.array(id)
    result = list()
    result.append(artist)
    result.append(song)
    result.append(link)
    result.append(text)
    result.append(id)

    return result


import re  # python's regular expression package


def tokenize(sent):
    # input: a single sentence as a string.
    # output: a list of each "word" in the text
    # must use regular expressions

    tokens = []
    # <FILL IN>
    tokens = re.split(r'(\<[^\>]*\>)|((?:[A-Z]\.)+)|((?:#[\w|\-|\']+\b))|((?:@[\w|\-|\']+\b))|((?:\b[\w|\-|\']+\b))|([^ ])', sent)

    tokens = list(filter(None, tokens))

    while '' in tokens:
        tokens.remove('')

    while '\'' in tokens:
        tokens.remove('\'')

    while ' ' in tokens:
        tokens.remove(' ')

    while '  ' in tokens:
        tokens.remove('  ')

    while ':' in tokens:
        tokens.remove(':')

    while '(' in tokens:
        tokens.remove('(')
    while ')' in tokens:
        tokens.remove(')')

    while '[' in tokens:
        tokens.remove('[')

    while ']' in tokens:
        tokens.remove(']')

    while '{' in tokens:
        tokens.remove('{')

    while '}' in tokens:
        tokens.remove('}')

    while '[' in tokens:
        tokens.remove('[')

    while ']' in tokens:
        tokens.remove(']')

    return tokens

def tokenize_stuff(title_list, lyric_list):
    lyric_begin = "<s>"
    lyric_end = "</s>"

    lyric_token = list()

    for i in lyric_list:
        new_edit = lyric_begin
        for j in i:
            if j == "\n":
                new_edit += "<newline>"
            else:
                new_edit += j
        new_edit += lyric_end
        lyric_token.append(tokenize(new_edit))

    lyric_token = np.array(lyric_token)
    title_token = list()

    for i in title_list:
        title_token.append(tokenize(i))

    title_token = np.array(title_token)

    result = list()
    result.append(title_token)
    result.append(lyric_token)

    return result


def word_vocabulary(lyrics):
    vocab = {}
    for i in lyrics:
        for j in i:
            if j in vocab:
                vocab[j] = vocab[j]+1
            else:
                vocab[j] = 1

    vocab['<OOV>'] = 0

    for k in list(vocab):
        if vocab[k] < 3:
            vocab['<OOV>'] = vocab['<OOV>']+vocab[k]
            vocab.pop(k)

    return vocab


# Vocab and lyric are both single dictionary and single list of lyric tokens.
def bigram_matrix(vocab, lyric_tokens):
    bigram_matrix = {}

    for lyric_token in lyric_tokens:
        for i in range(len(lyric_token) - 1):

            token_i = lyric_token[i] if (lyric_token[i] in vocab) else '<OOV>'
            token_j = lyric_token[i + 1] if (lyric_token[i+1] in vocab) else '<OOV>'

            if token_i not in bigram_matrix:
                bigram_matrix[token_i] = {}

            if token_j not in bigram_matrix[token_i]:
                bigram_matrix[token_i][token_j] = 0

            bigram_matrix[token_i][token_j] += 1

    return bigram_matrix


def trigram_matrix(vocab, lyric_tokens):
    # format: trigram_matrix[(first,second)][third]
    trigram_matrix = {}

    for lyric_token in lyric_tokens:
        for i in range(len(lyric_token) - 2):

            token_i = lyric_token[i] if (lyric_token[i] in vocab) else '<OOV>'
            token_j = lyric_token[i + 1] if (lyric_token[i+1] in vocab) else '<OOV>'
            token_k = lyric_token[i + 2] if (lyric_token[i+2] in vocab) else '<OOV>'

            if (token_i, token_j) not in trigram_matrix:
                trigram_matrix[(token_i, token_j)] = {}

            if token_k not in trigram_matrix[(token_i, token_j)]:
                trigram_matrix[(token_i, token_j)][token_k] = 0

            trigram_matrix[(token_i, token_j)][token_k] += 1

    return trigram_matrix


def word_possibility(vocab, w3, w2, bigram, w1 = None, trigram = None):

    # p(w3 | w1, w2) = trigramCounts[(w1, w2)][w3]+1/bigramCounts[(w1, w2)] + VOCAB_SIZE
    # p(w2 | w1) = bigramCounts[w2][w1]/bigramCounts[w1]
    vocab_size = 0

    pool_size = 0

    for i in vocab:
        vocab_size += vocab[i]

    if w2 not in vocab:
        w2 = '<OOV>'

    findicator = False
    for i in bigram[w2]:
        pool_size += bigram[w2][i]

    if pool_size > 0:
        findicator = True

    # Unigram Model
    if not findicator:
        if w3 in vocab:
            return vocab[w3] / vocab_size
        else:
            return 0

    # Bigram model
    if w1 is None:
        if w3 in bigram[w2]:
            result = bigram[w2][w3]+1
        else:
            result = 1

        result = result/(vocab[w2]+len(vocab))

        return result
    # Trigram model
    else:

        if w1 not in vocab:
            w1 = '<OOV>'

        # p(w3 | w1, w2) = trigramCounts[(w1, w2)][w3] + 1 /bigramCounts[(w1,w2)] + vocab_size
        if w3 in bigram[w2]:
            result = bigram[w2][w3]+1
        else:
            result = 1
        result = result/(vocab[w2]+len(vocab))

        if w3 in trigram[(w1, w2)]:
            temp = trigram[(w1, w2)][w3] + 1
        else:
            temp = 1

        temp = temp/(bigram[w1][w2]+len(vocab))

        return (result+temp)/2


def getConllTags(filename):
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent

def getFeaturesForTokens(tokens, wordToIndex):
    # input: tokens: a list of tokens,
    # wordToIndex: dict mapping 'word' to an index in the feature list.
    # output: list of lists (or np.array) of k feature values for the given target

    num_words = len(tokens)
    featuresPerTarget = list()  # holds arrays of feature per word
    for targetI in range(num_words):
        # <FILL IN>
        #tokens[targetI] has the word and wordToIndex[targetI] has the dict

        numVow = 0
        numCon = 0
        for charI in tokens[targetI]:
            if charI.isalpha():
                if charI in ('a','e','i','o','u','A','E','I','O','U'):
                    numVow+= 1
                else:
                    numCon+= 1

        x = wordToIndex.get(tokens[targetI])
        n = len(wordToIndex)
        prevR = np.zeros(n)
        targetR = np.zeros(n)
        nextR = np.zeros(n)

        if targetI > 0:
            ix = wordToIndex.get(tokens[targetI-1])
            if ix != None:
                prevR[ix] = 1

        if x != None:
            targetR[x] = 1

        if targetI < num_words-1:
            ix = wordToIndex.get(tokens[targetI + 1])
            if ix != None:
                nextR[ix] = 1

        ret = np.array([numVow, numCon])
        ret = np.concatenate([ret, prevR, targetR, nextR])

        featuresPerTarget.append(ret)
        pass
        # featuresPerTarget[targetI] = ...
    return featuresPerTarget  # a (num_words x k) matrix


def trainAdjectiveClassifier(features, adjs):
    # inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    # output: model -- a trained sklearn.linear_model.LogisticRegression object

    model = None
    # <FILL IN>

    x_trainsub, x_dev, y_trainsub, y_dev = train_test_split(features, adjs, test_size=0.2, random_state=42)
    candidate = 0
    for i in range(1, 7):
        temp_model = LogisticRegression(penalty = "l1", solver='liblinear', C=10^i)
        temp_model.fit(x_trainsub, y_trainsub)
        y_check = temp_model.predict(x_dev)

        size = len(y_check)
        accuracy = np.sum([1 if (y_check[j] == y_dev[j]) else 0 for j in range(size)])/size
        if(accuracy > candidate):
            model = temp_model
            candidate = accuracy

    return model


def title_extract_feature(titles, wordToIndex):
    result = []
    for title in titles:

        for i in range(len(title)):
            title[i] = title[i].lower()

        result.append(getFeaturesForTokens(title, wordToIndex))

    return result


# Step 3.3:
def find_title_adjective(model, features, title_tokens, ids):
    result = {}

    #flatter = [j for i in features for j in i]

    adjective = np.array([])

    for tokens in features:
        adj = np.array(model.predict(tokens))
        adjective = np.concatenate([adjective, adj])

    print(len(adjective))

    i = 0
    j = 0
    for title in title_tokens:
        for token in title:
            k = 0
            if adjective[j] == 1:
                if title[k] not in result:
                    result[title[k]] = [ids[i]]
                else:
                    result[title[k]].append(ids[i])
            j += 1
            k += 1
        i += 1

    pop_list = []

    for adj in result:
        if len(result[adj]) < 11:
            pop_list.append(adj)

    for adj in pop_list:
        result.pop(adj)

    return result


# Step 3.4:
def build_language_model(adjective_list, ids, lyric_tokens):
    model = []

    input_tokens = list()

    if adjective_list is None:
        return None

    for i in adjective_list:
        temp = lyric_tokens[np.where(ids == i)]
        input_tokens.append(temp[0])

    vocabs = word_vocabulary(input_tokens)
    bigram = bigram_matrix(vocabs, input_tokens)
    trigram = trigram_matrix(vocabs, input_tokens)

    model.append(vocabs)
    model.append(bigram)
    model.append(trigram)

    return model


# Step 3.5:
def generate_lyrics(model):
    lyrics = list()

    if model is None:
        return lyrics

    vocabs = model[0]
    bigram = model[1]
    trigram = model[2]

    candidate = []

    for key in bigram['<s>']:
        candidate.append(key)
    if '<OOV>' in candidate:
        candidate.remove('<OOV>')

    if len(candidate) == 0:
        return lyrics

    probability = []
    for w3 in candidate:
        probability.append(word_possibility(vocabs, w3, '<s>', bigram))

    denom = sum(probability)
    for i in range(len(probability)):
        probability[i] = probability[i]/denom

    next_word = candidate[np.random.choice(len(candidate), 1, p=probability)[0]]
    lyrics.append(next_word)

    probability = []

    candidate = []

    for key in trigram[('<s>', next_word)]:
        candidate.append(key)

    if '<OOV>' in candidate:
        candidate.remove('<OOV>')

    if len(candidate) == 0:
        return lyrics

    for w3 in candidate:
        probability.append(word_possibility(vocabs, w3, next_word, bigram, '<s>', trigram))

    denom = sum(probability)
    for i in range(len(probability)):
        probability[i] = probability[i] / denom

    next_word = candidate[np.random.choice(len(candidate), 1, p=probability)[0]]
    lyrics.append(next_word)

    while len(lyrics) < 33:
        candidate = []

        for key in trigram[(lyrics[len(lyrics)-2], lyrics[len(lyrics)-1])]:
            candidate.append(key)

        if '<OOV>' in candidate:
            candidate.remove('<OOV>')

        if len(candidate) == 0:
            return lyrics

        probability = []
        for w3 in candidate:
            probability.append(word_possibility(vocabs, w3, lyrics[len(lyrics)-1], bigram, lyrics[len(lyrics)-2], trigram))

        denom = sum(probability)
        for i in range(len(probability)):
            probability[i] = probability[i] / denom

        next_word = candidate[np.random.choice(len(candidate), 1, p=probability)[0]]
        if next_word == '</s>':
            return lyrics
        lyrics.append(next_word)

    return lyrics



#Main starts here
if __name__ == '__main__':
    datas = readMemory()
    print("memory finished reading")
    tokens = tokenize_stuff(datas[1], datas[3])
    print("tokenized lyric and title")
    title_token = tokens[0]
    lyric_token = tokens[1]
    ids = datas[4]

    #Stage 1 Checkpoint:
    print('Stage 1 Checkpoint: tokenized song title and lyric')
    index = np.where(ids == 'abba-burning_my_bridges')
    print('abba-burning_my_bridges: ')
    print(title_token[index])
    print(lyric_token[index])
    index = np.where(ids == 'beach_boys-do_you_remember?')
    print('beach_boys-do_you_remember?: ')
    print(title_token[index])
    print(lyric_token[index])
    index = np.where(ids == 'avril_lavigne-5,_4,_3,_2,_1_(countdown)')
    print('avril_lavigne-5,_4,_3,_2,_1_(countdown): ')
    print(title_token[index])
    print(lyric_token[index])
    index = np.where(ids == 'michael_buble-l-o-v-e')
    print('michael_buble-l-o-v-e: ')
    print(title_token[index])
    print(lyric_token[index])

    #Stage 2 Checkpoint:
    print('Stage 2 Checkpoint: matrixes')
    vocabs = word_vocabulary(lyric_token[:6000])
    print('vocab: ')
    print(len(vocabs))
    bigram = bigram_matrix(vocabs, lyric_token[:6000])
    trigram = trigram_matrix(vocabs, lyric_token[:6000])

    print('word probability synatx: P(next_word | previous_2_words)')
    print('p(you | I, love):')
    print(word_possibility(vocabs, 'you', 'love', bigram, 'I', trigram))
    print('p(special | (midnight, )):')
    print(word_possibility(vocabs, 'special', 'midnight', bigram))
    print('p(special | (very,)):')
    print(word_possibility(vocabs, 'special', 'very', bigram))
    print('p(special | (something, very)):')
    print(word_possibility(vocabs, 'special', 'very', bigram, 'something', trigram))
    print('p(funny | (something, very)):')
    print(word_possibility(vocabs, 'funny', 'very', bigram, 'something', trigram))

    taggedSents = getConllTags('daily547.conll')

    wordToIndex = set()  # maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            wordToIndex |= set([w.lower() for w in words])
    print("  [Read ", len(taggedSents), " Sentences]")
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}
    sentXs = []
    sentYs = []
    print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex))
            sentYs.append([1 if t == 'A' else 0 for t in tags])

    X = [j for i in sentXs for j in i]
    y = [j for i in sentYs for j in i]
    try:
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        sys.exit(1)
    print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    # Train the model.
    print("  [Training the model]")
    adjective_model = trainAdjectiveClassifier(X_train, y_train)
    print("  [Done]")

    features = title_extract_feature(title_token, wordToIndex)

    adjective_dict = find_title_adjective(adjective_model, features, title_token, ids)
    print('artist-songs associated with good')
    print(adjective_dict.get('good'))
    print('artist-songs associated with happy')
    print(adjective_dict.get('happy'))
    print('artist-songs associated with afraid')
    print(adjective_dict.get('afraid'))
    print('artist-songs associated with red')
    print(adjective_dict.get('red'))

    print('generating lyrics for keyword with large data:')
    good_model = build_language_model(adjective_dict.get('good'), ids, lyric_token)
    print('good 1:')
    print(generate_lyrics(good_model))
    print('good 2:')
    print(generate_lyrics(good_model))
    print('good 3:')
    print(generate_lyrics(good_model))

    happy_model = build_language_model(adjective_dict.get('happy'), ids, lyric_token)
    print('happy 1:')
    print(generate_lyrics(happy_model))
    print('happy 2:')
    print(generate_lyrics(happy_model))
    print('happy 3:')
    print(generate_lyrics(happy_model))

    print('generating lyrics with keyword with few data:')
    afraid_model = build_language_model(adjective_dict.get('afraid'), ids, lyric_token)
    print('afraid 1:')
    print(generate_lyrics(afraid_model))
    print('afraid 2:')
    print(generate_lyrics(afraid_model))
    print('afraid 3:')
    print(generate_lyrics(afraid_model))

    print('generating lyrics that doens\'t have any data:')
    red_model = build_language_model(adjective_dict.get('red'), ids, lyric_token)
    print('red:')
    print(generate_lyrics(red_model))

    blue_model = build_language_model(adjective_dict.get('blue'), ids, lyric_token)
    print('blue:')
    print(generate_lyrics(blue_model))