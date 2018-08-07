import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from tabulate import tabulate

full_corpus = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
full_corpus.columns = ["lable", "body_text"]
full_corpus.head()


full_corpus['punct_removed'] = full_corpus['body_text'].str.replace(r'[^\w\s]+', '')

full_corpus['lower_case'] =  full_corpus['punct_removed'].str.lower()

full_corpus['stop_words_removed'] = full_corpus['lower_case'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))

full_corpus['tokinized_word'] = full_corpus['stop_words_removed'].apply(nltk.word_tokenize)

full_corpus['lemmatized_words'] = full_corpus['tokinized_word'].apply(lambda x: [WordNetLemmatizer().lemmatize(y) for y in x])

full_corpus['Bigrams'] = full_corpus['lemmatized_words'].apply(lambda x: list(ngrams(x, 2)))
bigram_group = full_corpus.groupby('lable').agg({'Bigrams': 'sum'})
unigram_group = full_corpus.groupby('lable').agg({'lemmatized_words': 'sum'})

unigram_spam = {}
for unigram in unigram_group.iat[1, 0]:
    if unigram not in unigram_spam:
        unigram_spam[unigram] = 1
    else:
        unigram_spam[unigram] += 1

unigram_ham = {}
for unigram in unigram_group.iat[0, 0]:
    if unigram not in unigram_ham:
        unigram_ham[unigram] = 1
    else:
        unigram_ham[unigram] += 1

bigram_spam = {}
for bigram in bigram_group.iat[1, 0]:
    if bigram not in bigram_spam:
        bigram_spam[bigram] = 1
    else:
        bigram_spam[bigram] += 1

bigram_ham = {}
for bigram in bigram_group.iat[0, 0]:
    if bigram not in bigram_ham:
        bigram_ham[bigram] = 1
    else:
        bigram_ham[bigram] += 1

message1 = "Sorry, ..use your brain dear"
message2 = "SIX chances to win CASH."

message1 = message1.lower()
message1 = re.sub(r'[^\w\s]', '', message1)
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(message1)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
message1 = " ".join(filtered_sentence)
word_tokens = word_tokenize(message1)

LemmatizedWords = []
for ms in word_tokens:
    LemmatizedWords.append(WordNetLemmatizer().lemmatize(ms))

bigrams = list(ngrams(LemmatizedWords, 2))


message2 = message2.lower()
message2 = re.sub(r'[^\w\s]', '', message2)
stop_words2 = set(stopwords.words('english'))
word_tokens2 = word_tokenize(message2)
filtered_sentence2 = [w for w in word_tokens2 if not w in stop_words2]
message2 = " ".join(filtered_sentence2)
word_tokens2 = word_tokenize(message2)

LemmatizedWords2 = []
for ms in word_tokens2:
    LemmatizedWords2.append(WordNetLemmatizer().lemmatize(ms))

bigrams2 = list(ngrams(LemmatizedWords2, 2))


def ham_bigram_probability(word1, word2):
    try:
        a = unigram_ham[word1]
    except KeyError:
        a = 0

    try:
        ab = bigram_ham[word1, word2]
    except KeyError:
        ab = 0

    V = len(unigram_ham)
    p = (ab + 1) / (a + V)
    return p


def spam_bigram_probability(word1, word2):
    try:
        a = unigram_spam[word1]
    except KeyError:
        a = 0

    try:
        ab = bigram_spam[word1, word2]
    except KeyError:
        ab = 0

    V = len(unigram_spam)
    p = (ab + 1) / (a + V)
    return p


ham_message1 = 1
for bigram in bigrams:
    ham_message1 = ham_message1 * ham_bigram_probability(bigram[0], bigram[1])

spam_message1 = 1
for bigram in bigrams:
    spam_message1 = spam_message1 * spam_bigram_probability(bigram[0], bigram[1])

ham_message2 = 1
for bigram in bigrams2:
    ham_message2 = ham_message2 * ham_bigram_probability(bigram[0], bigram[1])

spam_message2 = 1
for bigram in bigrams2:
    spam_message2 = spam_message2 * spam_bigram_probability(bigram[0], bigram[1])

print("Sorry, ..use your brain dear message ham:",ham_message1)
print("Sorry, ..use your brain dear message spam:",spam_message1)


if ham_message1>spam_message1:
    print("message1 is ham")
else:
    print("message1 is spam")

print("SIX chances to win CASH. message ham,:",ham_message2)
print("SIX chances to win CASH. message spam:",spam_message2)

if ham_message2>spam_message2:
    print("message2 is ham")
else:
    print("message2 is spam")
