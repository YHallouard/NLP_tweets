from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import string
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import unidecode

nltk.download('stopwords')


def clean_text_old(text):
    text = text.translate(string.punctuation)

    text = text.lower().split()

    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"#", "", text)

    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


def clean_text(to_remove_caracters: [str], stop_words: [str]) -> str:
    def f(text: str):
        text = text.lower()
        text = re.sub(r"([a-z]|[0-9]|_|\.)*@([a-z]|[0-9]|_|\.)*com", "", text)
        text = " ".join(text.split('('))
        text = " ".join(text.split(')'))
        text = re.sub(r"(https?://[^\s]+)", "", text)
        text = re.sub(r"(~|c|d|x){1}(:){0,1}(([a-z]|[0-9]|_|-)*(\\|/){1})+([a-z]|[0-9]|_|-)*", "", text)
        text = re.sub(r"(\d{2}(\.|/|-|\||:){1}\d{2}(\.|/|-|\||:){1}\d{2,4})", "", text)
        text = unidecode.unidecode(text)

        for old_char in to_remove_caracters:
            text = " ".join(text.split(old_char))

        text = "a".join(text.split('@'))
        text = re.sub(r"(\d)+", " ", text)
        text = re.sub(r"(((^| )win( |$))|((^| )w( |$)))", " windows ", text)

        text = text.split(' ')
        text = [word for word in text if ((word not in stop_words) and (len(word) > 1))]

        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

        return text

    return f


def get_sequences(vocabulary_size: int, texts: pd.DataFrame, maxlen: int, tokenizer=None):
    if not tokenizer:
        vocabulary_size = vocabulary_size
        tokenizer = Tokenizer(num_words=vocabulary_size)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=maxlen)

    return data, tokenizer


def get_embedding_matrix(tokenizer, word2vec, max_nb_words: int, enbedding_dim: int) -> np.array:
    word_index = tokenizer.word_index
    nb_words = min(max_nb_words, len(word_index)) + 1

    embedding_matrix = np.zeros((nb_words, enbedding_dim))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)

    return embedding_matrix
