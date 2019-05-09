from pymystem3 import Mystem
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.corpus import stopwords
import re
import emoji
import numpy as np
from tqdm import tqdm_notebook as tqdm
import re
import os
import pandas as pd
from gensim.models import word2vec
from collections import Counter
import string
ascii_string = set("""!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~""")
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models import ldamodel
import gensim.corpora as corpora

rus_names = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTFo0BlN7l7xo5vcNwiIOGncVjrd5tRsHohOm9XtybupNMgijajQGynAbxYJjqUsx7AVynCR2fyxrXS/pub?gid=1788578038&single=true&output=csv')
rus_names = list(rus_names['Name'].str.lower())

if stopwords.words('russian'):
    from nltk.corpus import stopwords
    stops = stopwords.words('russian')
    if rus_names:
        stops.extend(['\n', '\t', '\r'] + rus_names)
    else:
        stops.extend(['\n', '\t', '\r'])
    stops = [e for e in stops if e not in ['хорошо', 'больше', 'не']]
else:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stops = stopwords.words('russian')
    if rus_names:
        stops.extend(['\n', '\t', '\r'] + rus_names)
    else:
        stops.extend(['\n', '\t', '\r'])
    stops = [e for e in stops if e not in ['хорошо', 'больше', 'не']]
    
m = Mystem()

def remove_emoji(text):
    """
    input: string with emoji
    output: list with preprocessed strings
    removing emojies from string
    """
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in list(emoji.UNICODE_EMOJI.keys())]
    clean_text = re.sub("|".join(emoji_list), "", text)
    return clean_text

def preprocess_text(texts, indexes, lemmatize = True, stopwords = True, russian_only = True, file_path = os.getcwd() + '/clean_text.csv'):
    """
    input: pd.Series with durty texts and pd.Series with their indexes
    output: csv file with preprocessed texts associated with their initial indexes
    lemmatizing, removing punctuation, special characters and stopwods 
    """
    i = 0
    header = True
    for text, index in tqdm(zip(texts, indexes)):
        if russian_only:
            ru_text_only = re.compile('[^а-яА-Я]')
            text = ru_text_only.sub(' ', text)
        else:
            translator = str.maketrans('', '', string.punctuation + '...')
            text = remove_emoji(text)
            text = text.translate(translator)
            text = ' '.join(re.sub(r'(http\S+)|(RT)|(\@\w+)|(\#\w+)|(\u2060)|(<.*?>)|(\d+)', ' ', text).split())
        if lemmatize:
            text = m.lemmatize(text)
        else: 
            text = text.split(' ')
        if stopwords:
            text = [w for w in text if w not in stops]
        text = ' '.join(text)
        text = re.sub(r'(\s+)', ' ', text).strip()
        text = re.sub('не ', 'не', text).strip()
        if header:
            with open(file_path,'w') as f: 
                f.write('clean_texts,index\n')
                f.write(f'{text},{index}\n') 
                header = False
        else:
            with open(file_path,'a') as f:
                f.write(f'{text},{index}\n') 
        i += 1
    return print(f'{i} texts written to {file_path}')
    
def train_word2vec(sentences, num_workers, num_features, min_word_count, context, iterations, file_path = os.getcwd() + '/emopok_w2v_model'):
    '''
    word2vec embeddings using gensim implementation of word2vec
    sentences: list of sentences
    save_model_to: string with path where model will be saved
    '''
    logging.basicConfig(filename = os.getcwd() + '/emopok_w2v.log', format = "%(levelname)s - %(asctime)s: %(message)s", datefmt = '%H:%M:%S', level = logging.INFO, mode='w')

    model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context,
                          iter=iterations)
    model.save(file_path)
    print("Model was successfully saved with a name ", file_path)
    return model


def textfeatures(text, index):
    """
    get text features of a string
    imput: dirty string
    output: pd.DataFrame with text, index, n_chars, n_commas, n_digits, n_exclaims, n_hashtags, 
    n_lowers, n_mentions, n_urls, n_words, n_nonasciis, n_uppers
    """
    text = remove_emoji(text)
    n_chars = len(text)
    n_commas = text.count(",")
    n_digits = sum(list(map(str.isdigit, text)))
    n_exclaims = text.count("!")
    n_hashtags = len(re.findall(r"(#\w+)", text))
    n_lowers = sum(list(map(str.islower, text)))
    n_mentions = len(re.findall(r"(@\w+)", text))
    n_urls = text.count("http")
    ru_text_only = re.compile('[^а-яА-Я]')
    ru_text = ru_text_only.sub(' ', text)
    n_words = len(ru_text.split(" "))
    n_nonasciis = len(text.encode()) - n_chars
    n_uppers = sum(list(map(str.isupper, text)))
    df = pd.DataFrame({'text': text, 'index': index, 'n_chars': n_chars, 'n_commas': n_commas, 'n_digits': n_digits, 'n_exclaims': n_exclaims, \
                  'n_hashtags': n_hashtags, 'n_lowers': n_lowers, 'n_mentions': n_mentions, 'n_urls': n_urls, \
                  'n_words': n_words, 'n_nonasciis': n_nonasciis, 'n_uppers': n_uppers}, index=[0])
    return df


def train_doc2vec(texts, embed_size, window_size, n_epochs, n_neg, save_model_to = os.getcwd() + '/emopok_d2v_model'):
    """
    doc2vec embeddings using gensim implementation of doc2vec
    for speed boost you can lower the n_epochs parameter
    """
    import logging  # setting up the loggings to monitor gensim
    logging.basicConfig(filename = os.getcwd() + '/emopok_d2v.log', format = "%(levelname)s - %(asctime)s: %(message)s", datefmt = '%H:%M:%S', level = logging.INFO, mode='w')

    documents = [TaggedDocument(doc, [i]) for i, doc in tqdm(enumerate(texts))]
    model = Doc2Vec(documents, vector_size=embed_size, window=window_size, 
                    epochs=n_epochs, min_count=2, negative=n_neg)

    model.save(save_model_to)
    print("Model was successfully saved with a name ", save_model_to)
    return save_model_to, model.docvecs.doctag_syn0
    
def get_d2v_embedding(texts, path_to_model):
    """
    get embedding for a sentence
    path_to_model: path to a trained gensim doc2vec model
    """
    model = Doc2Vec.load(path_to_model) 
    return model.infer_vector(texts)

def get_lda_model(sentences, num_topics, file_path = os.getcwd() + '/emopok_lda_model'):
    id2word = corpora.Dictionary(sentences)
    corpus = [id2word.doc2bow(text) for text in sentences]
    logging.basicConfig(filename = os.getcwd() + '/emopok_lda.log', format = "%(levelname)s - %(asctime)s: %(message)s", datefmt = '%H:%M:%S', level = logging.INFO)
    lda_model_emo = ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=1,
                                               alpha=0.001,
                                               per_word_topics=True,
                                               minimum_probability=0)
    lda_model_emo.save(file_path)
    print("Model was successfully saved with a name ", file_path)
    return corpus

def get_topics_for_docs(corpus, lda_model_emo, k, clean_texts):
    '''get most probable topics for corpus documents'''
    docs_train_ids = []
    topics_train = []
    for i in tqdm(list(range(len(corpus)))):
        seen_doc = corpus[i]
        vector = lda_model_emo[seen_doc]
        probs = []
        for t in list(range(k)):
            probs.append(vector[0][t][1])
        docs_train_ids.append(i)
        topics_train.append(probs.index(max(probs)))
    df_topics = pd.DataFrame({'clean_text': clean_texts, 'topic': topics_train})
    return df_topics

def get_emoji_sentences(text, emoji_list):
    """
    input: string with emoji
    output: list with lists of words
    """
    emoji_text = re.findall("|".join(emoji_list), text)
    return emoji_text

def search_for_kmeans(k_max, X):
    for n_cluster in range(2, k_max):
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

def search_for_kmeans(k_max, X):
    for n_cluster in range(2, k_max):
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
        
def train_kmeans(n_clusters, X, emojis_found, save_to = os.getcwd() + '/emopok_clusters.csv'):
    k_means = KMeans(n_clusters=n_clusters, random_state=1)
    k_means.fit(X)
    k_means_labels = k_means.labels_ # array with cluster of emojis (concat to df)


    k_means_labels_unique = np.unique(k_means_labels)
    k_means_cluster_centers = k_means.cluster_centers_
    emo_clusters = pd.DataFrame(k_means_labels, index=emojis_found, columns=['cluster_group']).reset_index()
    emo_clusters.to_csv(save_to, index = False)
    return emo_clusters
