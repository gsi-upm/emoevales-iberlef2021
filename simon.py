import pandas as pd
import string
from collections import Counter
from itertools import chain
from nltk.corpus import stopwords
from gsitk.features import simon
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV

def simon_pipeline():
    simon_pipe = Pipeline([
        ('lr', LogisticRegressionCV(cv=10, random_state=42, n_jobs=-1, solver='liblinear'))
    ])
    return simon_pipe

def generate_custom_lexicon(text):
    filter_words = set(stopwords.words('spanish')) | set(string.punctuation)

    counter = Counter(chain.from_iterable(text.str.split(' ').values))
    selection = sorted([(word, count) for word, count in counter.items()], key=lambda wc: wc[1], reverse=True)
    selection = [word for word, _ in selection if word not in filter_words]
    selection = [selection]
    return selection


def load_resources(all_texts):
    print('Generating custom lexicon')
    custom_lexicon = generate_custom_lexicon(all_texts)
    print('Done')

    # facebook fasttext embeddings
    print('Loading embeddings')
    embbeddings = KeyedVectors.load_word2vec_format(
        '/home/jovyan/work/projects/data/WordEmbeddings/eng/crawl-300d-2M.vec', binary=False)
    print('Done')

    return custom_lexicon, embbeddings

def main(text_train, text_dev, text_test, all_texts, dataset, n_lexicon_words=2000, percentile=50):
    custom_lexicon, embbeddings = load_resources(all_texts)

    _simon_model = simon.Simon(lexicon=custom_lexicon,
                               n_lexicon_words=n_lexicon_words,
                               embedding=embbeddings)
    simon_model = simon.simon_pipeline(simon_transformer=_simon_model, percentile=percentile)

    print('Training and predicting SIMON feats')
    X_simon_train = simon_model.fit_transform(pd.Series(text_train).str.split(' '), dataset['train']['emotion'])
    X_simon_dev = simon_model.transform(pd.Series(text_dev).str.split(' '))
    X_simon_test = simon_model.transform(pd.Series(text_test).str.split(' '))
    print('Done')

    print('Training classifier and predicting')
    simon_pipe = simon_pipeline()
    simon_pipe.fit(X_simon_train, dataset['train']['emotion'])
    simon_preds_dev = simon_pipe.predict(X_simon_dev)
    simon_preds_test = simon_pipe.predict(X_simon_test)
    print('Done')

    return simon_preds_dev, simon_preds_test
