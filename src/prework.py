import pandas as pd
import numpy as np
import sys, os, io
import re

from random import shuffle
from nltk.stem.snowball import PorterStemmer
import spacy
from nltk import ngrams
import matplotlib.pyplot as plt
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess, simple_tokenize
from sklearn.cross_validation import KFold
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score

import nltk



def split_essay(inpath, outpath, spc):
    all_files = []
    for path, sub, filen in os.walk(inpath):
        for f in filen:
            if f == '.DS_Store':
                continue
            all_files.append(f)
    for filename in all_files:
        with open(inpath+'/'+filename) as f:
            doc = f.read()
        sents = [str(s) for s in spc(doc.decode('utf-8')).sents]
        doc_length = len(sents)
        ck1 = ' '.join(sents[:doc_length/3])
        with open(outpath+'/'+'1_'+filename, 'w') as f:
            f.write(ck1)
        ck2 = ' '.join(sents[doc_length/3:2*(doc_length/3)])
        with open(outpath+'/'+'2_'+filename, 'w') as f:
            f.write(ck2)
        ck3 = ' '.join(sents[2*(doc_length/3):])
        with open(outpath+'/'+'3_'+filename, 'w') as f:
            f.write(ck3)
        

def read_file(path):
    '''
    Inputs:
    path: A file's absolute path
    
    Returns:
    The text content of the given file
    '''
    with open(path) as f:
        content = f.read()
    return content


def load_data(root, label_pos, class1, class2):
    '''
    To load individual text files into a pandas dataframe with only 2 classes
    Inputs:
    root: The absolute path of the root directory of ALL text files
    label_pos: This function is able to extract a document's class label from the text filename if given
               e.g. filename = 'W_CHN_PTJ0_021_A2_0.txt', then label_pos = 1 (label is 'CHN', position
               counting starts from 0)
    class1: label of class 1, e.g. 'CHN'
    class2: label of class 2, e.g. 'ENS'
    
    Returns:
    A pandas dataframe containing a bunch of columns describing a text file's information. The most important column here are:
    'label' and 'essay_content'
    
    E.g. load_data('data/all_txt_files', s, 'CHN', 'ENS')
    
    '''
    all_docs = []
    for path, sub, filen in os.walk(root):
        for f in filen:
            if f == '.DS_Store':
                continue
            doc_path = path+'/'+f
            all_docs.append(doc_path)
    print len(all_docs)
    df = pd.DataFrame({'path':all_docs, 'doc_id':range(1,len(all_docs)+1)})
    df['author_code'] = df['path'].apply(lambda x: x.split('/')[-1])
    df['essay_content'] = df['path'].apply(read_file)
    df['label'] = df['author_code'].apply(lambda x: x.split('_')[label_pos])
    df['target'] = df['label'].map({class1:1, class2:0})
    #df['target'] = df['label'].map({'CHN':1, 'JPN':0})
    return df


def load_data_multi(root):
    '''
    Serve the same purpose as the "load_data" function above, but for multiple
    class labels, as oppose to only binary classes. This funtion assumes that
    the label position is 1, so less flexible
    
    Inputs:
    root: The absolute path of the root directory of ALL text files
    
    Returns:
    A pandas dataframe containing a bunch of columns describing a text file's information. The most important column here are:
    'label' and 'essay_content'
    
    E.g. load_data_multi('data/all_txt_files')
    '''
    all_docs = []
    for path, sub, filen in os.walk(root):
        for f in filen:
            if f == '.DS_Store':
                continue
            doc_path = path+'/'+f
            all_docs.append(doc_path)
    #print len(all_docs)
    df = pd.DataFrame({'path':all_docs, 'doc_id':range(1,len(all_docs)+1)})
    df['author_code'] = df['path'].apply(lambda x: x.split('/')[-1])
    df['essay_content'] = df['path'].apply(read_file)
    df['label'] = df['author_code'].apply(lambda x: x.split('_')[1])
    return df


def stemmed_words(doc):
    '''
    This function is normally called as the sklearn vectorizer's analyzer
    so that tokenize can be performed when a vectorizer is initialized
    
    Inputs:
    doc: the untokenized text body of a document
    
    Returns:
    The tokenized version of the document
    
    E.g. vectorizer = CountVectorizer(lowercase=True, analyzer=stemmed_words)
    '''
    stemmer = PorterStemmer()
    analyzer = TfidfVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))


def ngram(string, n, sep):
    lst = string.split(sep)
    result = []
    for i, w in enumerate(lst):
        if len(lst[i:i+n])==n:
            result.append(' '.join(lst[i:i+n]))
    return result

'''
# Deprecated
def label_sentences(df, col, shuffle_sent=False):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = datapoint[col]
        labeled_sentences.append(LabeledSentence(words=tokenized_words, tags=['ESSAY_{}'.format(datapoint['doc_id'])]))
    if shuffle_sent:
        shuffle(labeled_sentences)
    return labeled_sentences


def train_doc2vec_model(labeled_sentences, window, size):
    model = Doc2Vec(min_count=1, window=window, size=size, negative=20)
    model.build_vocab(labeled_sentences)
    # Manually loop through epochs is old-fashion, adapt to new approach
    #for epoch in range(10):
        #model.train(utils.shuffle(labeled_sentences), total_examples=model.corpus_count, epochs=1)
        #model.alpha -= 0.002 
        #model.min_alpha = model.alpha
    model.train(labeled_sentences, total_examples=model.corpus_count, epochs=10)
    return model


def doc2vec_essays(df, d2v_model, add_col):
    y = []
    doc_vec = []
    #for i in range(1,df.shape[0]+1):
    for i in df['doc_id'].values:
        label = 'ESSAY_{}'.format(i)
        doc_vec.append(d2v_model.docvecs[label])
    df[add_col] = doc_vec
    return df
'''

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def tag_docs(docs, col, literal=True):
    '''
    Inputs:
    doc: pandas dataframe containing text columns
    col: column name of the text to train the doc2vec model
    literal: True when the text column contains the original human language text
             False when the text column is the converted text with syntactic parsing tags
             default: True
    
    Returns:
    a TaggedDocument object, each element contains tokenized text content and document tag, the tag here     contains 'label' and 'doc_id' columns
    
    E.g. 
    train_tagged = tag_docs(df_train, 'essay_content')
    or
    train_tagged = tag_docs(df_train, 'DT_pos', literal=False)

    '''
    if literal:
        tagged = docs.apply(lambda r: TaggedDocument(words=simple_preprocess(r[col]), tags=[r.label]), axis=1)
    else:
        tagged = docs.apply(lambda r: TaggedDocument(words=r[col].split(), tags=[r.label]), axis=1)
    return tagged


def train_doc2vec_model(tagged_docs, window, size):
    '''
    Inputs:
    tagged_docs: a TaggedDocument object
    window: number neighbor words(single direction) the model will be checking for the current word
    size: number of nodes in the NN's hidden layer
    
    Returns:
    The trained doc2vec model
    
    E.g.
    model = train_doc2vec_model_new(train_tagged, 5, 100)
    '''
    sents = tagged_docs.values
    # train doc2vec model to get vector representation of documents
    doc2vec_model = Doc2Vec(sents, size=size, window=window, iter=20, dm=1)
    return doc2vec_model


def vec_for_learning(doc2vec_model, tagged_docs):
    '''
    Inputs:
    doc2vec_model: The trained doc2vec model
    tagged_docs: a TaggedDocument object
    
    Returns:
    targets: the class label of a data point
    regressors: the document vector of the data point
    
    
    E.g.
    y_train, X_train = vec_for_learning(model, train_docs)
    '''
    sents = tagged_docs.values
    targets, regressors = zip(\
            *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


def plot_pattern(df, col, statistic):
    '''
    This function plot the histogram for a given column of a given pandas dataframe.
    Inputs:
    df: the dataframe to be plotted
    col: the column to be plotted. The column should contain a numerical list
    e.g. [1, 9, 3, 16, 8, 8, 1, 3, 7, 1, 11, 2, 4]
    statistic: a function to calculate certain empirial moment for a given list. e.g. np.mean()
    
    Returns:
    None
    
    E.g. plot_pattern_country(df_1, 'DT_ROOT_idx', np.median)
    '''
    for c in df['label'].unique():
        df_c = df[df['label']==c]
        df_c['col_statistic'] = df_c[col].apply(lambda x: statistic(x))
        plt.hist(df_c['col_statistic'], label=c)
        plt.legend()
    return None


def plot_pattern_country(df, col, xlower, xupper, ylower, yupper, title, statistic=np.mean):
    '''
    A refined version of the "plot_pattern() function above"
    df: the dataframe to be plotted
    col: the column to be plotted. The column should contain a numerical list
    e.g. [1, 9, 3, 16, 8, 8, 1, 3, 7, 1, 11, 2, 4]
    statistic: a function to calculate certain empirial moment for a given list. e.g. np.mean()
               default: np.mean
    xlower: plt.xlim's lower end
    xupper: plt.xlim's upper end
    ylower: plt.ylim's lower end
    yupper: plt.ylim's upper end
    title: title of the whole figure
    
    Returns:
    None
    
    E.g. plot_pattern_country(df_1, 'DT_ROOT_idx', 0, 20, 0, 0.5, 'ROOT Word Position per Country', np.median)
    '''
    countries = list(df.groupby('label').size().index)
    fig, axs = plt.subplots(3,3)
    for c, ax in zip(countries, axs.flatten()):
        data = df[df['label']==c][col].apply(lambda x: statistic(x))
        ax.hist(data, bins=10, normed=1)
        ax.set_xlim(xlower, xupper)
        ax.set_ylim(ylower, yupper)
        ax.set_title(c)
    fig.suptitle(title, fontsize=20)
    fig.set_size_inches(18.5, 10.5, forward=True)
    return None


def k_fold_doc2vec_clf(cv_data, text_col, word_window, hidden_nodes, clf, literal=True):
    '''
    Inputs:
    cv_data: normally the training data set after train_test_split()
    text_col: the column to be taken in as the text body to train the vector representations
    word_window: number of neibor words the doc2vec model will look into (both forward and backward)
    hidden_nodes: number of nodes in the doc2vec neurual network's hidden layer
    clf: the classifier to be valicated
    literal: True when the text column contains the original human language text
             False when the text column is the converted text with syntactic parsing tags
             default: True
    
    Returns:
    the mean score of the kfold cross validations
    
    E.g. k_fold_doc2vec_clf(train_data, col, 1, 100, clf, literal=False)
    '''
    scores = []
    kf = KFold(n=cv_data.shape[0], n_folds=5)
    X = cv_data
    for train, test in kf:
        cv_train_data = X.iloc[train]
        cv_test_data = X.iloc[test]
    
        train_docs = tag_docs(cv_train_data, text_col, literal)
        test_docs = tag_docs(cv_test_data, text_col, literal)
        model = train_doc2vec_model(train_docs, word_window, hidden_nodes)
    
        y_train, X_train = vec_for_learning(model, train_docs)
        y_test, X_test = vec_for_learning(model, test_docs)
    
        #logreg = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return scores


def tag_sent_gram(s, n):
    result = []
    n_grams = ngrams(s.split(' '), n)
    for grams in n_grams:
        result.append('_'.join(grams))
    return result

def loop_body(body, n):
    result = []
    for sent in body:
        result.append(tag_sent_gram(sent, n))
    return ' '.join(sum(result, []))


def split_mix_essays(inpath, outpath, topic):
    all_files = []
    for path, sub, filen in os.walk(inpath):
        for f in filen:
            if f == '.DS_Store':
                continue
            all_files.append(f)
    for filename in all_files:
        with open(inpath+'/'+filename) as f:
            doc = f.read()
        essays = re.split(topic+"[0-9]+\.[0-9].*\r\n", doc)[:]
        #essays = re.split("[0-9][0-9][0-9]\r\n\r\n", doc)[1:]
        print len(essays)
        for i, e in enumerate(essays):
            with open(outpath+'/'+topic.replace(' ','_')+str(i+1)+'_3_'+'ENS.txt', 'w') as f:
                f.write(e)
    return None


def print_confusion_matrix(y_test, y_pred):
    '''
    Inputs:
    y_test: the actual class labels of each data point, type: numpy.ndarray
    y_pred: the predicted class labels of each data point, type: numpy.ndarray
    
    Returns: 
    The confusion matrix as a pandas dataframe
    E.g. print_confusion_matrix(y_test.values, y_pred, countries)
    '''
    y_test = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    miss_countries = list(set(y_test.unique())-set(y_pred.unique()))
    countries = df_confusion.columns[:-1]
    if len(miss_countries)>0:
        for c in miss_countries:
            df_confusion[c] = 0
        cols = list(countries) + miss_countries + ['All']
        df_confusion = df_confusion[cols]
    countries = list(df_confusion.index[:-1])
    recalls = []
    tp_all = 0
    for i, c in enumerate(countries):
        tp_all += df_confusion[c][i]
        recalls.append(df_confusion[c][i]*1./df_confusion['All'][i])
    # Calculate overall accuracy
    #recalls.append(str(tp_all*1./df_confusion['All']['All'])+'(Overall Accuracy)')
    recalls.append(tp_all*1./df_confusion['All']['All'])
    df_confusion['Recall'] = recalls
    return df_confusion