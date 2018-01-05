import pandas as pd
import numpy as np
import sys, os, io


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
    with open(path) as f:
        content = f.read()
    return content


def load_data(root, label_pos, class1, class2):
    all_docs = []
    for path, sub, filen in os.walk(root):
        for f in filen:
            if f == '.DS_Store':
                continue
            doc_path = path+'/'+f
            all_docs.append(doc_path)
    print len(all_docs)
    df = pd.DataFrame({'path': all_docs, 'doc_id': range(1, len(all_docs)+1)})
    df['author_code'] = df['path'].apply(lambda x: x.split('/')[-1])
    df['essay_content'] = df['path'].apply(read_file)
    df['label'] = df['author_code'].apply(lambda x: x.split('_')[label_pos])
    df['target'] = df['label'].map({class1: 1, class2: 0})
    # df['target'] = df['label'].map({'CHN':1, 'JPN':0})
    return df


def load_data_multi(root):
    all_docs = []
    for path, sub, filen in os.walk(root):
        for f in filen:
            if f == '.DS_Store':
                continue
            doc_path = path+'/'+f
            all_docs.append(doc_path)
    print len(all_docs)
    df = pd.DataFrame({'path': all_docs, 'doc_id': range(1, len(all_docs)+1)})
    df['author_code'] = df['path'].apply(lambda x: x.split('/')[-1])
    df['essay_content'] = df['path'].apply(read_file)
    df['label'] = df['author_code'].apply(lambda x: x.split('_')[1])
    return df
