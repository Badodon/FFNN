# coding: utf-8
import numpy as np
from gensim import matutils
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary
import sys
"""
reffered to 
  gensimソース    http://pydoc.net/Python/gensim/0.12.4/gensim.parsing.preprocessing/
  BOW作成         http://stmind.hatenablog.com/entry/2013/11/04/164608
@Badodon
"""

def load_data(fname):
    
    print 'input file name:', fname

    target = [] #ラベル
    source = [] #文書ベクトル

    #文書リストを作成
    document_list = []
    word_list = []
    for l in open(fname, 'r').readlines():
        sample = l.strip().split(' ',  1)
        label = sample[0]
        target.append([label]) #ラベル
        word_list = preprocess_string(sample[1]) #ストップワード除去, ステミング
        document_list.append(word_list) #文書ごとの単語リスト
    
    #辞書を作成
    #低頻度と高頻度のワードは除く
    dct = Dictionary(document_list)
    dct.filter_extremes(no_below=3, no_above=0.6)

    #文書のBOWでベクトル化
    for doc in document_list:
        tmp = dct.doc2bow(doc) # ex.[(4, 1), (23,1),..., (119,2)] 
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dct)).T[0])
        source.append(dense)

    dataset = {}
    dataset['target'] = np.array(target)    
    dataset['source'] = np.array(source)    

    return dataset #, max_len, width

