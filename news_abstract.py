'''
这里只处理单条新闻

https://blog.csdn.net/mouday/article/details/89469583
http://blog.itpub.net/31562039/viewspace-2286669/
snownlp用的也是textrank
参考这个方法
'''
import networkx as nx
from typing import List
import jieba
import numpy as np
import os
import csv


def stopwords():
    stops = []
    with open('d:/code/collection/data/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stops.append(str(line.strip()))
    return stops


def readnews(fname):
    # 读取数据

    #
    with open('d:/code/collection/data/news/{}.txt'.format(fname), 'r', encoding='utf-8') as f:
        texts = [line.strip().replace('\xa0', '')
                 for line in f.readlines()[5:]]
    return texts


def preprocess(lines: List)->List:
    pass


def filter_news(text):
    """
    keep or not
    True: yes, keep it
    False: no, pass it
    """
    filterword = ['投资者提问', '交易提示']
    for w in filterword:
        if w in text:
            return False
    return True


def filter_content(text):
    """
    keep or not
    True: yes, keep it
    False: no, pass it
    """
    if len(text) < 20 and ('记者' in text or '编辑' in text):
        return False
    filterword = ['本文来自', '违规转载法律必究', '注意投资风险',
                  '不构成投资建议', '免责声明', '文章来源', '微信公众号']
    for w in filterword:
        if w in text:
            return False
    return True


def prepare(fname):
    '''
    0. 准备数据
    '''
    # ========读取新闻
    lines = readnews(fname)
    lines = '。'.join(lines).split('。')

    # ========预处理
    # 每行新闻
    sents = [l.replace(' ', '').strip() for l in lines]
    sents = [l for l in sents if len(l) > 10]
    sents = [l for l in sents if filter_content(l)]

    # 每行新闻的每个词
    words = [jieba.cut(l) for l in sents]

    # 去除停用词
    sw = stopwords()
    words = [[w for w in word if w not in sw] for word in words]

    index = [i for i, ws in enumerate(words) if len(ws) > 1]
    words = [words[i] for i in index]
    sents = [sents[i] for i in index]
    return sents, words


def present():
    '''
    1. sentence present
    '''
    pass


def similarity(words):
    '''
    2. similarity  
    '''
    sim_mat = np.zeros([len(words), len(words)])
    for i, sent1 in enumerate(words):
        for j, sent2 in enumerate(words):
            if i == j:
                continue
            sim_mat[i][j] = len(
                [w1 == w2 for w1 in sent1 for w2 in sent2])/len(set(sent1+sent2))

    return sim_mat


def ranking(sim_mat, topn=3):
    '''
    3. text rank
    '''
    # 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
    nx_graph = nx.from_numpy_array(sim_mat)

    # 得到所有句子的textrank值
    scores = nx.pagerank(nx_graph)

    # 根据textrank值对未处理的句子进行排序
    ranks = sorted(
        [(s, i) for i, s in enumerate(scores)], key=lambda x: x[0], reverse=True)

    # 取出得分最高的前10个句子作为摘要
    rank_index = [r[-1] for r in ranks[:topn]]

    return rank_index


def get_abstract(fn, topn=3):
    sents, words = prepare(fn)
    if len(sents) < topn+1:
        return None
    # print(sents)
    sim = similarity(words)
    ranks = ranking(sim, topn)
    # print(ranks)
    return [sents[inx] for inx in ranks]


def get_abstract_batch(code: int, topn=3):
    dirp = 'd:/code/collection/data/news/'
    code = str(code)
    data = []
    head = ['code', 'title', 'abstract']
    for fn in os.listdir(dirp):
        if code in fn:
            if filter_news(fn) is False:
                continue
            abt = get_abstract(fn.replace('.txt', ''), topn)
            if abt is None:
                continue
            # print(abt)
            abstract = '#####'.join(abt)
            #code, title = fn.split('_')
            data.append(fn.split('_') + [abstract])
    # print(data)
    with open('d:/code/collection/data/news_abstract/{}.csv'.format(code), 'w', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(data)


def test():
    fn = '601567_11月18日上市公司晚间公告速递'
    sents, words = prepare(fn)
    sim = similarity(words)
    ranks = ranking(sim)

    for inx in ranks:
        print(sents[inx])


if __name__ == '__main__':
    get_abstract_batch(600285, 3)
