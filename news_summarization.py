'''
输入：一个list，里面每个item都是一个list，其中有三句新闻。
输出：三个句子

使用k中心点
'''
import pandas as pd
import numpy as np
import random
import jieba


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C


def readabstract(code: int):
    '''
    code, title, abstract
    abstract 使用 ###### 分割
    '''
    #code = str(code)
    path = 'd:/code/collection/data/news_abstract/{}.csv'
    df = pd.read_csv(path.format(code), dtype={'code': str, 'abstract': str})
    abt_list = list(df['abstract'])
    return abt_list


def prepare(code, topn=None):
    '''
    0. 准备数据
    '''
    # ========读取新闻
    if topn is not None:
        news_abstract.get_abstract_batch(code, topn)
    abt_list = readabstract(code)
    # print(abt_list)
    # ========预处理
    sents = []
    for abt in abt_list:
        sents += str(abt).split('#####')
    # 这里不分词，所以计算相似度的时候使用的是每次字符

    words = []
    for s in sents:
        words.append(list(jieba.cut(s)))
    return sents, words
    '''
    return sents
    '''


def present():
    '''
    1. sentence present
    '''
    pass


def similarity(sents):
    '''
    2. similarity  
    '''
    sim_mat = np.zeros([len(sents), len(sents)])
    for i, sent1 in enumerate(sents):
        for j, sent2 in enumerate(sents):
            if i == j:
                continue
            sim_mat[i][j] = len(set(sent1) & set(sent2)) / \
                len(set(sent1) | set(sent2))

    return sim_mat


def get_news_summarization(code, atopn=3, stopn=2):

    sents, words = prepare(code)  # , atopn)
    sim_mat = similarity(words)
    # print(sents)
    # print(sim_mat)
    Inds, C = kMedoids(sim_mat, stopn)
    # print(Inds)
    print([sents[i] for i in Inds])
    return [sents[i] for i in Inds]


if __name__ == '__main__':
    get_news_summarization(600285, stopn=2)
