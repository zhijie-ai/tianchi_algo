#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/12/1 16:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import jieba
import jieba.analyse
from gensim.models import word2vec

with open('../data/t.txt',encoding='utf8') as f:
    document = f.read()
    document_cut = jieba.cut(document)

    result = ' '.join(document_cut)
    with open('../data/t2.txt', 'w',encoding='utf8') as f2:
        f2.write(result)


raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]
# 切分词汇
sentences= [s.split() for s in raw_sentences]
print(sentences)

model = word2vec.Word2Vec(sentences, min_count=1)# sentences是一个二维数组
print(model.wv.vocab)

