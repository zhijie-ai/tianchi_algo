#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/12/1 14:23                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
import jieba

text = """我是一条天狗呀！
我把月来吞了，
我把日来吞了，
我把一切的星球来吞了，
我把全宇宙来吞了。
我便是我了！"""
sentences = text.split()
print(sentences)
sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
print(sent_words)
document = [" ".join(sent0) for sent0 in sent_words]
print(document)

tfidf_model = TfidfVectorizer().fit(document)
print(tfidf_model.vocabulary_)
# {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}
sparse_result = tfidf_model.transform(document)
print(sparse_result)

