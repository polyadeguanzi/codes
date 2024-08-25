import networkx as nx
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from  graph_zoo.python.DeepWalk import DeepWalk
from  graph_zoo.python.utils_deepwalk import generate_graph
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt


def train(sentences, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = sentences#生成的截断随机游走序列
        kwargs["min_count"] = kwargs.get("min_count", 0)#忽略所有word出现频率小于min_count
        kwargs["vector_size"] = embed_size#embedding维度
        kwargs["sg"] = 1  # 使用skip gram模式
        kwargs["hs"] = 1  # 使用 Hierarchical Softmax
        kwargs["workers"] = workers#使用线程数
        kwargs["window"] = window_size#上下文窗口宽度
        kwargs["epochs"] = iter#语料库上的迭代次数

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)#训练word2vec模型
        print("Learning embedding vectors done!")

        w2v_model = model
        return w2v_model    #返回训练好的word2vec模型

def get_embeddings(w2v_model,graph):#返回词嵌入embedding向量
        if w2v_model is None:
            print("model not train")
            return {}

        _embeddings = {}
        for word in graph.nodes():
            _embeddings[word] = w2v_model.wv[word]

        return _embeddings

def test_DeepWalk():
                    
    G=generate_graph(ver_num=10)     #ver_num图顶点数，图顶点随机连接，生成无监督训练数据

    sentences = DeepWalk(G, walk_length=3, num_walks=2, workers=1).get_randomwalk()#返回随机游走序列
    w2v_model = train(sentences,window_size=3, iter=1)#训练并返回返回word2vec模型
    embeddings = get_embeddings(w2v_model,G)#将图节点导入word2vec模型得到embedding向量



    for key, value in embeddings.items():
        assert ~np.isnan(value).any(), f"emmbeddings中键 '{key}' 对应的值包含NaN。"

    for key, value in embeddings.items():
        assert ~np.isinf(value).any(), f"emmbeddings中键 '{key}' 对应的值包含Inf。"
    print('测试完成')
if __name__ == "__main__":
    test_DeepWalk()