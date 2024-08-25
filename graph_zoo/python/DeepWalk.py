import itertools
import random
from joblib import Parallel, delayed
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo")

from  graph_zoo.python.utils_deepwalk import partition_num
class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.G = graph#输入图网络
        self.w2v_model = None#word2vec模型
        
        self.walk_length = walk_length#游走序列最大长度
        self.num_walks = num_walks# 每个节点作为起始节点生成随机游走序列的个数
        self.workers = workers#线程数

    def deepwalk_walk(self, walk_length, start_node):#输入游走长度和开始节点返回游走序列

        walk = [start_node]#保存节点起点

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))#汇总所有邻接节点
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))#随机选择邻接节点
            else:
                break#没有邻接节点，直接跳出循环，返回已保存序列
        return walk

    
    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        #并行计算游走序列
        G = self.G

        nodes = list(G.nodes())

        results = Parallel(#开始并行计算
            n_jobs=workers,#线程数
            verbose=verbose,#控制并行计算的详细程度
        )(
            delayed(self._simulate_walks)(nodes, num, walk_length)#并行调用游走函数
            for num in partition_num(num_walks, workers)#任务平均分配
        )

        walks = list(itertools.chain(*results))

        return walks#返回游走序列

    def _simulate_walks(#被并行调用游走函数
        self,
        nodes,
        num_walks,
        walk_length,
    ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)#每次循环将节点顺序打乱
            for v in nodes:
                walks.append(
                        self.deepwalk_walk(walk_length=walk_length, start_node=v)
                    )
                    
                
        return walks

    def get_randomwalk(self):#返回截断随机游走序列
        self.sentences = self.simulate_walks(#
            self.num_walks, self.walk_length, self.workers, verbose=1
        )
        return self.sentences
