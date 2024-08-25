import math
import torch
import random
import pyhocon
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo/pytorch")
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo/pytorch/classify")
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo/pytorch/model")
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo") #!加这个

#/mae_pat/xi.sun/venus-zoo/graph_zoo/pytorch/classifiy 
from pytorch.classify.classification import Classification
from pytorch.classify.utils_graphsage import DataCenter
from pytorch.models.GraphSage import SageLayer,GraphSage


def test_graphsage():
	#定义训练函数
	def apply_model(dataCenter, ds, graphSage, classification, b_sz, device, learn_method):
		
		#数据集加载
		test_nodes = getattr(dataCenter, ds+'_test')              # 获取测试集    
		val_nodes = getattr(dataCenter, ds+'_val')                # 获取验证集
		train_nodes = getattr(dataCenter, ds+'_train')            # 获取测试集
		labels = getattr(dataCenter, ds+'_labels')                # 获取标签
		train_nodes = shuffle(train_nodes)                        # 打乱训练数据
		models = [graphSage, classification]
		
		#指定哪些参数是需要在梯度，并对梯度进行初始化
		params = []
		for model in models:                                      # 初始化模型参数
			for param in model.parameters():
				if param.requires_grad:
					params.append(param)
		optimizer = torch.optim.SGD(params, lr=0.7)               # 梯度优化算法
		optimizer.zero_grad()                                     # 梯度清零
		for model in models:                                 
			model.zero_grad()                                
		
		#生成node_batch,并基于node_batch进行训练
		batches = math.ceil(len(train_nodes) / b_sz)              # 有多少个batches
		visited_nodes = set()
		for index in range(batches):
			#接收结果embs_batch
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]  # 选择当前需要训练的节点
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = graphSage(nodes_batch)                   # 学习到节点表征
			assert embs_batch.isnan().sum() == 0  #测试预测结果数值是否存在nan
			assert embs_batch.isinf().sum() == 0  #测试预测结果数值是否存在inf
			
			#计算loss并迭代优化
			if learn_method == 'sup':
				logists = classification(embs_batch)              #  分类函数
				loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)  # 计算损失
				loss_sup /= len(nodes_batch)                      # 计算平均损失
				loss = loss_sup                                   
			print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
			loss.backward()                                       # 方向传播
			for model in models:
				nn.utils.clip_grad_norm_(model.parameters(), 5)  
			optimizer.step()                                      # 更新梯度
			optimizer.zero_grad()                                 # 梯度清零
			for model in models:
				model.zero_grad()                                  
		return graphSage, classification

	def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
		#加载eval数据
		test_nodes = getattr(dataCenter, ds+'_test')
		val_nodes = getattr(dataCenter, ds+'_val')
		labels = getattr(dataCenter, ds+'_labels')
		models = [graphSage, classification]
		params = []
		#为graphSage, classification模型参数，去掉梯度
		for model in models:
			for param in model.parameters():
				if param.requires_grad:
					param.requires_grad = False
					params.append(param)
		#返回结果
		embs = graphSage(val_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_val = labels[val_nodes]
		assert len(labels_val) == len(predicts)
		#基于结果计算f1-score用于模型评估
		comps = zip(labels_val, predicts.data)
		vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
		print("Validation F1:", vali_f1)
		if vali_f1 > max_vali_f1:
			max_vali_f1 = vali_f1
			embs = graphSage(test_nodes)
			logists = classification(embs)
			_, predicts = torch.max(logists, 1)
			labels_test = labels[test_nodes]
			assert len(labels_test) == len(predicts)
			comps = zip(labels_test, predicts.data)
			test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
			print("Test F1:", test_f1)
			for param in params:
				param.requires_grad = True
			torch.save(models, 'model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))
		for param in params:
			param.requires_grad = True
		return max_vali_f1

	random.seed(args.seed)                                                # 固定随机种子
	np.random.seed(args.seed)                                             # 固定随机种子
	torch.manual_seed(args.seed)                                          # 固定随机种子
	torch.cuda.manual_seed_all(args.seed)                                 # 固定随机种子

	config = pyhocon.ConfigFactory.parse_file(args.config)                # 加载配置文件
	ds = args.dataSet                                                     # 获取数据集路径
	dataCenter = DataCenter(config)                                       # 读取数据
	dataCenter.load_dataSet(ds)                                           # 读取数据
	features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)   
	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
	graphSage.to(device)                                                  # 定义graphSage网络
	num_labels = len(set(getattr(dataCenter, ds+'_labels')))              # label的数量
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)                                             # 定义分类网络
	for epoch in range(args.epochs):
		print('----------------------EPOCH %d-----------------------' % epoch)
		graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, args.b_sz, device, args.learn_method)
		if args.learn_method != 'unsup':
			args.max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, args.max_vali_f1, args.name, epoch)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
	parser.add_argument('--dataSet', type=str, default='cora')                # 数据集名字
	parser.add_argument('--agg_func', type=str, default='MAX')               # 采用什么聚合函数
	parser.add_argument('--epochs', type=int, default=50)                     # 迭代次数
	parser.add_argument('--b_sz', type=int, default=20)                       # batch大小
	parser.add_argument('--seed', type=int, default=824)                      # 随机种子
	parser.add_argument('--cuda', action='store_true',help='use CUDA')        # 是否使用GPU
	parser.add_argument('--gcn', action='store_true')                         # 师傅使用GCN
	parser.add_argument('--learn_method', type=str, default='sup')            # 选择损失函数         
	parser.add_argument('--max_vali_f1', type=float, default=0)
	parser.add_argument('--name', type=str, default='debug')                  # 保存文件名
	parser.add_argument('--config', type=str, default='/mae_pat/xi.sun/venus-zoo/graph_zoo/pytorch/classify/experiments.conf')   # 获取的一些配置文件
	args = parser.parse_args()                                               
	#args = parser.parse_known_args()[0]                                       # jupyter 中使用
	device = torch.device("cuda" if args.cuda else "cpu")
	test_graphsage()
	print("test finished!")