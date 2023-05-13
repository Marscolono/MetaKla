
import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc,precision_recall_curve

# from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import argparse
#----->>
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

from collections import Counter
import math
import re
# from time import time
import prettytable as pt
# from scipy import interp
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence

import time
from tqdm import tqdm

from Bio import SeqIO

def load_fasta(data):
	seq_dict = {}
	id_list = []
	seq_list = []
	
	pos_list = []
	neg_list = []
	for seq_record in SeqIO.parse(data, "fasta"):
		id = seq_record.id
		seq = str(seq_record.seq)
		if id =='1':
			pos_list.append(seq)
		elif id =='0':
			neg_list.append(seq)
		seq_dict[id] = seq
		id_list.append(id)
		seq_list.append(seq)
	return seq_dict, id_list, seq_list, pos_list, neg_list

def transform_token2index(sequences):  #输入序列列表，将词源转换成索引，返回每条序列的索引列表表示，以及最大长度
	# token2index = pickle.load(open('/home/lz/New/6mA_project/residu2idx/residue29idx.pkl', 'rb'))
	token2index = {'+': 0, '>': 1, '|': 2, '#': 3, 'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9, 'G': 10, 'K': 11, 'Y': 12, 'P': 13, 'H': 14, 'Z': 15, 'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20, 'R': 21, 'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27, '-':28, 'O':29} #'>' is [CLS], '|' is [SEP], '+' is [PAD], '#' is [MASK]
	
	print(token2index) #打印的token2index
	config.token2index = token2index
	
	for i, seq in enumerate(sequences):
		sequences[i] = list(seq)
	
	token_list = list()
	max_len = 0
	for seq in sequences:
		seq_id = [token2index[residue] for residue in seq]
		token_list.append(seq_id)
		if len(seq) > max_len:
			max_len = len(seq)
	print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
	print('sequences_residue', sequences[0:5])  #显示前5个序列列表
	print('token_list', token_list[0:5])        #显示前5个索引列表
	return token_list, max_len


def make_data_with_unified_length(token_list, max_len): #添加首位占位符，用0填充短序列，仍是一维列表
	# token2index = pickle.load(open('/home/lz/New/6mA_project/residu2idx/residue29idx.pkl', 'rb'))
	
	token2index = config.token2index
	data = []
	for i in range(len(token_list)):
		if config.Add_CLS_SEP == True:
			token_list[i] = [token2index['>']] + token_list[i] + [token2index['|']] #前后加上[CLS]，[SEP]
		else:
			token_list[i] = token_list[i] #前后什么也不加
		
		# [17, 10, 23, 17, 23, 18, 10, 18, 23, 17, 18, 23, 10, 23, 23, 18, 17, 17, 10, 10, 18, 23, 18, 18, 17, 17, 23, 18, 10, 18, 17, 18, 17, 17, 17, 10, 10, 23, 17, 18, 10] 
		#-->> 
		# [1, 17, 10, 23, 17, 23, 18, 10, 18, 23, 17, 18, 23, 10, 23, 23, 18, 17, 17, 10, 10, 18, 23, 18, 18, 17, 17, 23, 18, 10, 18, 17, 18, 17, 17, 17, 10, 10, 23, 17, 18, 10, 2]
		
		if max_len >= len(token_list[i]): #如果每个序列比man_len短，用0填充
			n_pad = max_len - len(token_list[i])
			token_list[i].extend([0] * n_pad)    #在序列短的的输入填充0
			data.append(token_list[i])
			
		elif max_len < len(token_list[i]): #如果每个序列比man_len长，截取前max_len个元素
			data.append(token_list[i][:max_len])
	
	print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
	print('--> max(config.max_len ,max_len_train/test):', max_len)
	print('token_list + [pad]', token_list[0:5])
	
	return data


	
def load_train_val_bicoding(data):#(path_pos_data, path_neg_data): #切割正负样本，并带标签
	
	# sequences_pos = load_data_bicoding(path_pos_data)
	# sequences_neg = load_data_bicoding(path_neg_data)
	
	seq_dict, id_list, seq_list, sequences_pos, sequences_neg = load_fasta(data)
	
	token_list_pos, max_len_pos = transform_token2index(sequences_pos) #打印1次
	token_list_neg, max_len_neg = transform_token2index(sequences_neg) #打印1次
	# token_list_train: [[1, 5, 8], [2, 7, 9], ...]
	max_len_train = max(max_len_pos,max_len_neg) 
	# max_len = max(config.max_len ,max_len_train)  # 取config和实际数据的最大长度
	max_len = config.max_len #取config定义的长度
	
	print('^-^ ========>> data_train_max_len: ', max_len_train)
	# config.max_len = max_len + 2
	# max_len = config.max_len
	
	Positive_X = make_data_with_unified_length(token_list_pos,max_len) #打印的即上一行命令的max_len
	Negitive_X = make_data_with_unified_length(token_list_neg,max_len)
	
	data_train = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])
	
	np.random.seed(42)
	np.random.shuffle(data_train)

	X = np.array([_[:-1] for _ in data_train])
	y = np.array([_[-1] for _ in data_train])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/8, random_state=42) #分割train/valid

	return X_train, y_train, X_test, y_test
	
def load_test_bicoding(data):
	
	# sequences_pos = load_data_bicoding(path_pos_data)
	# sequences_neg = load_data_bicoding(path_neg_data)
	
	seq_dict, id_list, seq_list, sequences_pos, sequences_neg = load_fasta(data)
	
	token_list_pos, max_len_pos = transform_token2index(sequences_pos) #打印1次
	token_list_neg, max_len_neg = transform_token2index(sequences_neg) #打印1次
	# token_list_train: [[1, 5, 8], [2, 7, 9], ...]
	max_len_test = max(max_len_pos,max_len_neg)
	# max_len = max(config.max_len ,max_len_test) #取config和实际数据的最大长度
	max_len = config.max_len #取config定义的长度
	
	print('^-^ ========>> data_test_max_len: ', max_len_test)
	
	Positive_X = make_data_with_unified_length(token_list_pos,max_len)
	Negitive_X = make_data_with_unified_length(token_list_neg,max_len)

	data_test = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])
	
	np.random.seed(42)
	np.random.shuffle(data_test)

	X_test = np.array([_[:-1] for _ in data_test])
	y_test = np.array([_[-1] for _ in data_test])
	
	return X_test, y_test


def load_in_torch_fmt(X_train, y_train):
	# X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/vec_len), vec_len)
	# X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/vec_len), vec_len)
	#print(X_train.shape)
	X_train = torch.from_numpy(X_train).long()
	y_train = torch.from_numpy(y_train).long()
	#y_train = torch.from_numpy(y_train).float()
	# X_test = torch.from_numpy(X_test).float()
	# X_test, y_test = shuffleData(X_train, y_train)
	return X_train, y_train

# Positive_X = Positive_X.reshape(Positive_X.shape[0], int(Positive_X.shape[1]/vec_len), vec_len) #折叠
# Negitive_X = Negitive_X.reshape(Negitive_X.shape[0], int(Negitive_X.shape[1]/vec_len), vec_len) #折叠

class Embedding(nn.Module): #将索引列表表示成tok+pos：[batch, seq_len+2, d_model]
	def __init__(self, config):
		super(Embedding, self).__init__()
		global max_len, d_model, vocab_size	
		max_len = config.max_len + 2	  #41
		d_model = config.d_model 		  #32
		vocab_size = config.vocab_size    #30
		
		self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table) 
		self.pos_embed = nn.Embedding(max_len, d_model)     # position embedding
		self.norm = nn.LayerNorm(d_model)
	
	def forward(self, x):
		seq_len = x.size(1)  # x: [batch_size, seq_len] #[64,43]
		pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
		# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
		#     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
		#     36, 37, 38, 39, 40, 41, 42])
		pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len] #[64,43]
		# tensor([[ 0,  1,  2,  ..., 40, 41, 42],
		#         [ 0,  1,  2,  ..., 40, 41, 42],
		#         [ 0,  1,  2,  ..., 40, 41, 42],
		#         ...,
		#         [ 0,  1,  2,  ..., 40, 41, 42],
		#         [ 0,  1,  2,  ..., 40, 41, 42],
		#         [ 0,  1,  2,  ..., 40, 41, 42]])
		embedding = self.pos_embed(pos)           # [64, 43, 32]
		embedding = embedding + self.tok_embed(x) # [64, 43, 32] + [64, 43, 32]
		embedding = self.norm(embedding)
		return embedding

#保存模型 ----
def save_checkpoint(state,is_best,OutputDir,test_index):
	if is_best:
		print('=> Saving a new best from epoch %d"' % state['epoch'])
		torch.save(state, OutputDir + '/' + str(test_index) +'_checkpoint.pth.tar')
		
	else:
		print("=> Validation Performance did not improve")
		
def ytest_ypred_to_file(y, y_pred, out_fn):#保存标签和预测值，用来画roc曲线等等工作
	with open(out_fn,'w') as f:
		for i in range(len(y)):
			f.write(str(y[i])+'\t'+str(y_pred[i])+'\n')
			
			
def chunkIt(seq, num): #将序列平均分割成num份
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
		
	return out

def shuffleData(X, y):
	index = [i for i in range(len(X))]
	# np.random.seed(42)
	random.shuffle(index)
	new_X = X[index]
	new_y = y[index]
	return new_X, new_y

def shuffleData_index(X): #返回打乱后的索引列表，然后取前80%做训练，后20%做验证
	index = [i for i in range(len(X))]
	random.seed(42)
	random.shuffle(index)
	split_edge = int(len(X)*0.8) #分割界限
	train_index = index[:split_edge]
	valid_index = index[split_edge:]
	return train_index,valid_index

def round_pred(pred,threshold):
	# list_result = []
	# for i in pred:
	# 	if i >0.5:
	# 		list_result.append(1)
	# 	elif i <=0.5:
	# 		list_result.append(0)
	# threshold = 0.5
	list_result = [0 if instance < threshold else 1 for instance in list(pred)]
	return torch.tensor(list_result)

#------------------>>>
def get_entropy(probs):
	ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
	# A.mean(0) 计算每一列的平均值
	return ent

def get_cond_entropy(probs):
	cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
	return cond_ent
	
class Tim_loss(nn.Module):
	def __init__(self):
		super(Tim_loss, self).__init__()
		
		self.criterion_loss = nn.CrossEntropyLoss()
	
	def forward(self, logits, label):
		loss = self.criterion_loss(logits.view(-1, config.num_class), label.view(-1)) #交叉熵结果
		loss = (loss.float()).mean() #求平均
		loss = (loss - config.b).abs() + config.b      #b = 0.06
		
		# Q_sum = len(logits)
		logits = F.softmax(logits, dim=1)  # softmax归一化
	
		sum_loss = loss + get_entropy(logits) - get_cond_entropy(logits)
		return sum_loss[0]

def calculateScore(y, pred_y):
	y = y.data.cpu().numpy()
	tempLabel = np.zeros(shape = y.shape, dtype=np.int32) #将预测小数变成0或者1 # y_int
	
	for i in range(len(y)):
		if pred_y[i] < config.threshold: #如果想看不需要best_threshold的结果，将值调回0.5
			tempLabel[i] = 0;
		else:
			tempLabel[i] = 1;
			
	accuracy = metrics.accuracy_score(y, tempLabel)
	
	confusion = confusion_matrix(y, tempLabel)
	TN, FP, FN, TP = confusion.ravel()
	
	sensitivity = recall_score(y, tempLabel)
	specificity = TN / float(TN+FP)
	MCC = matthews_corrcoef(y, tempLabel)
	
	F1Score = (2 * TP) / float(2 * TP + FP + FN)
#	precision = TP / float(TP + FP)
	
	precision = metrics.precision_score(y, tempLabel, pos_label=1)  # 精确度
	recall = metrics.recall_score(y, tempLabel, pos_label=1)
	
	pred_y = pred_y.reshape((-1, ))
	
	# ROCArea = roc_auc_score(y, pred_y)
	ROCArea = metrics.roc_auc_score(y, pred_y)
	fpr, tpr, thresholds = roc_curve(y, pred_y)
	
	pre, rec, threshlds = precision_recall_curve(y, pred_y)
	pre = np.fliplr([pre])[0] 
	rec = np.fliplr([rec])[0]  
	AUC_prec_rec = np.trapz(rec,pre)
	AUC_prec_rec = abs(AUC_prec_rec)
	
	# print('sn' , sensitivity, 'sp' , specificity, 'acc' , accuracy, 'MCC' , MCC, 'AUC' , ROCArea,'precision' , precision, 'F1' , F1Score)
	
	metrics_main7 = {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea,'precision' : precision, 'F1' : F1Score}
	
	#----------------->>>
	
	# print outcome
	try:
		tb = pt.PrettyTable()
		tb.field_names = ["model name" ]+ [str(i) for i in metrics_main7.keys()]
		# for key, values in metrics_result.items():
		text = [config.model_name]+[round(i,4) for i in metrics_main7.values()] #显示4位小数
		tb.add_row(text)
		print(tb)
	except NameError:
		print(metrics_main7)
	
	metrics_all = {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea,'precision' : precision, 'F1' : F1Score,'recall' : recall , 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds,'pre_recall_curve':AUC_prec_rec,'prec':pre,'reca':rec}
	
	return metrics_all
	
def analyze_sigle(temp, OutputDir, species,state):
	
	trainning_result = temp;
	#写文件
	file = open(OutputDir + '/{0}_performance_{1}.txt'.format(species,state), 'w') #创建文件
	
	for x in [trainning_result]:
	
		title = '{0}ing_{1}_'.format(state, species)
		
		file.write(title +  'results\n') #第一次写入标题
		
		for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1']:
			
			total = [] #逐个加入，逐个计算
			
			for val in x:
				total.append(val[j])
			#第二次写入（逐个写入'sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue'的值）
			file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n') 
			
		# file.write('\n\n______________________________\n') #第三次写入
	file.close();
	
	#画图
	for x in [trainning_result]:
		
		tprs = []
		aucs = []
		mean_fpr = np.linspace(0, 1, 100)
		
		#画10折ROC曲线
		#************************** ROC Curve*********************************
		for val in x: # 10个{}
			tpr = val['tpr']
			fpr = val['fpr']
			tprs.append(np.interp(mean_fpr, fpr, tpr))
			tprs[-1][0] = 0.0
			roc_auc = auc(fpr, tpr)
			aucs.append(roc_auc)
			plt.plot(fpr, tpr, lw=1, alpha=0.3,label='%s (AUC = %0.2f)' % (config.model_name, roc_auc)) 
	
		print;
		
		#画对角斜线
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8) 
		
		#画平均ROC
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		plt.plot(mean_fpr, mean_tpr, color='b',
				label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
				lw=2, alpha=.8)
		
		#画误差区域
		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.') 
		#整张图的设置
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic curve')
		plt.legend(loc="lower right")
		
	
		title = '{}ing_'.format(state)
		
		plt.savefig( OutputDir + '/' + title +'ROC.png')
		plt.close('all');
	
		#************************** Precision Recall Curve*********************************
		prs = []
		pre_aucs = []
		mean_recal= np.linspace(0, 1, 100)
		for val in x:
			pre = val['prec']
			rec = val['reca']
			prs.append(np.interp(mean_recal, rec, pre))
			prs[-1][0] = 0.0
			p_r_auc = auc(rec, pre)
			pre_aucs.append(p_r_auc)
			plt.plot(rec, pre, lw=1, alpha=0.3,label='%s (AUC = %0.2f)' % (config.model_name, p_r_auc))
			
		print;
		
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)
		
		mean_pre = np.mean(prs, axis=0)
		mean_pre[-1] = 1.0
		mean_auc = auc(mean_recal, mean_pre)
		std_auc = np.std(pre_aucs)
		plt.plot(mean_recal, mean_pre, color='b',
				label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
				lw=2, alpha=.8)
		
		std_pre = np.std(prs, axis=0)
		pre_upper = np.minimum(mean_pre + std_pre, 1)
		pre_lower = np.maximum(mean_pre - std_pre, 0)
		plt.fill_between(mean_recal, pre_lower, pre_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.')
		
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision Recall curve')
		plt.legend(loc="lower right")
		
		title = '{}ing_'.format(state)
			
		plt.savefig( OutputDir + '/' + title +'Pre_R_C.png')
		plt.close('all')

def analyze(temp, OutputDir, species):
	trainning_result, validation_result, testing_result = temp;
	
	#写文件
	file = open(OutputDir + '/{}_performance_allindex.txt'.format(species), 'w') #创建文件
	
	index = 0
	for x in [trainning_result, validation_result, testing_result]:
		
		title = ''
		if index == 0:
			title = 'training_{}_'.format(species)
		if index == 1:
			title = 'validation_{}_'.format(species)
		if index == 2:
			title = 'testing_{}_'.format(species)
			
		index += 1;
		
		file.write(title +  'results\n') #第一次写入标题
		
		for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1']:
			
			total = [] #逐个加入，逐个计算
			
			for val in x:
				total.append(val[j])
			#第二次写入（逐个写入'sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue'的值）
			file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n') 
			
		file.write('\n\n______________________________\n') #第三次写入
	file.close();
	
	#画图
	index = 0
	for x in [trainning_result, validation_result, testing_result]:
		
		tprs = []
		aucs = []
		mean_fpr = np.linspace(0, 1, 100)
		
		#画10折ROC曲线
		#************************** ROC Curve*********************************
		i = 0
		for val in x: # 10个{}
			tpr = val['tpr']
			fpr = val['fpr']
			tprs.append(np.interp(mean_fpr, fpr, tpr))
			tprs[-1][0] = 0.0
			roc_auc = auc(fpr, tpr)
			aucs.append(roc_auc)
			plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc)) 
		
			i += 1
		
		print;
		
		#画对角斜线
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8) 
		
		#画平均ROC
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		plt.plot(mean_fpr, mean_tpr, color='b',
				label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
				lw=2, alpha=.8)
		
		#画误差区域
		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.') 
		#整张图的设置
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic curve')
		plt.legend(loc="lower right")
		
		title = ''
		if index == 0:
			title = 'training_'
		if index == 1:
			title = 'validation_'
		if index == 2:
			title = 'testing_'
			
		plt.savefig( OutputDir + '/' + title +'ROC_all.png')
		plt.close('all');
	
		#************************** Precision Recall Curve*********************************
		i = 0
		prs = []
		pre_aucs = []
		mean_recal= np.linspace(0, 1, 100)
		for val in x:
			pre = val['prec']
			rec = val['reca']
			prs.append(np.interp(mean_recal, rec, pre))
			prs[-1][0] = 0.0
			p_r_auc = auc(rec, pre)
			pre_aucs.append(p_r_auc)
			plt.plot(rec, pre, lw=1, alpha=0.3,label='PRC fold %d (AUC = %0.2f)' % (i+1, p_r_auc))
			
			i += 1
			
		print;
		
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)
		
		mean_pre = np.mean(prs, axis=0)
		mean_pre[-1] = 1.0
		mean_auc = auc(mean_recal, mean_pre)
		std_auc = np.std(pre_aucs)
		plt.plot(mean_recal, mean_pre, color='b',
				label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
				lw=2, alpha=.8)
		
		std_pre = np.std(prs, axis=0)
		pre_upper = np.minimum(mean_pre + std_pre, 1)
		pre_lower = np.maximum(mean_pre - std_pre, 0)
		plt.fill_between(mean_recal, pre_lower, pre_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.')
		
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision Recall curve')
		plt.legend(loc="lower right")
		
		title = ''
		
		if index == 0:
			title = 'training_'
		if index == 1:
			title = 'validation_'
		if index == 2:
			title = 'testing_'
			
		plt.savefig( OutputDir + '/' + title +'Pre_R_C_all.png')
		plt.close('all')
		
		index += 1;
		
'''
def get_kfold_data(k, i, X, y):  #拆分成训练和测试集合
	
	# 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
	fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
	
	val_start = i * fold_size
	if i != k - 1:
		val_end = (i + 1) * fold_size
		X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
		X_train = torch.cat((X[0:val_start], X[val_end:]), dim = 0)
		y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
	else:  # 若是最后一折交叉验证
		X_valid, y_valid = X[val_start:], y[val_start:]     # 若不能整除，将多的case放在最后一折里,把数据利用的很完整
		X_train = X[0:val_start]
		y_train = y[0:val_start]
	
	return X_train, y_train, X_valid,y_valid

'''
def get_k_fold_data(k, i, X, y):  # 改装，用所有数据
	
	# 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
	fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
	
	val_start = i * fold_size
	if i != k - 1:
		val_end = (i + 1) * fold_size
		X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
		X_train = np.concatenate([X[0:val_start], X[val_end:]], 0)
		y_train = np.concatenate([y[0:val_start], y[val_end:]], 0)
	else:  # 若是最后一折交叉验证
		X_valid, y_valid = X[val_start:], y[val_start:]     # 若不能整除，将多的case放在最后一折里,把数据利用的很完整
		X_train = X[0:val_start]
		y_train = y[0:val_start]
	
	return X_train, y_train, X_valid, y_valid
	
#!! Adapt moedl --------->>    +
class Adapt_emb_CNN(nn.Module):
	def __init__(self,config):
		super(Adapt_emb_CNN, self).__init__()
		
		kernel_size = config.kernel_size  #10
		num_class = config.num_class	
	
		self.embedding = Embedding(config)
		#---------------->>>
		self.conv1 = nn.Sequential(  # input shape [batch , 4, 41]
			nn.Conv1d(
					in_channels=32,      # input height
					out_channels=256,    # n_filters
					kernel_size = kernel_size),     # filter size
					#padding = int(kernel_size/2)),
			# padding=(kernel_size-1)/2 当 stride=1
			# nn.ReLU(),    # activation
			nn.LeakyReLU(0.1),
			nn.MaxPool1d(kernel_size=2),
			nn.BatchNorm1d(256),# 设置的参数就是卷积的输出通道数
			nn.Dropout())
		# self.leaky_relu = nn.LeakyReLU()
		self.fc_task = nn.Sequential(
			nn.Linear(256, 32),
			nn.Dropout(0.5),
			# nn.ReLU(),
			nn.LeakyReLU(0.1),
			nn.Linear(32, 2),
		)
		self.classifier = nn.Linear(2, int(num_class))
	
	#---------------->>>
	def forward(self, x):
		x = x.long() 											 #[64, 43]
		x = self.embedding(x)  # [bach_size, seq_len, d_model]   #[64,43,32]
		x = x.transpose(1, 2)	 # [64,43,32] --> [64,32,43]
		x = self.conv1(x) 		 # [64, 256, 17]
		x = x.transpose(1, 2)    # [64, 17, 256]
		#rnn layer
		# out, _ = self.lstm(x, (h0, c0))
		reduction_feature = self.fc_task(torch.mean(x, 1)) #[64, 17, 256]-->[64,256]-->>[64, 2]
		representation = reduction_feature				   #[64, 2]
		logits_clsf = self.classifier(representation)      #[64, 1]
		logits_clsf = torch.sigmoid(logits_clsf)       	   #[64, 1]
		#输出sigmoid结果 torch.sigmoid(out) / F.softmax(out)
		#!!! 输出sigmoid/softmax
		return logits_clsf, representation


def load_config():
	parse = argparse.ArgumentParser(description='Model set')
	# preoject setting
	parse.add_argument('-learn-name', type=str, default='Adapt/Onehot/Hmm..', help='learn name')

	parse.add_argument('-species', type=str, default='Kla', help='880/15w and so on ')
	parse.add_argument('-max-len', type=int, default=51, help='the real max length of input sequences')
	# parse.add_argument('-choose-len', type=int, default=51, help='choose length of input sequences')
	
	#截取长度
	parse.add_argument('-mode',default='train',help="Set the model to train/test/all")
	parse.add_argument('-loss-Function', type=str, default='Tim', help='BE, Cr, Tim, Get_loss, FL, Poly, DVIB_loss,remember adjust -num-class')
	
	parse.add_argument('-train-fasta', type=str, default='./Kla_data/Kla_data/upTrain.fa', help=' so on ')
	# parse.add_argument('-test-fasta', type=str, default='./Kla_data/Kla_data/fungiForTest.fa', help=' so on ')
	
	parse.add_argument('-d-model', type=int, default=32, help='the dim of embbending layer')
	parse.add_argument('-vocab-size', type=int, default=30, help='the number of diction in pkl file')
	
	
	# training parameters
	parse.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parse.add_argument('-batch-size', type=int, default=256, help='number of samples in a batch')
	parse.add_argument('-kernel-size', type=int, default=8, help='number of kernel size')
	parse.add_argument('-epoch', type=int, default=50, help='number of iteration')  # 30
	parse.add_argument('-k-fold', type=int, default= -1 , help='10,-1 represents train-test approach')
	
	# 如果想讲train和test拼在一起做k折，下面命令改为True
	parse.add_argument('-concat_traintest', type=bool, default=True, help='是否拼接train和test的数据做k折')
	parse.add_argument('-ensemble', type=bool, default=False,help='是否融合k折所有模型预测独立测试集的结果')
	
	parse.add_argument('-num-class', type=int, default=2, help='number of classes')
	parse.add_argument('-cuda', type=bool, default=True, help='if not use cuda')
	parse.add_argument('-device', type=int, default=2, help='device id')
	
	parse.add_argument('-Add-CLS-SEP', type=bool, default=False, help='add:[CLS],[SEP]')
	parse.add_argument('-gradient-clipping', type=bool, default=True, help=' avoid exploding gradient')
	parse.add_argument('-best-threshold', type=bool, default=True, help='if rechoose a new threshold not 0.5 in test/G-mean')
	parse.add_argument('-threshold', type=float, default=0.5, help='use to convert the float to int in result')
		
	#对抗模块
	parse.add_argument('-adversarial', type=bool, default= False) #True
	
	config = parse.parse_args()
	return config



#!! Main  +_+
if __name__ == '__main__':
	# Hyper Parameters------------------>>
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	np.random.seed(42)
	random.seed(42)
	torch.backends.cudnn.deterministic = True
	
	config = load_config()
	
	config.b = 0.06
	
	loss_F = config.loss_Function
	
	if loss_F == 'BE':
		config.num_class = 1
	else:
		config.num_class = 2
		
	'''set device'''
	torch.cuda.set_device(config.device) #选择默认GPU：1
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device_cpu = torch.device("cpu") #在cpu上测试

	net = Adapt_emb_CNN(config).to(device)
	print(net)
	
	net_name = str(net).split('(')[0]
	config.model_name = net_name
	
	# k_folds = 10
	EPOCH = config.epoch
	BATCH_SIZE = config.batch_size
	LR = config.lr
	kernel_size = config.kernel_size
	
	trainning_result = []
	validation_result = []
	testing_result = []
	
	train_loss_sum, valid_loss_sum, test_loss_sum= 0, 0, 0
	train_acc_sum , valid_acc_sum , test_acc_sum = 0, 0, 0
	
	test_acc = []     
	test_auc = []
	test_losses = []   

	k_folds = config.k_fold  
	if k_folds == -1:       
		all_index = ['-1','-2','-3'] 
	elif k_folds != -1:
		all_index = [i for i in range(k_folds)]
	
	#-------------------------------------------------->>>
	metrics_test_index = {'ACC':[],'AUC':[],'SN':[],'SP':[],'MCC':[],'F1':[],'precision':[],'recall':[]}
	
	species = config.species
	# data_path = './{0}_data/{0}_train_test/'.format(species)
	# train_pos_fa = data_path+'{0}_positive_train.fa'.format(species)
	# train_neg_fa = data_path+'{0}_negative_train.fa'.format(species)
	# test_pos_fa = data_path +'{0}_positive_test.fa'.format(species)
	# test_neg_fa = data_path +'{0}_negative_test.fa'.format(species)
	
	train_all_X, train_all_y = load_test_bicoding(config.train_fasta)
	print('train_all_X:',train_all_X.shape[0])
	
	#------------------------------------------------------------------------------------>>
	#load data over
	
	for index_fold in all_index:  # index_fold  # test_index
		print('*'*45,'第', '{}/{}'.format(index_fold,len(all_index)) ,'折','*'*45)
		X_train, y_train, X_valid, y_valid, X_test, y_test = [],[],[],[],[],[]
		train_sigle_result,valid_sigle_result,test_sigle_result = [],[],[]

		OutputDir = './model_result/{0}/{1}_{2}_{0}_{3}/fold_{4}'.format(species,net_name,config.loss_Function,k_folds, index_fold)
		config.metrics_save_dir = '/'.join(OutputDir.split('/')[:-1])
		# './model_result/{0}/{1}_{2}_{0}_{3}'.format(species,net_name,config.loss_Function,k_folds) 
		
		OutputDir_tsne_pca = OutputDir +'/Out_tsne_pca'
		if os.path.exists(OutputDir_tsne_pca):
			print('OutputDir is exitsted')
		else:
			os.makedirs(OutputDir_tsne_pca)
			print('success create dir test')
		
		if k_folds == -1: #independent test
			train_index, valid_index = shuffleData_index(train_all_X) 
			X_train = train_all_X[train_index] 
			X_valid = train_all_X[valid_index]
		
			y_train = train_all_y[train_index] 
			y_valid = train_all_y[valid_index]
			
			X_train, y_train = load_in_torch_fmt(X_train, y_train)
			X_valid, y_valid = load_in_torch_fmt(X_valid, y_valid)
			print('Train数据:',X_train.shape[0])
			print('Valid 数据:',X_valid.shape[0])

		
		elif k_folds != -1: #k-fold cross-validation
			X_train, y_train, X_valid, y_valid = get_k_fold_data(k_folds,int(index_fold),train_all_X,train_all_y)
			X_train, y_train = load_in_torch_fmt(X_train, y_train)
			X_valid, y_valid = load_in_torch_fmt(X_valid, y_valid)
				
			
			
		if 'train' in config.mode :
			train_loader = Data.DataLoader(Data.TensorDataset(X_train,y_train), BATCH_SIZE, shuffle = True)
			val_loader = Data.DataLoader(Data.TensorDataset(X_valid, y_valid), BATCH_SIZE, shuffle = False)
		
		
		model = Adapt_emb_CNN(config).to(device)
		
		if loss_F == 'Tim':
			criterion = Tim_loss()
			
		t0 = time.time()
		
		optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
		# optimizer = torch.optim.AdamW(params = model.parameters(), lr=LR, weight_decay=0.0025)
		
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95) #动态学习率
		
		train_losses = []
		val_losses = [] 
		
		train_acc = []
		val_acc = []
		
		#保存--
		best_acc = 0
		best_loss = 500
		patience = 0
		patience_limit = 10
		
		epoch_list = [] #统计用到的epoch，用于画图
		torch_val_best, torch_val_y_best  = torch.tensor([]),torch.tensor([]) #用于保存最好的epoch的val结果
		
		
		
		if 'train' in config.mode :
			for epoch in range(EPOCH):
				t1 = time.time()
				repres_list, label_list = [],[]
				
				torch_train, torch_train_y = torch.tensor([]),torch.tensor([])
				torch_val, torch_val_y     = torch.tensor([]),torch.tensor([])
				
				train_losses_in_epoch = []
				
				model.train() #!!! Train
				if epoch % 10 == 0 and epoch > 0: #间隔调整
					scheduler.step()
					print("Learning rate: %.16f" % optimizer.state_dict()['param_groups'][0]['lr'] )
					
				correct = 0
				train_loss = 0

				for step, batch in enumerate(train_loader):
					train_x, train_y = batch
					train_x = Variable(train_x, requires_grad=False).to(device)
					train_y = Variable(train_y, requires_grad=False).to(device)
					
					if loss_F == 'BE':
						fx, presention = model(train_x)
						loss = criterion(fx.squeeze(), train_y.type(torch.FloatTensor).to(device))   	#BE
					
					elif loss_F == 'Tim':
						fx, presention = model(train_x)
						loss = criterion(fx, train_y)                                                 #Cross
					
					optimizer.zero_grad()
					
					if config.gradient_clipping == True:
						clip_value = 1
						torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
					
					train_losses_in_epoch.append(loss.item())
					loss.backward()
					
					optimizer.step()
					
					repres_list.extend(presention.cpu().detach().numpy())
					label_list.extend(train_y.cpu().detach().numpy())
					
					if loss_F == 'BE':
						pred = round_pred(fx.data.cpu().numpy(),threshold=config.threshold).to(device)
						pred_prob = fx
					elif loss_F == 'Tim':
						pred = fx.max(1, keepdim = True)[1]
						pred_prob = fx[:,1]
					
					correct += pred.eq(train_y.view_as(pred)).sum().item()
					train_loss += loss.item() * len(train_y)
					
					torch_train = torch.cat([torch_train,pred_prob.data.cpu()],dim=0)
					torch_train_y = torch.cat([torch_train_y,train_y.data.cpu()],dim=0)
					
					if (step+1) % 10 == 0:
						print ('Epoch : %d/%d, Iter : %d/%d,  epoch_Loss: %.4f'%(epoch + 1, EPOCH, 
																			step + 1, len(X_train)//BATCH_SIZE, 
																			loss.item()))
				
				train_losses.append(np.average(train_losses_in_epoch))
				epoch_list.append(epoch)
				accuracy_train = 100.*correct/len(X_train)
				
				time_ = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
				print('Epoch: {}/{}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%), Time_use: {:.3f}s, Time:{}'.format(epoch + 1,EPOCH, np.average(train_losses_in_epoch), correct, len(X_train), accuracy_train, (time.time()-t1), time_ ))
				train_acc.append(accuracy_train)
				
				model.eval() #!!! Valid
				val_loss = 0
				correct = 0
				repres_list_valid , label_list_valid = [],[]
				
				with torch.no_grad():
					for step, batch in enumerate(val_loader):     #val_loader
						(valid_x, valid_y) = batch
						valid_x = Variable(valid_x, requires_grad=False).to(device) #----cuda
						valid_y = Variable(valid_y, requires_grad=False).to(device) #----cuda
						
						optimizer.zero_grad()
							
						if loss_F == 'BE':
							y_hat_val, presention_valid = model(valid_x) 
							loss = criterion(y_hat_val.squeeze(), valid_y.type(torch.FloatTensor).to(device)).item()   #BE
							pred_val = round_pred(y_hat_val.data.cpu().numpy(),threshold=config.threshold).to(device)  #BE
							# pred_val = torch.tensor([0 if instance < 0.5 else 1 for instance in y_hat_val.data.cpu().numpy()]).to(device) 
							pred_prob_val = y_hat_val                                                                  #BE
							
						elif loss_F == 'Tim':
							y_hat_val, presention_valid = model(valid_x) 
							loss = criterion(y_hat_val, valid_y.to(device)).item()      # batch average loss		  #Cross
							pred_val = y_hat_val.max(1, keepdim = True)[1]					        #Cross: softmax
							pred_prob_val = y_hat_val[:,1] #取第二类别的概率						#Cross
							
							
						val_loss += loss * len(valid_y)# sum up batch loss
						
						repres_list_valid.extend(presention_valid.cpu().detach().numpy())
						label_list_valid.extend(valid_y.cpu().detach().numpy())
						
						repres_list.extend(presention_valid.cpu().detach().numpy())
						label_list.extend(valid_y.cpu().detach().numpy())
						
						correct += pred_val.eq(valid_y.view_as(pred_val)).sum().item()
						torch_val = torch.cat([torch_val,pred_prob_val.data.cpu()],dim=0)
						torch_val_y = torch.cat([torch_val_y,valid_y.data.cpu()],dim=0)
			
				val_losses.append(val_loss/len(X_valid))
				accuracy_valid = 100.*correct/len(X_valid)
				
				print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
					val_loss/len(X_valid), correct, len(X_valid), accuracy_valid))
					
				val_acc.append(accuracy_valid)
				
				cur_acc = accuracy_valid
				cur_loss = val_losses[-1]
				
				is_best = bool(cur_acc >= best_acc and cur_loss < best_loss )
				best_acc = max(cur_acc, best_acc)
				
				if is_best:
					torch_val_best = torch_val
					torch_val_y_best = torch_val_y
					
				save_checkpoint(
					{'epoch': epoch+1,
					'state_dict': model.state_dict(),
					'best_accuracy': best_acc,
					'optimizer': optimizer.state_dict()},
				is_best,OutputDir,index_fold)

				if not is_best:
					patience+=1
					if patience >= patience_limit:
						break
				else:
					patience = 0
				print('> best acc:',best_acc)
			
			torch.save(model.state_dict(), OutputDir + '/trained_model.pkl')
			print('the last trained model saved to \'%s/trained_model.pkl\'' % (OutputDir))
			
			print('-' * 10 + '<-[ End Training ]->' + '-' * 10)
			print("Total time = %ds" % (time.time() - t0))
			
			train_loss_sum += sum(train_losses)/len(train_losses)
			valid_loss_sum += sum(val_losses)/len(val_losses)
			
			train_acc_sum += sum(train_acc)/len(train_acc) 
			valid_acc_sum += sum(val_acc)/len(val_acc)  
			
			print('torch_train:',torch_train.shape[0],'torch_val:',torch_val.shape[0])
			
			print('train:')
			metrics_train = calculateScore(torch_train_y, torch_train.numpy())
			trainning_result.append(metrics_train)
			print('vaild_best:')
			metrics_valid = calculateScore(torch_val_y_best, torch_val_best.numpy())
			validation_result.append(metrics_valid)
			
			temp_train_dict = ([metrics_train])
			analyze_sigle(temp_train_dict, OutputDir,species,'train')
			temp_valid_dict = ([metrics_valid])
			analyze_sigle(temp_valid_dict, OutputDir,species,'valid')
			
			out_val_file = OutputDir+'/val_result_{}.txt'.format(index_fold)
			ytest_ypred_to_file(torch_val_y_best.numpy(), torch_val_best.numpy(), out_val_file)
			
			out_train_file = OutputDir+'/train_result_{}.txt'.format(index_fold)
			ytest_ypred_to_file(torch_train_y.numpy(), torch_train.numpy(), out_train_file)
			
			# draw_figure_train_valid(epoch_list,train_acc,train_losses,val_acc,val_losses,'out_figture_Fold{}'.format(index_fold))
			
			file = open(OutputDir + '/{}_log.txt'.format(species), 'w') 
			file.write('epoch' + ','+'train_acc' +','+ 'train_loss' +','+ 'val_acc' +','+'val_loss'+'\n')
			for i in range(len(epoch_list)):
				file.write(str(epoch_list[i])[:5] +','+str(train_acc[i])[:5] +','+str(train_losses[i])[:5] +','+str(val_acc[i])[:5] +','+str(val_losses[i])[:5] +'\n')
			file.close()
			
			print('   Num of run epoch :{}/{}'.format(len(epoch_list),EPOCH))
			
			
			
			