
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
import argparse
#---
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
	parse.add_argument('-loss-Function', type=str, default='Tim', help='BE, Cr, Tim, Get_loss, FL, Poly, DVIB_loss,remember adjust -num-class')

	parse.add_argument('-test-fasta', type=str, default='./Kla_data/Kla_data/fungiForTest.fa', help=' so on ')
	parse.add_argument('-d-model', type=int, default=32, help='the dim of embbending layer')
	parse.add_argument('-vocab-size', type=int, default=30, help='the number of diction in pkl file')
	
	# training parameters
	parse.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parse.add_argument('-batch-size', type=int, default=256, help='number of samples in a batch')
	parse.add_argument('-kernel-size', type=int, default=8, help='number of kernel size')
	parse.add_argument('-epoch', type=int, default=50, help='number of iteration')  # 30
	parse.add_argument('-k-fold', type=int, default= -1 , help='10,-1 represents train-test approach')

	parse.add_argument('-num-class', type=int, default=2, help='number of classes')
	parse.add_argument('-cuda', type=bool, default=True, help='if not use cuda')
	parse.add_argument('-device', type=int, default=2, help='device id')
	parse.add_argument('-Add-CLS-SEP', type=bool, default=False, help='add:[CLS],[SEP]')
	parse.add_argument('-gradient-clipping', type=bool, default=True, help=' avoid exploding gradient')
	parse.add_argument('-best-threshold', type=bool, default=True, help='if rechoose a new threshold not 0.5 in test/G-mean')
	parse.add_argument('-threshold', type=float, default=0.5, help='use to convert the float to int in result')
		
	
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
	

	test_all_X, test_all_y   = load_test_bicoding(config.test_fasta)
	seq_dict, id_list, seq_list, pos_list, neg_list = load_fasta(config.test_fasta)
	print('test_all_X:',test_all_X.shape[0])
	
	#------------------------------------------------------------------------------------>>
	#load data over
	X_test, y_test = [],[]
	train_sigle_result,valid_sigle_result,test_sigle_result = [],[],[]

	OutputDir = './model_result/model'
	
	OutputDir_tsne_pca = OutputDir +'/Out_tsne_pca'
	if os.path.exists(OutputDir_tsne_pca):
		print('OutputDir is exitsted')
	else:
		os.makedirs(OutputDir_tsne_pca)
		print('success create dir test')
	
	X_test, y_test = test_all_X, test_all_y
	X_test, y_test   = load_in_torch_fmt(X_test, y_test)
	print('Test 数据:',X_test.shape[0])
	test_loader = Data.DataLoader(Data.TensorDataset(X_test, y_test), BATCH_SIZE, shuffle = False)
	
	
	model = Adapt_emb_CNN(config).to(device)
	
	if loss_F == 'Tim':
		criterion = Tim_loss()
		
	t0 = time.time()
	
	torch.cuda.empty_cache()
	#保存所有输入和预测值
	torch_test, torch_test_y   = torch.tensor([]),torch.tensor([])
	checkpoint = torch.load('./model_result/model/checkpoint.pth.tar') 

	model = net
	print('model loaded...')
	model.load_state_dict(checkpoint['state_dict']) 
	# model.load_state_dict(torch.load(OutputDir + '/trained_model.pkl'))
	
	model.to(device)
	model.eval()
	
	test_loss = 0
	test_correct = 0
	
	for step, batch in enumerate(test_loader):    
		
		(test_x, test_y) = batch
		#y_pred_prob_train = [] #Tensorboard
		# gives batch data, normalize x when iterate train_loader
		test_x = Variable(test_x, requires_grad=False).to(device)  
		test_y = Variable(test_y, requires_grad=False).to(device) 

		if loss_F == 'BE':
			y_hat_test, presention_test = model(test_x)
			loss = criterion(y_hat_test.squeeze(), test_y.type(torch.FloatTensor).to(device)).item()   #BE
			pred_test = round_pred(y_hat_test.data.cpu().numpy(),threshold=config.threshold).to(device) 				#BE
			pred_prob_test = y_hat_test															#BE

		elif loss_F == 'Tim' :
			y_hat_test, presention_test = model(test_x)
			loss = criterion(y_hat_test, test_y.to(device)).item()
			pred_test = y_hat_test.max(1, keepdim = True)[1]                                #Cross   
			pred_prob_test = y_hat_test[:,1]

		test_loss += loss * len(test_y)
		test_correct += pred_test.eq(test_y.view_as(pred_test)).sum().item()
		
		torch_test = torch.cat([torch_test,pred_prob_test.data.cpu()],dim=0)#收集所有的输出概率/最大值位置0-1
		torch_test_y = torch.cat([torch_test_y,test_y.data.cpu()],dim=0)	#收集所有标签
	
	if config.best_threshold ==True:
		fpr_test, tpr_test, thresholds_new = roc_curve(torch_test_y.data.cpu().numpy(), torch_test.reshape((-1, )))
		gmeans = np.sqrt(tpr_test * (1-fpr_test)) #
		# locate the index of the largest g-mean   #优化阈值
		
		ix = np.argmax(gmeans)
		print('Best Threshold=%f, G-Mean=%.3f' % (thresholds_new[ix], gmeans[ix]))
		best_threshold = thresholds_new[ix]	
		
		config.threshold = best_threshold
		
		pred_test = round_pred(torch_test.data.cpu().numpy(),threshold=best_threshold) #将所有的输出和新阈值进行比较得出新标签
		test_correct = pred_test.eq(y_test.view_as(pred_test)).sum().item() #y_test: 所有test的标签
		
	#----------------------------------------->>>>
	test_losses.append(test_loss/len(X_test)) # all loss / all sample
	accuracy_test = 100.*test_correct/len(X_test)
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
		test_loss/len(X_test), test_correct, len(X_test), accuracy_test))

	test_acc.append(accuracy_test)
	
	#统计10折-test总结果
	test_loss_sum += test_losses[-1] # test_loss of 10 fold
	test_acc_sum += test_acc[-1]     # test_loss of 10 fold 
	
	#统计所有预测结果------------------------------->>>
	print('torch_test:',torch_test.shape[0])
	
	print('test:')
	metrics_test = calculateScore(torch_test_y, torch_test.numpy())
	testing_result.append(metrics_test)
	
	#update dictionary
	metrics_test_index['SN'].append(metrics_test['sn'])
	metrics_test_index['SP'].append(metrics_test['sp'])
	metrics_test_index['ACC'].append(metrics_test['acc'])
	metrics_test_index['MCC'].append(metrics_test['MCC'])
	metrics_test_index['AUC'].append(metrics_test['AUC'])
	metrics_test_index['F1'].append(metrics_test['F1'])
	metrics_test_index['precision'].append(metrics_test['precision'])
	metrics_test_index['recall'].append(metrics_test['recall'])
	
	#!!! 保存所有的标签和预测值
	out_test_file = OutputDir+'/test_result.txt'
	ytest_ypred_to_file(torch_test_y.numpy(), torch_test.numpy(), out_test_file)
	
	auroc = metrics.roc_auc_score(torch_test_y.numpy(), torch_test.numpy())
	test_auc.append(auroc)
	
	#单独检查每个index_fold的结果
	temp_test_dict = ([metrics_test])
	analyze_sigle(temp_test_dict, OutputDir,species,'test')
	