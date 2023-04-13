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
	