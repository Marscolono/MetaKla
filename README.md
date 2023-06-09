# MetaKla


MetaKla: a novel deep learning framework for accurate prediction of lysine lactylation sites based on learning embedding features and transductive information maximization loss

Protein lysine lactylation (Kla) is a novel type of protein post-translational modification (PTM) that is found in fungi and mammalian cells.This modification can be stimulated by exogenous or glucose-derived endogenous lactate (L-lactate) and is involved in the regulation of gene transcription and glycolytic flux. The identification of Kla sites is critical to better understanding their functional mechanisms. However, the existing experimental techniques for detecting Kla sites are cost-ineffective, to a great need for new computational methods to address this problem. We here describe MetaKla, an advanced deep learning model that utilizes adaptive embedding and is based on a convolutional neural network together with ransductive information maximization loss. On the independent testing set, MetaKla outperformed the current state-of-the-art Kla prediction model. Compared to other Kla models, MetaKla additionally had a more robust ability to distinguish between lactylation  and other lysine modifications. These results indicate that self-adaptive embedding features perform better than handcrafted features in capturing discriminative information.

![MetaKla](https://user-images.githubusercontent.com/30385448/231618500-48c7ec49-999b-4a4d-bb49-86af1f8bfc2a.png)

Usage：

# train
python MetaKla_train.py -k-fold -1 -train-fasta './Kla_data/Kla_data/upTrain.fa' 

python MetaKla_train.py -k-fold 5 -train-fasta './Kla_data/Kla_data/upTrain.fa' 

# test
python MetaKla_test.py -test-fasta './Kla_data/Kla_data/fungiForTest.fa' 

python MetaKla_test.py -test-fasta './Kla_data/Kla_data/Kla_human.txt' 
