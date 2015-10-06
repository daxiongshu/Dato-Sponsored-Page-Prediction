'''
randomly generate dozens of xgboost models
'''
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.ensemble import ExtraTreesClassifier as ET
from xgboost_c import XGBC
from sklearn.cross_validation import StratifiedKFold as KFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
import random
from scipy import sparse
import pickle


def random_model(seed):
	random.seed(seed*218)
	num_round = random.choice([200,300])
	max_depth = random.choice([8,20,30])
	eta = random.choice([0.05, 0.1])
	min_child_weight = random.choice([2,5])
	colsample_bytree = random.choice([0.5,0.6,0.8])
	gamma = random.choice([0.3,0])
	subsample = random.choice([0.8,1])
	print num_round, max_depth, eta,min_child_weight, colsample_bytree
	model = XGBC(num_round = num_round, max_depth = max_depth, eta = eta, min_child_weight = min_child_weight,
		colsample_bytree = colsample_bytree, gamma = gamma, subsample = subsample, seed = seed)
	
	return model


def xz_features():
	xz_fea = pickle.load(open('xz_tokens_01.p'))
	xz_id = pickle.load(open('xz_tokens_id.p'))
	xz_fea = sparse.csr_matrix(xz_fea)

	return xz_fea, xz_id

def carl_features():
	train_fea = pickle.load(open('Xtrain.p'))
	train_id = pickle.load(open('trainid.p'))
	train_label = pickle.load(open('ytrain.p'))
	test_fea = pickle.load(open('Xtest.p'))
	test_id = pickle.load(open('testid.p'))

	return train_fea, test_fea, train_id, test_id, train_label

def gen_data():
	# read data
	train = pd.read_csv("train_v2.csv")
	test = pd.read_csv("sampleSubmission_v2.csv")
	del test['sponsored']
	xz_feature_v2 = pd.read_csv("xz_tokens_v2.csv")
	xz_feature_v2.columns = ['file'] + ['xz_token_%i'%x for x in xrange(xz_feature_v2.shape[1]-1)]
	train = pd.merge(train, xz_feature_v2, how = 'inner', on = 'file')
	test = pd.merge(test, xz_feature_v2, how = 'inner', on = 'file')
	xz_feature_v3 = pd.read_csv("xz_tokens_v3.csv")
	xz_feature_v3.columns = ['file'] + ['xz_token_3_%i'%x for x in xrange(xz_feature_v3.shape[1]-1)]
	train = pd.merge(train, xz_feature_v3, how = 'inner', on = 'file')
	test = pd.merge(test, xz_feature_v3, how = 'inner', on = 'file')

	xz_fea, xz_id = xz_features() # xz features
	carl_fea_train,carl_fea_test,carl_train_id, carl_test_id, train_label = carl_features() # carl features

	xz_id_df = pd.DataFrame(data = {'file':xz_id,'xz_order':range(len(xz_id))})
	carl_train_id_df = pd.DataFrame(data = {'file':carl_train_id,'carl_order':range(len(carl_train_id))})
	carl_test_id_df = pd.DataFrame(data = {'file':carl_test_id,'carl_order':range(len(carl_test_id))})
	train = pd.merge(train, xz_id_df, how = 'inner', on = 'file')
	test = pd.merge(test, xz_id_df, how = 'inner', on = 'file')
	train = pd.merge(train, carl_train_id_df, how = 'inner', on = 'file')
	test = pd.merge(test, carl_test_id_df, how = 'inner', on = 'file')
	# sort train and test by carl_order
	train.sort('carl_order', inplace = True)
	test.sort('carl_order', inplace = True)
	xz_fea_train = xz_fea[list(train['xz_order']),:]
	xz_fea_test = xz_fea[list(test['xz_order']),:]

	del train['file'],train['sponsored'], train['xz_order']
	del test['file'], test['xz_order']
	train = sparse.csr_matrix(train.as_matrix())
	test = sparse.csr_matrix(test.as_matrix())

	# sparse matrix stack
	X = sparse.hstack((train, carl_fea_train,xz_fea_train),format='csr')
	X_test = sparse.hstack((test, carl_fea_test, xz_fea_test),format='csr')
	
	return carl_train_id, carl_test_id, train_label, X, X_test

def gen_cv_pred(seed_list):
	print "generating files...."
	train_id, test_id, label, X_a, X_test_a = gen_data()
	kf = KFold(label, n_folds = 4)

	for seed in seed_list:
		random.seed(seed*2464)
		# get random percentage of features
		pcnt = random.choice([0.65,0.75,0.9,1])
		fea_pcnt = random.sample(range(X_a.shape[1]), int(pcnt * X_a.shape[1]))
		X = X_a[:,fea_pcnt]
		X_test = X_test_a[:,fea_pcnt]
		# cv file
		clf = random_model(seed)
		cv_train = np.zeros(len(label))
		for i, (train_fold, validate) in enumerate(kf):
			if i == 3:
				X_train, X_validate, label_train, label_validate = X[train_fold,:], X[validate,:], label[train_fold], label[validate]
				clf.fit(X_train,label_train)
				cv_train[validate] = clf.predict_proba(X_validate)[:,1]
				current_auc = AUC(label_validate, cv_train[validate])
				print "finishing one fold with AUC %.3f"%current_auc
		if current_auc > 0.96: # threshold to include the random model
			cv_output = pd.DataFrame({'file':train_id[validate], 'sponsored':label_validate,
				'pred':cv_train[validate]})
			cv_output.to_csv("cv/dato_cv_%.5f.csv"%current_auc, index = False)

			# pred file
			clf.fit(X, label)
			pred = clf.predict_proba(X_test)[:,1]
			submission = pd.DataFrame({'file':test_id, 'sponsored':list(pred)})
			submission.to_csv("pred/dato_pred_%.5f.csv"%current_auc, index = False)

if __name__ == '__main__':
	seed_list = range(60,80)
	gen_cv_pred(seed_list)
	print "ALL DONE!!!"



