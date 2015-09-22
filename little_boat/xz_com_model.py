'''
xgboost it using fold 4 as validation
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

def cv_model_list():
	model_list = []
	for num_round in [200]:
		for max_depth in [20]:
			for eta in [0.1]:
				for min_child_weight in [2]:
					for colsample_bytree in [0.6]:
						for subsample in [1]:
							model_list.append((XGBC(num_round = num_round, max_depth = max_depth, eta = eta, min_child_weight = min_child_weight,
								colsample_bytree = colsample_bytree, subsample = subsample), 
							'xgb_tree_%i_depth_%i_lr_%f_child_%i_colsam_%f_subsam_%f'%(num_round, max_depth, eta, min_child_weight, 
								colsample_bytree, subsample)))
	
	return model_list

def best_model_list():
	#model = (RF(n_estimators=100, n_jobs=-1, max_features = None), 'rf_none')
	model = cv_model_list()
	model = model[0]
	return model

def empty_list():
	empty_file_list = list()
	with open('empty_file_list.txt') as f:
		for x in f:
			empty_file_list.append(x.strip())
	return empty_file_list

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
	'''
	# get rid of some rows.
	true_train = pd.read_csv("train_v2.csv")
	true_train = set(true_train['file'])
	true_train_index = [i for i in xrange(len(train_id)) if train_id[i] in true_train]
	train_id = train_id[true_train_index]
	train_fea = train_fea[true_train_index,:]
	train_label = train_label[true_train_index]
	print "filter out the non exist rows..."
	'''

	return train_fea, test_fea, train_id, test_id, train_label

def gen_data():
	# read data
	#empty_file_list = empty_list()
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
	print "original train, test shapes..."
	print train.shape, test.shape

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
	print "after cleaning, train and test shape..."
	print train.shape, test.shape

	print "obtained xz features..."
	print xz_fea_train.shape, xz_fea_test.shape

	# sparse matrix stack
	print carl_fea_train.shape, xz_fea_train.shape
	X = sparse.hstack((train, carl_fea_train,xz_fea_train),format='csr')
	print carl_fea_test.shape, xz_fea_test.shape
	X_test = sparse.hstack((test, carl_fea_test, xz_fea_test),format='csr')
	print "the final shape..."
	print X.shape, X_test.shape
	
	return carl_train_id, carl_test_id, train_label, X, X_test


def cv_model(model_list):
	print "generating cv csv files...."
	#train_id, test_id, label,X = gen_data()
	train_id, test_id, label, X, X_test = gen_data()
	
	kf = KFold(label, n_folds = 4)
	for j, (clf, clf_name) in enumerate(model_list):
		print "modelling %s...."%clf_name
		cv_train = np.zeros(len(label))
		for i, (train_fold, validate) in enumerate(kf):
			if i == 3:
				X_train, X_validate, label_train, label_validate = X[train_fold,:], X[validate,:], label[train_fold], label[validate]
				clf.fit(X_train,label_train)
				cv_train[validate] = clf.predict_proba(X_validate)[:,1]
				print "finishing one fold with AUC %.6f"%AUC(label_validate, cv_train[validate])
				cv_output = pd.DataFrame({'file':train_id[validate], 'sponsored':label_validate,
				'pred':cv_train[validate]})
				cv_output.to_csv("xz_com_cv.csv", index = False)
def final_result(model):
	clf, clf_name = model
	print "generating full model result csv files...."
	print "modelling %s..."%clf_name
	train_id, test_id, label, X, X_test = gen_data()

	clf.fit(X, label)
	pred = clf.predict_proba(X_test)[:,1]
	submission = pd.DataFrame({'file': test_id,'sponsored': list(pred)})
	sampleSubmission = pd.read_csv("sampleSubmission_v2.csv")
	not_in = sampleSubmission[~sampleSubmission['file'].isin(submission['file'])]
	print "files not being predicted..."
	print not_in.shape
	submission = pd.concat([submission,not_in],ignore_index = True)
	submission.to_csv("dato_pred_0.7999.csv", index = False)

def model_data(cv = False):
	if cv:
		cv_model(cv_model_list())
		print "cross validating is done..."
	else:
		final_result(best_model_list())
		print "predicted csv file is ready..."

if __name__ == '__main__':
	#model_data(cv=True)
	model_data(cv=False)

