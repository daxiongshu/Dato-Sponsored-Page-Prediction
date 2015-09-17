'''
xgboost it using folder 4 as validation
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
		for max_depth in [50]:
			for eta in [0.1]:
				for min_child_weight in [2]:
					for colsample_bytree in [0.7]:
						model_list.append((XGBC(num_round = num_round, max_depth = max_depth, eta = eta, min_child_weight = min_child_weight,
							colsample_bytree = colsample_bytree), 'xgb_tree_%i_depth_%i_lr_%f_child_%i'%(num_round, max_depth, eta, min_child_weight)))
	
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

def carl_features():
	train_fea = pickle.load(open('Xnew_train.p'))
	train_id = pickle.load(open('IDnew_train.p'))
	train_label = pickle.load(open('ynew_train.p'))
	test_fea = pickle.load(open('Xnew_test.p'))
	test_id = pickle.load(open('IDnew_test.p'))

	# get rid of some rows.
	true_train = pd.read_csv("train_v2.csv")
	true_train = set(true_train['file'])
	true_train_index = [i for i in xrange(len(train_id)) if train_id[i] in true_train]
	train_id = train_id[true_train_index]
	train_fea = train_fea[true_train_index,:]
	train_label = train_label[true_train_index]
	print "filter out the non exist rows..."
	print train_fea.shape

	return train_fea, test_fea, train_id, test_id, train_label


def gen_data():
	# read data
	empty_file_list = empty_list()
	train = pd.read_csv("train_v2.csv")
	test = pd.read_csv("sampleSubmission_v2.csv")

	xz_feature = pd.read_csv("xz_tokens.csv")
	xz_feature.columns = ['file'] + ['xz_token_%i'%x for x in xrange(xz_feature.shape[1]-1)]

	train = pd.merge(train, xz_feature, how = 'inner', on = 'file')
	test = pd.merge(test, xz_feature, how = 'inner', on = 'file')
	
	del test['sponsored']
	print train.shape, test.shape

	#train = train[~train['file'].isin(empty_file_list)].reset_index(drop = True)
	# carl features
	carl_fea_train,carl_fea_test,carl_train_id, carl_test_id, train_label = carl_features()
	# carl's order
	carl_train_id_df = pd.DataFrame(data = {'file':carl_train_id,'carl_order':range(len(carl_train_id))})
	carl_test_id_df = pd.DataFrame(data = {'file':carl_test_id,'carl_order':range(len(carl_test_id))})
	train = pd.merge(train, carl_train_id_df, how = 'inner', on = 'file')
	test = pd.merge(test, carl_test_id_df, how = 'inner', on = 'file')
	# sort train and test by carl_order
	train.sort('carl_order', inplace = True)
	test.sort('carl_order', inplace = True)
	del train['file'],train['sponsored'], train['carl_order']
	del test['file'], test['carl_order']
	train = sparse.csr_matrix(train.as_matrix())
	test = sparse.csr_matrix(test.as_matrix())
	# sparse matrix stack
	print train.shape, carl_fea_train.shape
	X = sparse.hstack((train,carl_fea_train),format='csr')
	print test.shape, carl_fea_test.shape
	X_test = sparse.hstack((test, carl_fea_test),format='csr')

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
		#print "the AUC is %.6f"%AUC(label, cv_train)
		#cv_output = pd.DataFrame(np.column_stack((train_id,cv_train, label)), columns=['ID','pred','label'])
		#cv_output.convert_objects(convert_numeric=True).to_csv("validate/%s_cv.csv"%clf_name, index = False)

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
	submission.to_csv("xz_sub0.csv", index = False)

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

