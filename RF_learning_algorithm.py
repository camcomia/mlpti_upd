#########################################################################
##                 Copyright (C). All Rights Reserved.                   ##
##      "Harnessing machine learning to guide                            ##
##                              phylogenetic-tree search algorithms"     ##
##                                                                       ##
## by Dana Azouri, Shiran Abadi, Yishay Mansour, Itay Mayrose, Tal Pupko ##
##                                                                       ##
##                                                                       ##
##   For information, contact danaazouri@mail.tau.ac.il                  ##
##                                                                       ##
## For academic, non-commercial use.                                     ##
## If you use the code, please cite the paper                            ##
##                                                                       ##
#########################################################################
import math
from defs_PhyAI import *
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.svm import SVR
from statistics import mean, median, StatisticsError
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
    

##### helper functions #####

def log_experiment(model_name, params, metrics):
    """
    Log the parameters and results of the learning experiment.
    :param model_name: The name of the model used.
    :param params: Dictionary of model hyperparameters.
    :param metrics: Dictionary of evaluation metrics.
    """
    print(f"Model: {model_name}")
    print(f"Parameters: {params}")
    print(f"Metrics: {metrics}")
    print("------------------------------------")
	
def get_newick_tree(tree):
	"""
	:param tree: newick tree string or txt file containing one tree
	:return:	tree: a string of the tree in ete3.Tree format
	"""
	if os.path.exists(tree):
		with open(tree, 'r') as tree_fpr:
			tree = tree_fpr.read().strip()
	return tree

def get_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: list of branch lengths
	"""
	try:
		if type(tree) == str:
			tree = Tree(get_newick_tree(tree), format=1)
		tree_root = tree.get_tree_root()
	except:
		print(tree)
	if len(tree) == 1 and not "(" in tree:    # in one-branch trees, sometimes the newick string is without "(" and ")" so the .iter_decendants returns None
		return [tree.dist]
	branches = []
	for node in tree_root.iter_descendants(): # the root dist is 1.0, we don't want it
		branches.append(node.dist)

	return branches


def get_total_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: total branch lengths
	"""
	branches = get_branch_lengths(tree)
	return sum(branches)
############################

def score_rank(df_by_ds, sortby, locatein, random, scale_score):
	'''
	find the best tree in 'sortby' (e.g., predicted as the best) foreach dataset and locate its rank in 'locatein' (e.g., y_test)
	'''

	best_pred_ix = df_by_ds[sortby].idxmax()    # changed min to max!
	if random:
		best_pred_ix = np.random.choice(df_by_ds[sortby].index, 1, replace=False)[0]
	temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()   # changed ascending to False
	best_pred_rank = min(temp_df.index[temp_df["index"] == best_pred_ix].tolist())
	best_pred_rank += 1  # convert from pythonic index to position
	
	if scale_score:
		best_pred_rank /= len(df_by_ds[sortby].index)   # scale the rank according to rankmax
		best_pred_rank *= 100


	return best_pred_rank

def ds_scores(df, move_type, random, scale_score):
	"""
	Calculate the rank of predicted and actual scores for each dataset and compute Spearman correlation.

	:param df: DataFrame containing the data.
	:param move_type: Type of move (e.g., 'prune', 'rgft').
	:param random: Boolean indicating whether to use random selection.
	:param scale_score: Boolean indicating whether to scale the scores.
	:return: Tuple containing dictionaries of predicted and actual ranks, and a list of Spearman correlations.
	"""
	rank_pred_by_ds, rank_test_by_ds = {}, {}

	label = LABEL.format(move_type)
	sp_corrs = []
	grouped_df_by_ds = df.groupby(FEATURES[GROUP_ID], sort=False)
	for group_id, df_by_ds in grouped_df_by_ds:
		rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", label, random, scale_score)
		rank_test_by_ds[group_id] = score_rank(df_by_ds, label, "pred", random, scale_score)
		
		temp_df = df_by_ds[[label, "pred"]]
		
		 # Check for constant values or missing values
		# if temp_df["pred"].nunique() == 1:
		# 	print(f"Constant values detected in group {group_id}")
		# if temp_df[label].isnull().any() or temp_df["pred"].isnull().any():
		# 	print(f"Missing values detected in group {group_id}")

		sp_corr = temp_df.corr(method='spearman').iloc[1,0]
		if sp_corr:
			sp_corrs.append(sp_corr)
		else:
			sp_corrs.append(None)
	
	return rank_pred_by_ds, rank_test_by_ds, sp_corrs

def split_features_label(df, move_type, features):
	"""
	Splits the DataFrame into features and label arrays.

	:param df: DataFrame containing the data.
	:param move_type: Type of move (e.g., 'prune', 'rgft').
	:param features: List of feature column names.
	:return: Tuple containing feature array (x) and label array (y).
	"""
	attributes_df = df[features].reset_index(drop=True)
	label_df = df[LABEL.format(move_type)].reset_index(drop=True)

	x = np.array(attributes_df)
	y = np.array(label_df).ravel()

	return x, y


def apply_RFR(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	params = {
		"n_estimators": 70,
        "max_features": 0.33, # 0.33=nfeatures/3. this is like in R (instead of default=n_features)
		"oob_score": True,
		"n_jobs":-1
    }
	start_time = time.time()
	regressor = RandomForestRegressor(**params).fit(X_train, y_train) 
	y_pred = regressor.predict(X_test)
	oob = regressor.oob_score_
	f_imp = regressor.feature_importances_
	training_time = time.time() - start_time

	metrics = {
		"Train MSE": mean_squared_error(y_train, regressor.predict(X_train)),
		"Test MSE": mean_squared_error(y_test, y_pred),
		"Train RMSE": np.sqrt(mean_squared_error(y_train, regressor.predict(X_train))),
		"Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
		"R^2 Score": r2_score(y_test, y_pred),
		"Training Time (s)": training_time
    }

	log_experiment("Random Forest", params, metrics)

	return y_pred, oob, f_imp

def apply_KNN(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	# Scale features (important for KNN)
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	n_neighbors = max(5, int(0.15 * len(X_train_scaled)))  # Ensure at least 5 neighbor
	print(f"Number of neighbors: {n_neighbors}")

	params = {
		"n_neighbors": 5,
#		"weights": "uniform",
#		"algorithm": "auto",
#		"leaf_size": 30,
#		"p": 2
	}

	# KNN with k=5
	start_time = time.time()
	regressor = KNeighborsRegressor(**params).fit(X_train_scaled, y_train)
	y_pred = regressor.predict(X_test_scaled)
	training_time = time.time() - start_time

	# KNN doesn't have OOB score or feature importance
	oob = None  
	f_imp = None

	metrics = {
		"Test MSE": mean_squared_error(y_test, y_pred),
		"Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
		"R^2 Score": r2_score(y_test, y_pred),
		"Training Time (s)": training_time
	}

	log_experiment("K Nearest Neighbors", params, metrics)

	return y_pred, oob, f_imp

def apply_SVM(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	# Scale features (important for SVM)
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# SVR with RBF kernel (can be changed to linear or poly if needed)
	params = {
		"kernel": 'rbf', 
		"C": 0.1, 
		"epsilon": 5e-6, # epsilon=5e-6 is the value from Azouri study
#		"gamma": 'scale' # Automatically scales based on n_features * X.var()
	}
    
	start_time = time.time()
	regressor = SVR(**params).fit(X_train_scaled, y_train)
	y_pred = regressor.predict(X_test_scaled)
	training_time = time.time() - start_time
    
	metrics = {
        "Test MSE": mean_squared_error(y_test, y_pred),
		"Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R^2 Score": r2_score(y_test, y_pred),
        "Training Time (s)": training_time
    }
    
	log_experiment("Support Vector Machine", params, metrics)

	# SVR doesn't have OOB score or feature importance
	oob = None  

	# Approximate feature importance through permutation importance
	result = permutation_importance(regressor, X_test_scaled, y_test, n_repeats=10, random_state=42)
	f_imp = result.importances_mean

	return y_pred, oob, f_imp

def apply_BayesianRidge(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	# Scale features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	params = {
		# "n_iter": 300,
        #"tol": 1e-3,
        #"alpha_1": 1e-6,
        #"alpha_2": 1e-6,
        #"lambda_1": 1e-6,
        #"lambda_2": 1e-6,
        #"compute_score": False
	}
    
	start_time = time.time()
	regressor = BayesianRidge(**params).fit(X_train_scaled, y_train)
	y_pred = regressor.predict(X_test_scaled)
	training_time = time.time() - start_time

	metrics = {
		"Test MSE": mean_squared_error(y_test, y_pred),
		"Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
		"R^2 Score": r2_score(y_test, y_pred),
		"Training Time (s)": training_time
	}

	log_experiment("Bayesian Ridge", params, metrics)

	# Bayesian Ridge doesn't have OOB score, but has a score history. Not sure if it's useful
	oob = None

	# Feature importance is based on coefficient values
	f_imp = np.abs(regressor.coef_)

	return y_pred, oob, f_imp

def apply_Lasso(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	# Scale features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	params = {
#		"alpha": 1e-4,
#		"max_iter": 5000,
#		"tol": 1e-6,
#		"selection": "random"
    }
    
	start_time = time.time()
	regressor = Lasso(**params).fit(X_train_scaled, y_train)
	y_pred = regressor.predict(X_test_scaled)
	training_time = time.time() - start_time

	metrics = {
		"Test MSE": mean_squared_error(y_test, y_pred),
		"Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
		"R^2 Score": r2_score(y_test, y_pred),
		"Training Time (s)": training_time
	}
    
	log_experiment("Lasso Regression", params, metrics)
	# Lasso doesn't have OOB score
	oob = None

	# Feature importance is based on coefficient values
	f_imp = np.abs(regressor.coef_)
	return y_pred, oob, f_imp


def truncate(df):
	df = df.dropna()
	groups_ids = df[FEATURES[GROUP_ID]].unique()
    
	print(f"Initial number of datasets: {len(groups_ids)}")  # Debug print

	kfold = len(groups_ids) if KFOLD=="LOO" else KFOLD
	assert len(groups_ids) >= kfold
	ndel = len(groups_ids) % kfold
	if ndel != 0:   # i removed datasets from the end, and not randomly. from some reason..
		for group_id in groups_ids[:-ndel-1:-1]:
			df = df[df[FEATURES[GROUP_ID]] != group_id]

	groups_ids = df[FEATURES[GROUP_ID]].unique()
	new_length = len(groups_ids) - ndel
	test_batch_size = int(new_length / kfold)

	print(f"Number of datasets after truncation: {len(groups_ids)}")  # Debug print

	return df.reset_index(drop=True), groups_ids, test_batch_size


def cross_validation_RF(df, move_type, features, model, trans=False, validation_set=False, random=False, scale_score=False):
	df, groups_ids, test_batch_size = truncate(df)
	res_dict = {}
	oobs, f_imps, = [], []
	my_y_pred, imps = np.full(len(df), np.nan), np.full(len(df), np.nan)
	
	if not validation_set:
		print(f"test_batch_size: {test_batch_size}")
		for low_i in groups_ids[::test_batch_size]:
			low_i, = np.where(groups_ids == low_i)
			low_i = int(low_i[0])
			up_i = low_i + test_batch_size
	
			test_ixs = groups_ids[low_i:up_i]
			train_ixs = np.setdiff1d(groups_ids, test_ixs)
			df_test = df.loc[df[FEATURES[GROUP_ID]].isin(test_ixs)]
			df_train = df.loc[df[FEATURES[GROUP_ID]].isin(train_ixs)]
	
			if model == 'RFR':
				y_pred, oob, f_imp = apply_RFR(df_test, df_train, move_type, features)
			elif model == 'KNN':
				y_pred, oob, f_imp = apply_KNN(df_test, df_train, move_type, features)
			elif model == 'SVM':
				y_pred, oob, f_imp = apply_SVM(df_test, df_train, move_type, features)
			elif model == 'BR':
				y_pred, oob, f_imp = apply_BayesianRidge(df_test, df_train, move_type, features)
			elif model == 'Lasso':
				y_pred, oob, f_imp = apply_Lasso(df_test, df_train, move_type, features)
			else:
				raise ValueError(f"Unknown model: {model}")
	
			oobs.append(oob)
			f_imps.append(f_imp)
			my_y_pred[df_test.index.values] = y_pred       # sort the predictions into a vector sorted according to the respective dataset
			
		df["pred"] = my_y_pred
	
	else:     # namely if validation set strategy, and not cross validation
		# Camille: VALSET_FEATURES_LABEL is not defined in original code. I'm assuming that validation set strat was never used.
		df_train = df
		df_test = pd.read_csv(VALSET_FEATURES_LABEL)
		df_test = fit_transformation(df_test, move_type, trans).dropna()
		y_pred, oob, f_imp = apply_RFR(df_test, df_train, move_type, features)
		
		oobs.append(oob)
		f_imps.append(f_imp)
		df_test["pred"] = y_pred  # the predictions vec is the same lengths of test set
		df = df_test
	
	rank_pred_by_ds, rank_test_by_ds, corrs = ds_scores(df, move_type, random, scale_score)

	# averaged over cv iterations
	
	res_dict['oob'] = sum(oobs) / len(oobs) if oob is not None else None
	res_dict['f_importance'] = sum(f_imps) / len(f_imps) if f_imp is not None else None
	# foreach dataset (namely arrays are of lengths len(sampled_datasets)
	res_dict["rank_first_pred"] = rank_pred_by_ds
	res_dict["rank_first_true"] = rank_test_by_ds
	res_dict["spearman_corr"] = corrs
	
	return res_dict, df


def fit_transformation(df, move_type, trans=False):
	groups_ids = df[FEATURES[GROUP_ID]].unique()
	for group_id in groups_ids:
		scaling_factor = df[df[FEATURES[GROUP_ID]] == group_id]["orig_ds_ll"].iloc[0]
		df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)] /= -scaling_factor    # todo: make sure I run it with minus/abs to preserve order. also change 'ascending' to True in 'get_cumsun_preds' function
	
	if trans:
		df[LABEL.format(move_type)] = np.exp2(df[LABEL.format(move_type)]+1)
	
	return df


def parse_relevant_summaries_for_learning(datapath, outpath, move_type, step_number, tree_type='bionj'):
	for i,relpath in enumerate(os.listdir(datapath)):
		if i ==0:
			ds_path_init = datapath+relpath+"/"
			cols = list(pd.read_csv(SUMMARY_PER_DS.format(ds_path_init, move_type, OPT_TYPE, step_number)))
			cols.insert(1, "path")
			cols.extend([FEATURES[GROUP_ID], FEATURES["group_tbl"]])
			df = pd.DataFrame(index=np.arange(0), columns=cols)
			
		ds_path = datapath + relpath + "/"
		ds_tbl = get_total_branch_lengths(ds_path + PHYML_TREE_FILENAME.format(tree_type))
		summary_per_ds = SUMMARY_PER_DS.format(ds_path, move_type, OPT_TYPE, step_number)
		# print(summary_per_ds)
		
		if os.path.exists(summary_per_ds) and FEATURES["bl"] in pd.read_csv(summary_per_ds).columns:
			df_ds = pd.read_csv(summary_per_ds)
			df_ds.insert(1, "path", ds_path)
			df_ds[FEATURES[GROUP_ID]] = str(i)
			df_ds[FEATURES["group_tbl"]] = ds_tbl
			df = pd.concat([df, df_ds], ignore_index=True)
	
	df.to_csv(outpath)
	
	
def print_and_index_results(df_datasets, res_dict, features):
	
	#### score 1 ####
	spearman_corrs = res_dict['spearman_corr']
	df_datasets['corr'] = spearman_corrs
	try:
		mean_corr = mean([e for e in spearman_corrs if not math.isnan(e)])
	except StatisticsError:
		mean_corr = float('nan')
	try:
		median_corr = median(spearman_corrs)
	except StatisticsError:
		median_corr = float('nan')
	print("\nsapearman corr:\n" + "mean:", mean_corr, ", median:", median_corr)
	
	#### score 2 + 3 ####
	res_vec1 = np.asarray(list(res_dict['rank_first_pred'].values())) if type(res_dict['rank_first_pred']) is dict else res_dict['rank_first_pred']
	res_vec2 = np.asarray(list(res_dict['rank_first_true'].values()))  if type(res_dict['rank_first_true']) is dict else res_dict['rank_first_true']
	df_datasets['best_predicted_ranking'] = res_vec1
	df_datasets['best_empirically_ranking'] = res_vec2
	print("\nbest predicted rank in true:\n" + "mean:",np.mean(res_vec1), ", median:", np.median(res_vec1))
	print("\nbest true rank in pred :\n" + "mean:",np.mean(res_vec2), ", median:", np.median(res_vec2))
	
	mean_importances = res_dict['f_importance']   # index in first row only (score foreach run and not foreach dataset)
	for i, f in enumerate(features):
		colname = "imp_" + f
		df_datasets.loc[0, colname] = mean_importances[i] if mean_importances is not None else None

	#### additional information ####
	df_datasets.loc[0, 'oob'] = res_dict['oob']   # index in first row only (score foreach run and not foreach dataset)
	print("oob:", res_dict['oob'])
	print("ndatasets: ", len(res_vec1))

	return df_datasets


def sort_features(res_dict, features):
	feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(features, res_dict['f_importance'])]
	feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)  # most important first
	sorted_importances = [importance[1] for importance in feature_importances]
	sorted_features = [importance[0] for importance in feature_importances]
	
	return sorted_importances, sorted_features
	
	
def extract_scores_dict(res_dict, df_with_scores):
	res_dict['rank_first_pred'], res_dict["rank_first_true"] = df_with_scores['best_predicted_ranking'].values, df_with_scores['best_empirically_ranking'].values
	res_dict['spearman_corr'], res_dict['%neighbors'], res_dict['oob'] = df_with_scores['corr'].values, df_with_scores['required_evaluations_0.95'].values, df_with_scores.loc[0, 'oob']
	res_dict['f_importance'] = df_with_scores.loc[0, df_with_scores.columns[pd.Series(df_with_scores.columns).str.startswith('imp_')]].values

	return res_dict



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='arrange data for learning and implement learning algo')
	parser.add_argument('--validation_set', '-val', default=False, action='store_true') # whether to use validation set INSTEAD of cross validation
	parser.add_argument('--transform_target', '-trans', default=False, action='store_true')
	parser.add_argument('--model', '-m', type=str, default='RFR', choices=['RFR', 'KNN', 'SVM', 'BR', 'Lasso'], help='Specify the learning model to use')
	parser.add_argument('--project_path', '-p', type=str, default=PROJECT_PATH, help='Specify the project path')
	args = parser.parse_args()

	move_type, st = "merged", "1"

	start_time = time.time()
	
	df_path = args.project_path + "/" + LEARNING_DATA.format("all_moves", st)
	df_prune_features = args.project_path +  "/" + LEARNING_DATA.format("all_moves_prune", st)
	df_rgft_features = args.project_path +  "/" + LEARNING_DATA.format("all_moves_rgft", st)
	datapath = args.project_path + "/training_data/"

	if not os.path.exists(df_path):
		parse_relevant_summaries_for_learning(datapath, df_prune_features, "prune", st,)
		parse_relevant_summaries_for_learning(datapath, df_rgft_features, "rgft", st)
		shared_cols = FEATURES_SHARED + ["path","prune_name","rgft_name","orig_ds_ll", "ll"]
		complete_df = pd.read_csv(df_prune_features, dtype=types_dict).merge(pd.read_csv(df_rgft_features, dtype=types_dict),on=shared_cols, suffixes=('_prune', '_rgft'))
		complete_df = complete_df.rename(columns={FEATURES[f]: FEATURES[f] + "_rgft" for f in FEATURES_RGFT_ONLY})
		complete_df[LABEL.format(move_type)] = complete_df[LABEL.format("prune")]
		complete_df.to_csv(df_path)

	df_learning = pd.read_csv(df_path, dtype=types_dict)
	df_learning = fit_transformation(df_learning, move_type, trans=args.transform_target)
	
	features = FEATURES_PRUNE if move_type == "prune" else FEATURES_RGFT if move_type == "rgft" else FEATURES_MERGED
	features.remove(FEATURES[GROUP_ID])
	
	########################
	
	suf = "_{}_validation_set".format(st) if args.validation_set else "_{}".format(st)
	iftrans = "" if not args.transform_target else "_ytransformed"
	suf += iftrans + "_" + args.model
	csv_with_scores = args.project_path + SCORES_PER_DS.format(str(len(features))+ suf)
	csv_with_preds = args.project_path + DATA_WITH_PREDS.format(str(len(features)) + suf)
	if not os.path.exists(csv_with_scores) or args.validation_set:
		print("*@*@*@* scores for step{} with {} features are not available, thus applying learning".format(suf, len(features)))
		res_dict, df_out = cross_validation_RF(df_learning, move_type, features, model=args.model, trans=args.transform_target, validation_set=args.validation_set, random=False, scale_score=True)
		df_out.to_csv(csv_with_preds)

		df_datasets =  pd.DataFrame(columns=["init"])
	else:
		df_datasets = pd.read_csv(csv_with_scores)
		res_dict = extract_scores_dict({}, df_datasets)
	df_datasets = print_and_index_results(df_datasets, res_dict, features)
	df_datasets.to_csv(csv_with_scores)

	print("total time: ", time.time()-start_time)
