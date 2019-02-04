import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

def catboost_reg(df, y_name, test_size=0.25, returned_metric = 'r2', feature_importance_df = True, top_feat_number =  10):
	X_df = df.drop(columns = y_name)
	X_df = prep_for_catboost(X_df)

	y_df = df[y_name]

    xtrain, xtest, ytrain, ytest = train_test_split(X_df, y_df, test_size=test_size, random_state=123)

    #initiate regressor
    model_cat = CatBoostRegressor(random_state=123)
    model_cat.fit(xtrain, ytrain, cat_features = np.where(X.dtypes == 'object')[0], verbose=False)

    if returned_metric == 'r2':
    	if feature_importance == True:
    		return(r2_score(ytest, model_cat.predict(xtest)), 
    			   calculate_feature_importance(model_cat, top_feat_number = top_feat_number))
    	else:
    		return(r2_score(ytest, model_cat.predict(xtest)))

    elif returned_metric == 'adjusted_r2':
    	n = len(xtest)
    	p = xtest.shape[1]
    	r2 = r2_score(ytest, model_cat.predict(xtest))

    	if feature_importance == True:
    		return(calculate_adj_r2(r2, n, p), 
    			   calculate_feature_importance(model_cat, top_feat_number = top_feat_number))

    	else:
    		return(calculate_adj_r2(r2, n, p))
    
    elif returned_metric == 'both':
    	n = len(xtest)
    	p = xtest.shape[1]
    	r2 = r2_score(ytest, model_cat.predict(xtest))
    	adjusted_r2 = calculate_adj_r2(r2, n, p)
    	if feature_importance == True:
    		return(r2, adjusted_r2, calculate_feature_importance(model_cat, top_feat_number = top_feat_number))
    	else:
    		return(r2, adjusted_r2)
    #return("r2: {:.3f}, adjusted r2:{:.3f}.".format(r2, adj_r2))

def prep_for_catboost(df):
    categorical_features = np.where(df.dtypes == 'object')[0]
    for i in categorical_features:
        df_X.iloc[:,i] = df_X.iloc[:,i].astype('str')
    return(df)

def calculate_adj_r2(r2, n, p):
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return(adj_r2)    

def calculate_feature_importance(model, top_feat_number = 10):
	top_feats_df = pd.DataFrame({
		'feature_names': model.feature_names_,
		'feature_importance': model.feature_importance_
		}).sort_values(by = 'feature_importance', ascending= False).head(top_feat_number)

	return(top_feats_df)

def export_feature_importance_plot(feat_importance_df, fig_size_x = 15, fig_size_y = 10, font_size = 1.2):
	plt.figure(figsize=(fig_size_x, fig_size_y))
	sns.set(font_scale=font_size)

	importance_plot = sns.barplot(x=feat_importance_df['feature_names'], y =feat_importance_df['feature_importance'])
	importance_plot.xticks(rotation=90)
	importance_plot.savefig("importance_plot.png")




