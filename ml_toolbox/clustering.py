import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import DataGenerator, KneeLocator

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing(X, ohe=True, cat_features='auto'):
    if (ohe==True) & (cat_features == 'auto'):
        cat_features_cols = X.select_dtypes('object').columns
        if len(cat_features_cols) == 0:
            print("Data does not contain categorical features to preprocess with.")
        else:
            X_cat = X[cat_features_cols]
            X_cat_df_ohe = ohe_preprocessing(X_cat)
            return(X_cat_df_ohe)

    elif (ohe==True) & (type(cat_features)==list):
        cat_features_cols_input = cat_features
        X_cat = X[cat_features_cols_input]
        X_cat_df_ohe = ohe_preprocessing(X_cat)

    elif ohe==False:
    	next
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X_num = X.select_dtypes(include=numerics)
    X_num_scaled = num_preprocessing(X_num, col_names= X_num.columns)

    output_df = pd.merge(X_num_scaled, X_cat_df_ohe, left_index=True, right_index=True)
    return(output_df)
    #return(X_num_scaled, X_cat_df_ohe)

def num_preprocessing(X_num_df, col_names, remove_high_zeros=False, remove_high_vif=False):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=col_names)
    return(X_df_scaled)

def ohe_preprocessing(X_cat_df):
    X_cat_df_ohe = pd.get_dummies(X_cat_df)
    return(X_cat_df_ohe)

def calculate_dbscan_eps(X_preprocessed, n_neighbours = 100):
	nbrs = NearestNeighbors(n_neighbors= n_neighbours, algorithm='ball_tree').fit(X_preprocessed)
	distances, indices = nbrs.kneighbors(X_preprocessed)
	distance_df = pd.DataFrame(distances)

	avg_dist_df = pd.DataFrame(distance_df.apply(lambda x: x.mean())).reset_index()
	avg_dist_df.columns = ['clst', 'avg_dist']

	kneedle = KneeLocator(avg_dist_df['clst'], avg_dist_df['avg_dist'], S=1.0, curve='concave', direction='increasing')
	best_clst = kneedle.knee
	eps_value = avg_dist_df[avg_dist_df['clst'] == best_clst]['avg_dist'].values[0]
	return(eps_value)

def cluster_dbscan(X_preprocessed, dbscan_unique_clst = 10, dbscan_init_clst_num = 2 , dbscan_n_iter = 20):
	iteration = 0
	dbscan_perf_df = pd.DataFrame(columns = ['unique_clst', 'avg_silhouette_score', 'n_noise'])
	optimal_eps = calculate_dbscan_eps(X_preprocessed)

	while dbscan_unique_clst >=3 | iteration < dbscan_n_iter:
		dbscan_tmp = DBSCAN(eps = optimal_eps, min_samples= dbscan_init_clst_num).fit(X_preprocessed)
		labels = dbscan_tmp.labels_
		eval_df = X_preprocessed.copy()
		eval_df['label'] = labels
		n_noise = sum(eval_df['label'] == -1)
		eval_df[eval_df['label']!=-1].reset_index(drop=True, inplace=True)

		unique_clst = eval_df['label'].nunique()
		avg_silhouette_score = metrics.silhouette_score(eval_df.drop(columns='label'), labels)

		dbscan_perf_df = dbscan_perf_df.append(pd.DataFrame({'unique_clst': [unique_clst],
														     'avg_silhouette_score': [avg_silhouette_score],
														     'n_noise': [n_noise]
															}))

		dbscan_init_clst_num += 1
		iteration +=1

	dbscan_perf_df.reset_index(drop=True, inplace=True)
	best_clst_num = dbscan_perf_df.loc[dbscan_perf_df['avg_silhouette_score'].argmax()]['unique_clst']
	best_score = round(dbscan_perf_df.avg_silhouette_score.max(), 5)

	dbscan_opt = DBSCAN(eps = optimal_eps, min_samples = best_clst_num).fit(X_preprocessed)
	output_df = X_preprocessed.copy()
	output_df['label'] = dbscan_opt.labels_

	return(output_df, best_score)

def clustering(X_preprocessed, method='kmeans', on_pca=False, pick_best=False,
               cluster_min=1, cluster_max=10, step=1,
               dbscan_unique_clst = 10, dbscan_init_clst_num =2, dbscan_n_iter = 20):

    if method == 'kmeans':
        if pick_best == False:
            elbow_df = pd.DataFrame(columns=['clst', 'sse'])
            for clst in np.arange(cluster_min, cluster_max, step):
                print('Fitting KMeans on cluster {}'.format(clst))
                kmeans = KMeans(n_clusters=clst, random_state=123).fit(X_preprocessed)
                sse = kmeans.inertia_
                elbow_df = elbow_df.append(pd.DataFrame({'clst': [clst], 'sse':[sse]}))
            elbow_plot = sns.pointplot(x='clst', y='sse', data=elbow_df)
            return(elbow_plot)
        
        elif pick_best==True:
            silhouette_df = pd.DataFrame(columns=['clst', 'avg_silhouette_score'])
            for clst in np.arange(cluster_min+1, cluster_max, step):
                kmeans = KMeans(n_clusters=clst, random_state=123).fit(X_preprocessed)
                clst_pred = kmeans.predict(X_preprocessed)
                score = metrics.silhouette_score(X_preprocessed.values, clst_pred)
                silhouette_df = silhouette_df.append(pd.DataFrame({'clst': [clst], 'avg_silhouette_score': [score]}))
                
            silhouette_df.reset_index(drop=True, inplace=True)
            best_score = silhouette_df['avg_silhouette_score'].max()
            best_clst = silhouette_df.loc[silhouette_df['avg_silhouette_score'].argmax()]['clst']
            
            kmeans = KMeans(n_clusters=best_clst, random_state=123).fit(X_preprocessed)
            clst_pred = kmeans.predict(X_preprocessed)

            output_df = X_preprocessed.copy()
            output_df['clst'] = clst_pred
            
            print("The best average Silhouette score was {:.3f} at cluster {}.".format(best_score, best_clst))
            return(output_df)
            
        elif type(pick_best)==int:
            kmeans = KMeans(n_clusters=pick_best, random_state=123).fit(X_preprocessed)
            clst_pred = kmeans.predict(X_preprocessed)

            output_df = X_preprocessed.copy()
            output_df['clst'] = clst_pred

            return(output_df)
        
    elif method == 'dbscan':
    	#determine eps using knn and min_sample using silhouette score
    	score, output_df = cluster_dbscan(X_preprocessed, dbscan_unique_clst=dbscan_unique_clst, 
    									  dbscan_init_clst_num=dbscan_init_clst_num , dbscan_n_iter = dbscan_n_iter)
    	return(score, output_df)
    #elif method == 'gaussian_mixture':
