"""
Functions relate to performing dimension reduction via PCA.

Author: Xi Liang
Email: xiliang0815@gmail.com
Version: 1.0

Dependencies
-------------
pandas
statsmodels
sklearn
matplotlib
seaborn
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def filter_high_zeros_count(X, threshold = 0.8):
    """
    Use this function if the user wishes to filter out features with high composition of zeros(user can provide a specific threshold).

    Parameters
    ----------
    X: a panda dataframe with all numeric features that the user wishes to perform PCA on.

    treshold: Positive num, optional(default:0.8) 
        If a column contains zeros with more than specified zero percentage (e.g. more than 80% of observations in the column is zero), will be filtered.

    Returns
    -------
    X[filtered_col_names]: panda dataframe

    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #make sure the the input dataframe is all numeric
    if X.select_dtypes(include=numerics).shape[1] == X.shape[1]:
        high_zero_count_df = pd.DataFrame(X.apply(lambda x: sum(x==0)/len(X))).reset_index()
        high_zero_count_df.columns = ['feat_names', 'pct_zeros']
        filtered_col_names = high_zero_count_df[high_zero_count_df['pct_zeros']<=threshold]['feat_names'].values
        return(X[filtered_col_names])
    else:
        print("Input dataframe is not all numerics. Process failed.")

def calculate_vif(X, threshold=5.0):
    """
    Use this function to find factors(features) that have high correlation to each other, as the result to remove multicollinearity in the dataset.

    Parameters
    ----------
    X: a panda dataframe returned by function filter_high_zeros_count.
    threshold: Postive int, optional(default: 5) 
        Threshold to remove features.

    Returns:
    X.iloc[:, variables]: a panda dataframe.
    """
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > threshold:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def PCA_w_scaler(X, threshold_by_component = True, n_components = 10, variance_sum_threshold=0.8):
    """
    Use this function to perform PCA.

    Parameters
    ----------
    X: a panda dataframe returned by function calculate_vif.

    threshold_by_component: Boolean, optional (default: True) 
        Use this variable to specific if the user wants the PCA functions to return a specific count of PC components,
        or returns PC componets that meet a specific sum of variance.

    n_componets: positive int, optional (default: 10) 
        This varibale would only be valid when user specifies True for threshold_by_component.

    variance_sum_threshold: postive num, optional(default: 0.8) 
        This variable would only be valid when user specifies False for threshold_by_component.

    Returns
    --------
    A panda dataframe with PC components.
    """
    #initiate scaler and apply scaler on input X df
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #initate PCA process by meeting variance threshold
    if threshold_by_component==False:
        pca = PCA(n_components = X_scaled.shape[1])
        pca.fit(X_scaled)
        variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        #find the number of component needed to reach the specified threshold
        num_comp_to_reach_variance_threshold = np.where(variance_ratio_cumsum>=variance_sum_threshold)[0][0]
        
        #run PCA again with the number of component above
        pca = PCA(n_components=num_comp_to_reach_variance_threshold)
        pca.fit(X_scaled)
        return(pd.DataFrame(pca.fit_transform(X_scaled), 
                            columns=list('pc_' + pd.Series(np.arange(1, num_comp_to_reach_variance_threshold+1)).astype('str'))))


    #initiate PCA process by meeting component threshold
    elif threshold_by_component == True:
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        variance_sum = sum(pca.explained_variance_ratio_)
        print("The sum of variance is {:.3f}.".format(variance_sum))
        return(pd.DataFrame(pca.fit_transform(X_scaled), 
               columns=list('pc_' + pd.Series(np.arange(1, n_components+1)).astype('str'))))

def tsne_on_pc(X, n_components = 2, verbose = 1, perplexity = 50, n_iter = 1000, method = 'barnes_hut', export_plot = False):
    """
    Use this function of perform t-SNE on principal components.

    Parameter
    ----------
    X: a panda dataframe contains principal components (dataframe returned by function PCA_w_scaler).

    n_components: number of components return from running the t-SNE function. It has to be 2.

    verbose: int ranging from 0 to 1, optional (default: 1)
        User could use this variable to specify if he/she wants the process to be print out.

    perplexity: float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. 
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. 
        The choice is not extremely critical since t-SNE is quite insensitive to this parameter. (from sklearn documentation)

    n_iter: int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at least 250. (from sklearn documentation)

    method: string (default: ‘barnes_hut’)
        By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. 
        method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. 
        The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. However, 
        the exact method cannot scale to millions of examples. (from sklearn documentation)

    export_plot: string, boolean (default: False)
        IF True, the function would return a 2-D tsne plot.

    Returns
    -------
    tsne_results_df: a pandas dataframe with 2 t-SNE components.

    tsne_plot: a scatter plot with 2 t-SNE components (only if the user specifies a plot to be reuturned).
    """
    pca_df = PCA_w_scaler(X, threshold_by_component = False)

    if n_components != 2:
        print("This specific t-SNE function could be ran with n_components not equal to 2.")

    elif n_components ==2 & export_plot = True:
        tsne = TSNE(n_components= n_components, verbose= verbose, perplexity=perplexity, n_iter=n_iter, method = method)
        tsne_results = tsne.fit_transform(pca_df)
        tsne_results_df = pd.DataFrame(tsne_results, columns = ['tsne_x', 'tsne_y'])

        tsne_plot = sns.scatterplot(x='tsne_x', y='tsne_y', data=tsne_results_df)
        return(tsne_results_df, tsne_plot)

    elif n_components ==2:
        tsne = TSNE(n_components= n_components, verbose= verbose, perplexity=perplexity, n_iter=n_iter, method = method)
        tsne_results = tsne.fit_transform(pca_df)
        tsne_results_df = pd.DataFrame(tsne_results, columns = ['tsne_x', 'tsne_y'])
        return(tsne_results_df)

def evaluate_tsne_result(tsne_results_df, evaluation_method = 'kmeans', evluation_metric = 'silhouette'):



