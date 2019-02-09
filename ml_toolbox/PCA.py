import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def filter_high_zeros_count(X, threshold = 0.8, impute_with_zeros=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #make sure the the input dataframe is all numeric
    if X.select_dtypes(include=numerics).shape[1] == X.shape[1]:
        high_zero_count_df = pd.DataFrame(X.apply(lambda x: sum(x==0)/len(X))).reset_index()
        high_zero_count_df.columns = ['feat_names', 'pct_zeros']
        filtered_col_names = high_zero_count_df[high_zero_count_df['pct_zeros']<=threshold]['feat_names'].values
        return(X[filtered_col_names])
    else:
        raise CustomException("Input dataframe is not all numerics. Process failed.")

def calculate_vif(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def PCA_w_scaler(X, threshold_by_component = True, n_components = 10,
        threshold_by_variance=False, variance_sum_threshold=0.8):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if threshold_by_variance==True & threshold_by_component==False:
        pca = PCA(n_components = X.shape[1])
        pca.fit(X_scaled)
        variance_sum = sum(pca.explained_variance_ratio_)
        if variance_sum < variance_sum_threshold:
            print("The sum of variance is {:.3f} and does not meet the threshold requirement.".format(variance_sum))
    else:
        #dont have to return everything 
        return(pd.DataFrame(pca.fit_transform(X_scaled), columns=list('pc_' + pd.Series(np.arange(1, n_components+1)).astype('str'))))



    elif threshold_by_component == True:
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        variance_sum = sum(pca.explained_variance_ratio_)
        print("The sum of variance is {:.3f}.".format(variance_sum))
        return(pd.DataFrame(pca.fit_transform(X_scaled), columns=list('pc_' + pd.Series(np.arange(1, n_components+1)).astype('str'))))

