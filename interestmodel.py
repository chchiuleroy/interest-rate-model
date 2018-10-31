# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:34:57 2018

@author: roy
"""

def tune(status):
    import pandas as pd, numpy as np
    
    factor = ['loan_amnt', 'term', 'annual_inc', 'verification_status', 'dti', 'fico_range_low', 
              'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util'] 
    
    from sklearn.externals import joblib
    from sklearn.ensemble import RandomForestRegressor as rfr
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.model_selection import RandomizedSearchCV
    from pprint import pprint
    
    train_x = d0[factor]
    train_y = d0['int_rate']
    
    n_estimators = [int(x) for x in np.linspace(20, 100, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    
    rf = rfr()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 4, verbose = 2, n_jobs = -1)
    model = rf_random.fit(train_x, train_y)
    pprint(rf_random.best_params_)
    joblib.dump(model, 'mod'+str(status)+'.pkl')
    fit = model.predict(train_x)
    mse = MSE(train_y, fit)
    
    return mse


def process(dataset):
    import numpy as np
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
    from sklearn.metrics import explained_variance_score as EVS, mean_squared_error as MSE
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.wrappers.scikit_learn import KerasRegressor
    
    col = ['int_rate', 'term', 'fico_range_high', 'dti', 'loan_amnt', 'bc_open_to_buy', 'annual_inc', 
            'inq_last_6mths', 'num_tl_op_past_12m', 'verification_status', 'total_rev_hi_lim', 'total_bc_limit', 
            'mo_sin_old_rev_tl_op', 'mo_sin_old_il_acct', 'total_acc', 'tot_hi_cred_lim']
    
    data = dataset[col]
    
    scaler = StandardScaler().fit(data)
    trans_data = scaler.transform(data)
    
    x = trans_data[:, 1:16]
    y = trans_data[:, 0]
    
    del col, dataset, data   
  
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1, random_state = 42)
    
    def create_model():
        model = Sequential()
        model.add(Dense(150, input_dim = 15, activation = 'relu', kernel_initializer = 'normal'))
#        model.add(Dense(100, input_dim = 15, activation = 'relu', kernel_initializer = 'normal'))
#        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer = 'normal'))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        return model

    model = KerasRegressor(build_fn = create_model, epochs = 30, batch_size = 40, verbose = 0)
    kfold = KFold(n_splits = 5, random_state = 100)
    results = cross_val_score(model, train_x, train_y, cv = kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    final = EVS(test_y, prediction)

    return final
