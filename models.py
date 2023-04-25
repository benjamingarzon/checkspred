
# P5/P6
# is there a difference between imputed and not imputed
# fix imputation

import os, sys
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from utils import fit_regression_model, fit_XGB_model, MAX_AGE, MIN_AGE
from sklearn.preprocessing import PolynomialFeatures

#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer

SCALES =  ['dles', 'dsif', 'ehoe', 'eles', 'fhoe', 'fles', 'mzuv', 'mfur', 'mgfd']
MS_SCALES = ['dles','dsif','ehoe','eles','esif','fhoe','fles','fsif','mfur','mgfd','mzuv']
WLES = [ 'wle_' + s for s in SCALES ]

MAX_MISS_SCALE = 2
MIN_POINTS = 3
MIN_POINTS_POLY = 10
AGE_DIFF = 0.5

#features_global1 = ['mean_1', 'sd_1']
#features_global2 = ['mean_2', 'sd_2']
#features_scale1 = [f"('{m}', '{f}')" for f in ['dles','dsif','ehoe','eles','mfur','mgfd','mzuv'] for m in ['mean_1']]
#features_scale2 = [f"('{m}', '{f}')" for f in ['dles','dsif','ehoe','eles','mfur','mgfd','mzuv'] for m in ['mean_2']]

features_scale_last = [f"('{m}', '{f}')" for f in MS_SCALES for m in ['last']]
features_scale_mean = [f"('{m}', '{f}')" for f in MS_SCALES for m in ['mean']]
features_scale_sd = [f"('{m}', '{f}')" for f in MS_SCALES for m in ['sd']]
features_scale_poly = [f"('{m}', '{f}')" for f in MS_SCALES for m in ['degree_0', 'degree_1']]

features_last = ['last'] 
features_mean = ['mean']
features_sd = ['sd']
features_poly = ['degree_0', 'degree_1'] 

features_demo = ['gender', 'mother_tongue']
features_age = ['age']
features_check = [ 'wle_%s_last'%s for s in SCALES ]

features_dict = {

  'g': features_demo,
  'a': features_age,
  'c': features_check,

  'l': features_last,
  'm': features_mean,
  's': features_sd,
  'p': features_poly,

  'L': features_scale_last,
  'M': features_scale_mean,
  'S': features_scale_sd,
  'P': features_scale_poly

}

# define different setups
#SCENARIOS = ['main', 'xgb', 'checks', 'adaptive', 'nonadaptive' ]

ALGOS = {
  'test': ['ridge'],
  'main': ['ridge'],
  'xgb': ['xgb'],
  'checks': ['ridge'],
  'adaptive': ['ridge'],
  'nonadaptive': ['ridge']
   }

DESCRIPTORS = {
  'test': ['gMI_7'],
  'main': ['gl_1', 'gL_2', 'gm_3', 'gM_4', 'gMs_5', 'gMp_6', 'gMI_7', 'gMc_8', 'gc_9'],
  'xgb': ['gM_4'],
  'checks': ['gMc_8', 'gc_9'],
  'adaptive': ['gM_4'],
  'nonadaptive': ['gM_4']
   }
   
USE_CASES = {
  'test': ['nonadaptive'],
  'main': ['all'],
  'xgb': ['all'],
  'checks': ['all'],
  'adaptive': ['adaptive'],
  'nonadaptive': ['nonadaptive']
    }

CHECK_TYPES = ['S2', 'P5', 'S3']

EQUALIZE = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'equal':
        EQUALIZE = True

if len(sys.argv) > 2 and sys.argv[2] == 'test':
    SCENARIOS = ['test' ]
else:
    SCENARIOS = ['main', 'checks', 'nonadaptive', 'adaptive', 'xgb']

def prettify_feature(feature):
    x = ''
    for l in feature:
        if l == ' ':
            x = x + '_'
        elif l not in [",", "(", ")", "'"]:
            x = x + l
    return(x)    

def get_features(descriptor):
    features = []
    for l in descriptor:
        if l == '_':
            break
        if l == 'I':
            continue
        features += features_dict[l]
    return(features)


def fit_model(check_type, model_type, descriptor, descriptors, use_case, equalize=False):
    descriptor_all = list(set(''.join(descriptors)))
    
    if equalize:
        out_path = './out/equalized'
    else:
        out_path = './out/not_equalized'
    print(out_path)

    DATA_PATH = './data/merged_data_%s.csv'%(check_type)
    Y_PATH = '%s/y_data_%s_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor, use_case)
    SCORES_PATH = '%s/scores_data_%s_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor, use_case)
    N_DATA_PATH = '%s/n_data_%s_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor, use_case)
    BEST_PARAMS_PATH = '%s/best_params_%s_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor, use_case)
    IMPORTANCES_PATH = '%s/importances_%s_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor, use_case)

    df = pd.read_csv(DATA_PATH)
    print(*df.columns.tolist(), sep=' ')
    #print("Nas")
    #print(df.isna().sum())
    #print("Min")
    #print(df.min())
    #print("Max")
    #print(df.max())
    #print("Mean")
    #print(df.mean())
    #df = df.dropna()
    df = df.rename(columns={'motherTongue':'mother_tongue'})
    
    features = get_features(descriptor)

    if use_case == 'adaptive':
        df = df.loc[df.useCase.isin(['ms-kompetenzbereich'])]
    if use_case == 'nonadaptive':
        df = df.loc[df.useCase.isin(['ms-kompetenz', 'ms-steps', 'ms-thema'])]

    df = df.dropna(subset='gender')
    df = df.dropna(subset='mother_tongue')
    df = df.dropna(subset='age')
    df = df.loc[df['N'] >= MIN_POINTS]

    df['missing'] = df[features].isnull().sum(axis=1)

    df['missing_last'] = df[features_last].isnull().sum(axis=1)
    df['missing_mean'] = df[features_mean].isnull().sum(axis=1)
    df['missing_sd'] = df[features_sd].isnull().sum(axis=1)
    df['missing_poly'] = df[features_poly].isnull().sum(axis=1)

    df['missing_scale_last'] = df[features_scale_last].isnull().sum(axis=1)
    df['missing_scale_mean'] = df[features_scale_mean].isnull().sum(axis=1)
    df['missing_scale_sd'] = df[features_scale_sd].isnull().sum(axis=1)
    df['missing_scale_poly'] = df[features_scale_poly].isnull().sum(axis=1)

    df['missing_check'] = df[features_check].isnull().sum(axis=1)

    df['missing_target'] = df[WLES].isnull().sum(axis=1)
    #print(df.loc[df['missing_check'] < 9]['missing_check'])
    #print(df.loc[df['missing_check'] < 9][features_check])
    #print(df.missing_check.value_counts(normalize=False))

    # filter data
    print(df.shape)
    print("Missing:", df['missing'].min(), df['missing'].max())
    print(df[features])
    print(df.shape)

    if 'l' in descriptor or (equalize and 'l' in descriptor_all):
        df = df.loc[df['missing_last'] < len(features_last)]
        df = df.loc[df['before_exam'] == True ]
    if 'm' in descriptor or (equalize and 'm' in descriptor_all):
        df = df.loc[df['missing_mean'] < len(features_mean)]
        df = df.loc[df['before_exam'] == True ]
    if 's' in descriptor or (equalize and 's' in descriptor_all):
        df = df.loc[df['missing_sd'] < len(features_sd)]
        df = df.loc[df['before_exam'] == True ]
    print(df.shape)
    if 'p' in descriptor or (equalize and 'p' in descriptor_all):
        df = df.loc[df['missing_poly'] == 0]
        df = df.loc[df['N'] >= MIN_POINTS_POLY]
        df = df.loc[df['age_diff'] >= AGE_DIFF]
        df = df.loc[df['before_exam'] == True ]
    print(df.shape)
    if 'c' in descriptor or (equalize and 'c' in descriptor_all):
        df = df.loc[df['missing_check'] <= MAX_MISS_SCALE]
    #print(df)
    #print(df.shape)
    #print(df['missing_check'].max())
    #print(df['missing'].max())
    #stope

    if 'L' in descriptor or (equalize and 'L' in descriptor_all):
        df = df.loc[df['missing_scale_last'] <= MAX_MISS_SCALE]
    if 'M' in descriptor or (equalize and 'M' in descriptor_all):
        df = df.loc[df['missing_scale_mean'] <= MAX_MISS_SCALE]
    if 'S' in descriptor or (equalize and 'S' in descriptor_all):
        df = df.loc[df['missing_scale_sd'] <= MAX_MISS_SCALE]
    if 'P' in descriptor or (equalize and 'P' in descriptor_all):
        df = df.loc[df['missing_scale_poly'] <= MAX_MISS_SCALE*len(features_scale_poly)]
        df = df.loc[df['N'] >= MIN_POINTS_POLY]
        df = df.loc[df['age_diff'] >= AGE_DIFF]
    print(df.shape)

    if equalize:
        #print(df.missing_target.value_counts(normalize=False))
        df = df.loc[df['missing_target'] == 0]
        
    # add interactions
    if 'I' in descriptor:
        
        numerical_features = [ff for ff in features if ff not in features_demo]
        df.loc[:, numerical_features] = df.loc[:, numerical_features].apply(lambda s: s - s.mean())
        X = df[features]
        lx = X.shape[1]
        interactions = [ pd.Series(X.iloc[:, i]*X.iloc[:, j], name=X.iloc[:, i].name + '-' + X.iloc[:, j].name) for i in range(lx-1) for j in range(i+1, lx)]
        
        # concatenate the original dataframe with the new interaction columns
        interactions = pd.concat(interactions, axis=1)
        features = features + interactions.columns.tolist()
        df = pd.concat((df, interactions), axis=1)

    print(df.shape)
        
    y_all = []
    ypred_all = []
    y_target = [] 
    r2_all = []
    mse_all = []
    nsub_all = []
    nfeat_all = []
    scores_target = [] 
    missing_all = []
    best_params_all = []
    importances_all = []
    
    for target in WLES:
        print(target)
        if model_type == 'xgb':
            y, ypred, r2, mse, missing, nsub, nfeat, best_params, importances = fit_XGB_model(df, features, target, descriptor, shuffle=False)
        else:
            y, ypred, r2, mse, missing, nsub, nfeat, best_params, importances = fit_regression_model(df, features, target, model_type, shuffle=False)
            #y, ypred, scores, missing = fit_XGB_model(df, features, target)

        y_all.append(y)
        ypred_all.append(ypred)
        r2_all.append(r2)
        mse_all.append(mse)
        nsub_all.append(nsub)
        nfeat_all.append(nfeat)
        y_target.extend(len(y)*[target])
        scores_target.extend(len(r2)*[target])
        missing_all.append(missing)
        best_params_all.append(best_params)
        importances = pd.DataFrame(importances, 
        columns = [ prettify_feature(x) for x in features ])
        importances['target'] = target
        importances_all.append(importances)
    
    y_data = pd.DataFrame({'check_type': check_type,
                           'target': y_target, 
                           'y': np.concatenate(y_all), 
                           'ypred': np.concatenate(ypred_all), 
                           'missing': np.concatenate(missing_all, dtype = int)})
                       
    scores_data = pd.DataFrame({'check_type': check_type,
                                'target': scores_target, 
                                'r2': np.concatenate(r2_all),
                                'mse': np.concatenate(mse_all)})

    n_data = pd.DataFrame({'check_type': check_type,
                           'target': WLES, 
                           'nsub': np.array(nsub_all),
                           'nfeat': np.array(nfeat_all)})
                      
    best_params = pd.DataFrame({k: [dic[k] for dic in best_params_all] 
    for k in best_params_all[0]})
    importances = pd.concat(importances_all)

    y_data.to_csv(Y_PATH, index=False)
    scores_data.to_csv(SCORES_PATH, index=False)
    n_data.to_csv(N_DATA_PATH, index=False)
    best_params.to_csv(BEST_PARAMS_PATH, index=False)
    #importances['target'] = WLES
    importances.to_csv(IMPORTANCES_PATH, index=False)
    
for scenario in SCENARIOS: 
    descriptors = DESCRIPTORS[scenario]
    use_cases = USE_CASES[scenario]
    algos = ALGOS[scenario]   

    for check_type in CHECK_TYPES:
    
        for algo in algos:
            for use_case in use_cases:
                for descriptor in descriptors:
                    print(100*"-")
                    print(scenario, check_type, algo, use_case,  descriptor)
                    print(100*"-")
                    if False:
                        fit_model(check_type, algo, descriptor, descriptors, 
                        use_case, equalize=EQUALIZE)
                        continue
                    try:
                        fit_model(check_type, algo, descriptor, 
                        descriptors, use_case, equalize=EQUALIZE)
                        
                    except Exception as e: 
                        print(f"An exception occurred for {check_type}")
                        print(e)
                        
