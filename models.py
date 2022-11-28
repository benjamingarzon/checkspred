
# is there a difference between imputed and not imputed
# fix imputation
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from utils import fit_regression_model, fit_XGB_model, MAX_AGE, MIN_AGE
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer

SCALES =  ['dles', 'dsif', 'ehoe', 'eles', 'fhoe', 'fles', 'mzuv', 'mfur', 'mgfd']
MS_SCALES = ['dles','dsif','ehoe','eles','mfur','mgfd','mzuv']
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

features_gender = ['gender']
features_age = ['age']
features_check = [ 'wle_%s_last'%s for s in SCALES ]

features_dict = {

  'g': features_gender,
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

#DESCRIPTORS = ['gl-1', 'gL-2', 'gm-3', 'gM-4', 'gMs-5', 'gMp-6', 'gMpc-7', 'gc-8', 'gmpc-9', 'gms-10', 'gmp-11']
DESCRIPTORS = ['gl_1', 'gL_2', 'gm_3', 'gM_4', 'gMs_5', 'gMp_6', 'gMpc_7', 'gc_8', 'gmpc_9', 'gms_10', 'gmp_11']
#DESCRIPTORS = ['gc_8']
#DESCRIPTORS = ['gM']

descriptor_all = list(set(''.join(DESCRIPTORS)))

CHECK_TYPES = ['S2', 'P5', 'S3']

MODEL_TYPES = [ 'ridge'] #'xgb',
MODEL_TYPES = ['ridge']
MODEL_TYPES = ['xgb']
EQUALIZE = True

def get_features(descriptor):
  features = []
  for l in descriptor:
      if l == '_':
          break
      features += features_dict[l]
  print(features)
  return(features)


def fit_model(check_type, model_type, descriptor, equalize=False):
    if equalize:
        out_path = './out/equalized'
    else:
        out_path = './out/not_equalized'
        
    DATA_PATH = './data/merged_data_%s.csv'%(check_type)
    Y_PATH = '%s/y_data_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor)
    SCORES_PATH = '%s/scores_data_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor)
    N_DATA_PATH = '%s/n_data_%s_%s_%s.csv'%(out_path, check_type, model_type, descriptor)

    df = pd.read_csv(DATA_PATH)
    #print(df.columns)

    #print("Nas")
    #print(df.isna().sum())
    #print("Min")
    #print(df.min())
    #print("Max")
    #print(df.max())
    #print("Mean")
    #print(df.mean())
    #df = df.dropna()

    features = get_features(descriptor)
    #print(df.shape)
    
    df = df.dropna(subset='gender')
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

    #print(df)

    # filter data
    print(df.shape)
    print("Missing:", df['missing'].min(), df['missing'].max())
    print(df[features])
    print(df.shape)

    if 'l' in descriptor or (equalize and 'l' in descriptor_all):
        df = df.loc[df['missing_last'] < len(features_last)]
    if 'm' in descriptor or (equalize and 'm' in descriptor_all):
        df = df.loc[df['missing_mean'] < len(features_mean)]
    if 's' in descriptor or (equalize and 's' in descriptor_all):
        df = df.loc[df['missing_sd'] < len(features_sd)]
    print(df.shape)
    if 'p' in descriptor or (equalize and 'p' in descriptor_all):
        df = df.loc[df['missing_poly'] == 0]
        df = df.loc[df['N'] >= MIN_POINTS_POLY]
        df = df.loc[df['age_diff'] >= AGE_DIFF]
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
    for target in WLES:
        print(target)
        if model_type == 'xgb':
            y, ypred, r2, mse, missing, nsub, nfeat = fit_XGB_model(df, features, target, descriptor, shuffle=False)
        else:
            y, ypred, r2, mse, missing, nsub, nfeat = fit_regression_model(df, features, target, model_type, shuffle=False)
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
    
    y_data = pd.DataFrame({'check_type': check_type,
                           'target': y_target, 
                           'y': np.concatenate(y_all), 
                           'ypred': np.concatenate(ypred_all), 
                           'missing': np.concatenate(missing_all, dtype = int) })
                       
    scores_data = pd.DataFrame({'check_type': check_type,
                                'target': scores_target, 
                                'r2': np.concatenate(r2_all),
                                'mse': np.concatenate(mse_all)})

    n_data = pd.DataFrame({'check_type': check_type,
                           'target': WLES, 
                           'nsub': np.array(nsub_all),
                           'nfeat': np.array(nfeat_all)})


    y_data.to_csv(Y_PATH, index=False)
    scores_data.to_csv(SCORES_PATH, index=False)
    n_data.to_csv(N_DATA_PATH, index=False)
    
for model_type in MODEL_TYPES: 
    for check_type in CHECK_TYPES:
        for descriptor in DESCRIPTORS:
            #fit_model(check_type, model_type, descriptor, equalize=EQUALIZE)
            #continue
            try:
                fit_model(check_type, model_type, descriptor, equalize=EQUALIZE)
            except Exception as e: 
                print(f"An exception occurred for {check_type}")
                print(e)

           
    
    
      
