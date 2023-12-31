import os, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import open_file, fix_ids, get_features, DEGREE, res_names, \
MAX_AGE, MIN_AGE, MEAN_AGE, get_mode

from joblib import Parallel, delayed

STAGES = ['demo', 'mindsteps', 'checks', 'parallel']
#STAGES = ['parallel']

SCALES =  ['dles', 'dsif', 'ehoe', 'eles', 'fhoe', 'fles', 'mzuv', 'mfur', 'mgfd']

WLES = [ 'wle_' + s for s in SCALES ]
SEWLES = [ 'sewle_' + s for s in SCALES ]

CHECKS_COLS = [
    'studentId',
    'check_type',
    'check_year',
#    'is_public_sc',
    'comparison'
  ] + WLES

MS_COLS = [
#    'assessmentSessionId',
    'studentId',
    'timestamp',
    'Ability',
    'useCase',
    'Scale'
#    'model'
  ]
  
DEMO_COLS = [
  'studentId',
  'date',
  'age',
  'Gender',
  'motherTongue',
  'frequency', 
  'previous_sessions', 
  'years_from_start'
  ]

FINAL_COLS = [
  'studentId', 
#  'assessmentId',
  'scale', 
  'age', 
  'gender', 
  'motherTongue', 
  'estimate', 
  'check_age', 
  'useCase', 
  'before_exam',
  'frequency', 
  'previous_sessions', 
  'years_from_start'
  ]

PIVOT_COLS = [
  'studentId', 
  'useCase', 
  'before_exam'
]


NSTUDS = 10000

TEST = False

last_check_dict = {'S3': 'S2', 'S2': 'P5', 'P5': 'P3'}

MS_PATH = './data/abilities_2pl.rda'
CHECKS_PATH = './data/CDW_data_forILD_2022-10-25.Rds' #'checks_data.RdsCDW_S2_data_forILD_2022-10-07.Rds'
DEMO_PATH = '/home/garben/mindsteps/data/mindsteps/dd_2023_Jan_valid_parameters.rds'

MODEL_TYPE = 'all_2_studentId_personpreds_itempreds_excludeBadSessions_onlyClosed'

CHECK_TYPES = ['S2', 'P5', 'S3'] #, 'P3']

start_time = time.time()

MS_TEST_PATH = './data/aux/ms_data_%d.csv'%(NSTUDS)
DEMO_TEST_PATH = './data/aux/demo_data_%d.csv'%(NSTUDS)
DEMO_CLEAN_PATH = './data/aux/demo_clean.csv'
DEMO_EXT_PATH = './data/aux/demo_ext.csv'
MS_AGE_PATH = './data/aux/ms_age.csv'
CHECKS_AGE_PATH = './data/aux/checks_age.csv'
CHECKS_TEST_PATH = './data/aux/checks_data_%d.csv'%(NSTUDS)


def data_descriptor(df, name, check_type):
    print(50*'*')
  
    print('Descriptor for %s, %s:'%(name, check_type))
    print(df.columns)
    print('Shape')
    print(df)
    print('Max age range:')
    #print(df.groupby('studentId').age.agg([('age_diff',  lambda x : np.max(x) - np.min(x))]).reset_index().max())
    print(df.groupby('studentId').age.agg([('age_diff',  lambda x : np.max(x) - np.min(x))]).reset_index().max())
    print(50*'-')

def prepare_data(check_type):
    last_check_type = last_check_dict[check_type]
    
    istest = '_test' if TEST else ''
    OUT_PATH = './data/merged_data_%s%s.csv'%(check_type, istest)
    
    CHECKS_TEST_PATH = './data/aux/checks_data_%d_%s.csv'%(NSTUDS, check_type)
    TEST_INT_PATH = './data/aux/aux_data_%d_%s.csv'%(NSTUDS, check_type)

    # checks data
    df_checks = open_file(CHECKS_PATH, TEST, NSTUDS, CHECKS_COLS, CHECKS_TEST_PATH)
    print(df_checks.columns)
    print(df_checks)
    df_checks = df_checks.dropna(subset=['studentId'])

    df_checks['studentId'] = df_checks['studentId'].apply(lambda x: fix_ids(x))
    
    # translate missing codes, anything above 90 should be nan
    df_checks[WLES] = np.where(df_checks[WLES] > 90, np.nan, df_checks[WLES])

    #df_checks = df_checks.loc[df_checks['is_public_sc'] == 1 ].drop(columns=['is_public_sc'])
    df_checks = df_checks.loc[df_checks['comparison'] == 1 ].drop(columns=['comparison'])
    
    df_checks = df_checks.sort_values(['studentId','check_year'])
    df_checks = df_checks.groupby(['studentId', 'check_type']).first().reset_index()
    df_checks['check_timestamp'] = df_checks['check_year'].apply(lambda x: pd.Timestamp(int(x), 2, 1, 0))
    df_checks['check_year'] = df_checks['check_year'].astype('int')

    df_checks_last = df_checks.loc[df_checks['check_type'] == last_check_type]
    # select only the given check type
    df_checks = df_checks.loc[df_checks['check_type'] == check_type]
    print(df_checks.columns)

    # keep only if date earlier than checks tests
    df = df_ms.merge(df_checks[['studentId','check_timestamp']], on='studentId', ## added how = left
    how='left').reset_index(drop=True)
    data_descriptor(df, 'df_pre', check_type)
    #df = df.loc[df.timestamp < df.check_timestamp]
    df['before_exam'] = df.timestamp < df.check_timestamp
    data_descriptor(df, 'df_post', check_type)
    df['check_age'] = (pd.to_datetime(df['check_timestamp']) - pd.to_datetime(df['btimestamp']))/timedelta(days=365.2425)
    df.to_csv(TEST_INT_PATH, index=False)
    
    df_group1 = df[FINAL_COLS].groupby(PIVOT_COLS + ['scale']).apply(get_features).reset_index()
    df_group2 = df[FINAL_COLS].groupby(PIVOT_COLS).apply(get_features).reset_index()
    print(df_group1)
    print(df_group2)
    df_group1 = df_group1.pivot(index=PIVOT_COLS, columns='scale', values=res_names[3:]).reset_index()
    colnames = PIVOT_COLS + df_group1.columns.tolist()[3:] 
    df_group1 = df_group1.set_axis(colnames, axis=1, inplace=False)
    df_group = df_group1.merge(df_group2, on=PIVOT_COLS).reset_index()
    df_group = df_group.merge(df_checks, on='studentId').reset_index()
    # generate also averaged version
    
    # add the last check data
    df_group = df_group.merge(df_checks_last, on='studentId', suffixes=('', '_last'), how='left').reset_index(drop=True)
    
    df_group.to_csv(OUT_PATH, index=False)
    data_descriptor(df_group, 'df_group', check_type)

    end_time = time.time()
    print(f"Total time: {np.round_(end_time - start_time, 3)} seconds.")

#for check_type in CHECK_TYPES:
#    prepare_data(check_type)

def do_prepare(check_type, df_ms):
    
    prepare_data(check_type)
    #return
  
    #try:
    #    prepare_data(check_type)
    #except Exception as e: 
    #    print(f"An exception occurred for {check_type}")
    #    print(e)

if 'demo' in STAGES:
    # demographic data
    df_demo = open_file(DEMO_PATH, TEST, NSTUDS, DEMO_COLS, DEMO_TEST_PATH)
    # arrange variables
    df_demo = df_demo.loc[df_demo['age'] >= MIN_AGE]
    df_demo = df_demo.loc[df_demo['age'] <= MAX_AGE]
    #df_demo['timestamp'] = df_demo['timestamp_2'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%d.%m.%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')))

    # for ages, take only into account month
    #df_demo['yearmonth'] = df_demo['timestamp'].dt.strftime('%Y-%m')
    df_demo['timestamp'] = df_demo['date']
    #df_demo['day'] = df_demo['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    df_demo['day'] = df_demo['timestamp'].dt.strftime('%Y-%m-%d')
    #df_demo = df_demo.drop(columns = ['timestamp_2'])
    df_demo = df_demo.drop_duplicates()
    data_descriptor(df_demo, 'demo', 'all')
    df_demo = df_demo.rename(columns={'age':'age_orig', 'Gender': 'gender'})
    #df_demo.gender.fillna(value=np.nan, inplace=True)
    print('=========')
    print(df_demo)
    
    df_demo_ext = df_demo.groupby(['studentId', 'day']).agg({ 
      'age_orig': np.nanmean, 
      'frequency' : np.nanmean, 
      'years_from_start' : np.nanmean, 
      'previous_sessions' : np.nanmean }).reset_index().rename(columns={'age_orig':'age_ext'})

    df_demo = df_demo.groupby(['studentId']).agg({ 
      'age_orig': 'first', 'timestamp': 'first', 'gender': get_mode, 'motherTongue': get_mode}).reset_index() 
    df_demo = df_demo.dropna(subset=['age_orig', 'timestamp'])

    df_demo['btimestamp'] = df_demo['timestamp'] - df_demo['age_orig'].apply(lambda x: timedelta(days=x*365.2425))
    
    #print(df_demo.shape)
    
    # add features
    #df_aux = df_demo.drop_duplicates(['studentId', 'timestamp']).sort_values(['studentId', 'timestamp'], ascending=True)
    #df_aux['previous_sessions'] = df_aux.set_index(['studentId']).groupby(level=[0]).cumcount().values
    #df_aux['years_from_start'] = df_aux.groupby('studentId', group_keys=False).age_orig.apply(lambda x: x - x.min()).values
    #df_aux['pace'] = df_aux['years_from_start']/(df_aux['previous_sessions'] + 1e-6)
    #df_aux['frequency'] = 1/(df_aux['pace'] + 1e-6)
    
    #df_demo.set_index(['studentId', 'timestamp'], inplace=True)
    #df_aux.set_index(['studentId', 'timestamp'], inplace=True)
    #df_demo = df_demo.join(df_aux[['frequency', 'previous_sessions', 'years_from_start']], how='left').reset_index()

    print(df_demo.shape)
    print(df_demo.columns)

    df_demo.to_csv(DEMO_CLEAN_PATH, index=False)
    df_demo_ext.to_csv(DEMO_EXT_PATH, index=False)
    print(df_demo)
    print('.........')
else:     
    df_demo = pd.read_csv(DEMO_CLEAN_PATH)
    df_demo_ext = pd.read_csv(DEMO_EXT_PATH)
    
if 'mindsteps' in STAGES:

    # abilities data
    df_ms = open_file(MS_PATH, TEST, NSTUDS, MS_COLS, MS_TEST_PATH)
    
    #['model', 'Ability', 'AbilityLong', 'assessmentSessionId',
    #   'Scale', 'month.year', 'useCase', 'timestamp', 'age', 'grade', 'gender',
    #   'motherTongue', 'Nassessments']
    
    df_ms = df_ms.rename(columns={'Ability':'estimate', 'Scale':'scale'})
    #df_ms = df_ms.loc[df_ms['model'] == MODEL_TYPE ]
    df_ms['studentId'] = df_ms['studentId'].astype(int)
    #df_ms = df_ms.rename(columns={'timestamp_3': 'timestamp', 'scale2': 'scale'})
    # there is an offset of 1 hour!!
    df_ms['timestamp'] = pd.to_datetime(df_ms['timestamp']) + timedelta(hours=1)
    df_ms['year'] = df_ms['timestamp'].dt.strftime('%Y')
    #df_ms['yearmonth'] = df_ms['timestamp'].dt.strftime('%Y-%m')
    df_ms['day'] = df_ms['timestamp'].dt.strftime('%Y-%m-%d')
    df_ms = df_ms.dropna(subset='estimate')

    print("Mindsteps data")
    print(df_ms)
    df_ms = df_demo.drop(columns='timestamp').merge(df_ms, on=['studentId'], how='right').reset_index() 
    df_ms = df_demo_ext[['studentId', 'day', 'age_ext', 
    'frequency', 'previous_sessions', 'years_from_start']].merge(df_ms, on = ['studentId', 'day'], how='right').reset_index(drop=True)
    df_ms['age'] = (pd.to_datetime(df_ms['timestamp']) - pd.to_datetime(df_ms['btimestamp']))/timedelta(days=365.2425)
    df_ms.loc[df_ms['age_ext'].notnull(), 'age'] = df_ms['age_ext'][df_ms['age_ext'].notnull()]

    data_descriptor(df_ms, 'df_ms', 'all')
    df_ms.to_csv(MS_AGE_PATH, index=False)
    print(df_ms.dropna())

else:     
    df_ms = pd.read_csv(MS_AGE_PATH)

if 'checks' in STAGES:

    df_checks = open_file(CHECKS_PATH, TEST, NSTUDS, CHECKS_COLS, CHECKS_TEST_PATH)
    df_checks = df_checks.dropna(subset=['studentId'])
    df_checks['studentId'] = df_checks['studentId'].apply(lambda x: fix_ids(x))
    
    # translate missing codes, anything above 90 should be nan
    df_checks[WLES] = np.where(df_checks[WLES] > 90, np.nan, df_checks[WLES])

    df_checks = df_checks.loc[df_checks['comparison'] == 1 ].drop(columns=['comparison'])
    
    df_checks = df_checks.sort_values(['studentId','check_year'])
    df_checks = df_checks.groupby(['studentId', 'check_type']).first().reset_index()
    df_checks['check_timestamp'] = df_checks['check_year'].apply(lambda x: pd.Timestamp(int(x), 2, 1, 0))
    df_checks['check_year'] = df_checks['check_year'].astype('int')
    
    # keep only if date earlier than checks tests
    df = df_ms.merge(df_checks[['studentId','check_timestamp','check_type'] + WLES], on='studentId').reset_index(drop=True)
    df['check_age'] = (pd.to_datetime(df['check_timestamp']) - pd.to_datetime(df['btimestamp']))/timedelta(days=365.2425)
    df.to_csv(CHECKS_AGE_PATH, index=False)

else:     
    df = pd.read_csv(CHECKS_AGE_PATH)

if 'parallel' not in STAGES:
  for check_type in CHECK_TYPES:
    do_prepare(check_type, df_ms)
else:
  Parallel(n_jobs=len(CHECK_TYPES))(delayed(do_prepare)(check_type, df_ms) for check_type in CHECK_TYPES)
