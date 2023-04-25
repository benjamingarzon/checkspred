import pandas as pd
import pyreadr
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GridSearchCV
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

XGBTHREADS = 10
DEGREE = 1
MIN_POINTS = 4
MAX_AGE_DIFF = 0.25
MEAN_AGE = 13
MAX_AGE = 18
MIN_AGE = 6
N_FOLDS = N_INNER_FOLDS = 10

res_names = ['age', 'gender', 'motherTongue', 'age_diff', 'N', 'mean', 'sd', 
'last', 'mean_1', 'sd_1', 'mean_2', 'sd_2'] + ['degree_%d'%i for i in range(DEGREE + 1)]

def fit_regression_model(df, features, target, model_type='ridge', nfolds=N_FOLDS, shuffle=False):
    
    y = df[target] 
    X = df[features] 
    X = X.loc[ y.notnull() ].apply(lambda col:pd.to_numeric(col, errors='coerce'))

    if shuffle:
        X = X.apply(lambda x: np.random.permutation(x.values), axis=0)
    
    missing = df['missing'][ y.notnull() ]
    y = y[ y.notnull() ]
    if model_type == 'enet':
        lm = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_alphas = 10, random_state=0) 
    if model_type == 'ridge':
        lm = RidgeCV(alphas=np.logspace(-3, 5, 9))

    pipeline =  Pipeline([
                ('imputer', IterativeImputer(max_iter=100, random_state=0)),
                ('scaler', StandardScaler()),
                ('lm', lm),
            ])
    
    outer_cv = KFold(n_splits=nfolds, shuffle=True)
    importances_list = []
    y_list = []
    mse_list = []
    r2_list = []
    X = X.to_numpy()
    y = y.to_numpy()
    
    for train, test in outer_cv.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        pipeline.fit(X_train, y_train)
        ypred = pipeline.predict(X_test)
        r2_list.append(r2_score(y_test, ypred))
        mse_list.append(mean_squared_error(y_test, ypred))
        y_list.append((y_test, ypred))
        importances_list.append(np.abs(pipeline.named_steps['lm'].coef_))
    
    y_test, ypred = zip(*y_list)
    y_test = np.concatenate(y_test)
    ypred = np.concatenate(ypred)
    r2 = np.array(r2_list)
    mse = np.array(mse_list)
    importances = np.vstack(importances_list)
    
    #r21 = cross_val_score(pipeline, X, y, scoring='r2', cv=nfolds)
    #mse1 = -cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=nfolds)
    #ypred = cross_val_predict(pipeline, X, y, cv=nfolds).ravel()
    r2_mean = r2.mean()
    mse_mean = mse.mean()

    pipeline.fit(X, y)
    best_params = {'alpha': pipeline.named_steps['lm'].alpha_}
    
    #importances = np.abs(pipeline.named_steps['lm'].coef_)
    
    print(f"{model_type}: {r2_mean:.3f}")
    return y_test, ypred, r2, mse, missing, X.shape[0], X.shape[1], best_params, importances


def fit_XGB_model(df, features, target, descriptor, innern = N_INNER_FOLDS, outern = N_FOLDS, shuffle=False, train_only=False):
  
    y = df[target] #.sample(frac=1)
    X = df[features]
    X = X.loc[ y.notnull() ]
    
    if shuffle:
        X = X.apply(lambda x: np.random.permutation(x.values), axis=0)

    missing = df['missing'][ y.notnull() ]
    y = y[ y.notnull() ]
    
    xgb = XGBRegressor(
        nthread = XGBTHREADS,
        eval_metric = 'rmse'
#        tree_method="hist"
    )
    
    xgb_estimator =  Pipeline([
                ('imputer', IterativeImputer(max_iter=10, random_state=0)),
                ('scaler', StandardScaler()),
                ('xgb', xgb),
            ])
    
    inner_cv = KFold(n_splits=innern, shuffle=True)
    outer_cv = KFold(n_splits=outern, shuffle=True)

    param_grid = {
     #'xgb__min_child_weight': [1, 3], #Minimum sum of instance weight (hessian) needed in a child. 
#     'xgb__eta': [0.1, 0.2], #[0.01, 0.05, 0.1, 0.15], # Step size shrinkage used in update to prevents overfitting [0, 1]
#     'xgb__max_depth':[2, 3, 4, 6], # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 
#     'xgb__gamma': [0.5, 1, 3, 5], #Minimum loss reduction required to make a further partition on a leaf node of the tree. [0, inf]
#     'xgb__reg_alpha':[0, 10, 20, 30], # l1 reg degault =0
     'xgb__eta': [0.2], #[0.01, 0.05, 0.1, 0.15], # Step size shrinkage used in update to prevents overfitting [0, 1]
     'xgb__max_depth':[4, 6], # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 
     'xgb__gamma': [1, 5], #Minimum loss reduction required to make a further partition on a leaf node of the tree. [0, inf]
     'xgb__reg_alpha':[0, 10], #[0, 20, 30], # l1 reg degault =0
     'xgb__reg_lambda':[1, 10], #[1, 10, 20, 30], # l2 reg default=1
     'xgb__n_estimators':[30, 60, 90], #[20, 30, 60, 90, 100] # number of trees
    }

    #param_grid = {}

    pipeline = GridSearchCV(estimator = xgb_estimator, 
                       param_grid = param_grid, 
                       cv=inner_cv,
                       verbose = 1)
                       

    X = X.to_numpy()
    y = y.to_numpy()
                       
    if not train_only: 
      
       importances_list = []
       y_list = []
       mse_list = []
       r2_list = []
    
       for train, test in outer_cv.split(X):
           X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
           pipeline.fit(X_train, y_train)
           ypred = pipeline.predict(X_test)
           r2_list.append(r2_score(y_test, ypred))
           mse_list.append(mean_squared_error(y_test, ypred))
           y_list.append((y_test, ypred))

           importances_list.append(pipeline.best_estimator_.named_steps['xgb'].feature_importances_)

       y_test, ypred = zip(*y_list)
       y_test = np.concatenate(y_test)
       ypred = np.concatenate(ypred)
       r2 = np.array(r2_list)
       mse = np.array(mse_list)
       importances = np.vstack(importances_list)

       #r2 = cross_val_score(pipeline, X, y, scoring='r2', cv=outer_cv)
       #mse = -cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=outer_cv)
       #ypred = cross_val_predict(pipeline, X, y, cv=outer_cv).ravel()

       r2_mean = r2.mean()
       mse_mean = mse.mean()
       #print(pearsonr(ypred, y))
       #print(scores)
       print(f"xgb: {r2_mean:.3f}")
    else:
       ypred = y*0
       r2 = np.zeros(outern)
       mse = np.zeros(outern)

    pipeline.fit(X, y)
#    plot_grid_pars(pipeline, param_grid, 'n_estimators',  'reg_alpha', 'xgb', target, descriptor) #, 'max_depth', 'min_child_weight')
#    plot_grid_pars(pipeline, param_grid, 'eta',  'n_estimators', 'xgb', target, descriptor) #, 'max_depth', 'min_child_weight')
#    plot_grid_pars(pipeline, param_grid, 'max_depth',  'reg_alpha', 'xgb', target, descriptor) #, 'max_depth', 'min_child_weight')

#    plt.get_figure().savefig('./data/', dpi = 300)
    model = pipeline.best_estimator_.named_steps['xgb'].get_booster()
    model.feature_names = features
    best_params = pipeline.best_params_
    #importances = pipeline.best_estimator_.named_steps['xgb'].feature_importances_
    
    plt.figure(figsize=(8,8))
    plot_importance(model)
    #plt.legend()
    plt.savefig(f"./figs/importance_{target}_{descriptor}.png")

    return y_test, ypred, r2, mse, missing, X.shape[0], X.shape[1], best_params, importances

def plot_grid_pars(grid_gbm, gs_param_grid, par1, par2, clfname, target, descriptor):
    y=[]
    par1 = clfname + '__' + par1
    par2 = clfname + '__' + par2
    other_params = gs_param_grid.keys()
    other_params = [ par for par in other_params if par not in [par1, par2]]
    cvres = grid_gbm.cv_results_
    p1=gs_param_grid[par1]
    p2=gs_param_grid[par2]
    
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        for other_param in other_params:
            if params[other_param] != grid_gbm.best_params_[other_param]:
                mean_score = None
                break
              
        if mean_score is not None:
            y.append(mean_score)

    y=np.array(y).reshape(len(p1),len(p2))

    plt.figure(figsize=(8,8))
    for y_arr, label in zip(y, p1):
        plt.plot(p2, y_arr, label=label)
    plt.legend()
    plt.xlabel(par2)
    plt.ylabel('Score')
    #plt.show()
    plt.savefig(f"./figs/{par2}_{target}_{descriptor}.png")
    plt.close()
    plt.figure(figsize=(8,8))
    for y_arr, label in zip(y.transpose(), p2):
        plt.plot(p1, y_arr, label=label)
    
#    plt.title('Error for different %s (keeping max_depth=%d(best_param))'%best_md)
    plt.legend()
    plt.xlabel(par1)
    plt.ylabel('Score')
    #plt.show()
    plt.savefig(f"./figs/{par1}_{target}_{descriptor}.png")
    plt.close()


def fit_model_SVM(df, innern = 10, outern = 10):
    # add also description
    df = transform_features(df)
    
    y = df['followed_by']
    X = df.drop(['name', 'followed_by'], axis = 1).values
    # Set up possible values of parameters to optimize over
    param_grid = {"svc__C": [np.exp(i) for i in np.arange(1, 3, 0.5)],
                  "svc__gamma": [np.exp(i) for i in np.arange(-3, 1, 0.5)]}  
    
    # We will use a Support Vector Classifier with "rbf" kernel
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability = True)) #LinearSVC(max_iter=2000)) 

    inner_cv = KFold(n_splits=innern, shuffle=True)
    outer_cv = KFold(n_splits=outern, shuffle=True)

    balanced_scorer = make_scorer(balanced_accuracy_score)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svm, param_grid=param_grid, cv=inner_cv, 
                       verbose = 0, scoring = balanced_scorer)
    # Nested CV with parameter optimization
    score = cross_val_score(clf, X=X, y=y, cv=outer_cv, 
                            scoring = balanced_scorer)


    #print(score)
    print(score.mean())
    clf.fit(X, y)
    plot_grid_pars(clf, param_grid, 'C',  'gamma', 'svc')
    return(clf, score.mean())


def fix_ids(x):
    
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    #if 'e+05' in x:
    #    x = x.replace('e+05', '00000')  
    if 'e' in x:
        return int(float(x))
    return int(x.strip())
    

def open_file(DATA_PATH, TEST, NSTUD, COLS, TEST_PATH):
    print(f"Opening {DATA_PATH}")
    if not TEST:
        rds = pyreadr.read_r(DATA_PATH)
        df = rds[None] 
        sIds = pd.unique(df.studentId)
        #print(len(sIds))
        sIds = np.random.choice(sIds, size=NSTUD, replace=False)
        df[df.studentId.isin(sIds)].to_csv(TEST_PATH, index=False)
    else:
        df = pd.read_csv(TEST_PATH)

    print(df.columns)
    print(df.shape)

    df = df[COLS]

    return df
  
def get_features(df, degree=DEGREE, ref_age='check'):
  
    #print(df)
    x = df[['age']].values
    
    if ref_age is None:
        ref_age = MEAN_AGE
    else:
        ref_age = df[['check_age']].values
        
    mean_age = x.mean()
    gender = df['gender'].values[0]
    motherTongue = df['motherTongue'].values[0]
    lx = len(x)
    y = df[['estimate']].values
    mean_y = y.mean()
    sd_y = y.std()
    last_y = y[np.argmax(x)][0]

    # age in last year
    age_1 = (df[['check_age']].values - x) <= 1
    age_2 = (df[['check_age']].values - x) <= 2
    
    if age_1.size == 0:
        mean_1 = np.nan
        sd_1 = np.nan
    elif np.sum(age_1) > 0:
        mean_1 = y[age_1].mean()
        sd_1 = y[age_1].std()
    else:
        mean_1 = y[age_1]
        sd_1 = np.nan

    if age_2.size == 0:
        mean_2 = np.nan
        sd_2 = np.nan
    elif np.sum(age_2) > 0:
        mean_2 = y[age_2].mean()
        sd_2 = y[age_2].std()
    else:
        mean_2 = y[age_2]
        sd_2 = np.nan
    
    age_diff = np.max(x) - np.min(x) 
    if age_diff < MAX_AGE_DIFF or lx < MIN_POINTS or np.isnan(mean_age) or np.isnan(mean_y) or np.isnan(ref_age).any():
        res = [mean_age, gender, motherTongue, age_diff, lx, mean_y, sd_y, last_y, mean_1, sd_1, mean_2, sd_2] + (degree + 1)*[np.nan] #studentId, scale, 
    else: 
        poly = PolynomialFeatures(degree=degree)
        try:
            polyx = poly.fit_transform(x - ref_age)
        except: 
            print(x)  
            print(ref_age)
            exit()
        reg = LinearRegression(fit_intercept=False)
        reg.fit(polyx, y)
        res = [mean_age, gender, motherTongue, age_diff, lx, mean_y, sd_y, last_y, mean_1, sd_1, mean_2, sd_2] + reg.coef_.ravel().tolist() 
        
    return pd.Series(dict(zip(res_names, res)))

def get_mode(x):
    try:
        z = pd.Series.mode(x)[0]
    except Exception as e: 
        #print("exception", x)
        z = np.nan
    return(z)
