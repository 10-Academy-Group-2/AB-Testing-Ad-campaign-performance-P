# Importing Pandas an Numpy Libraries to use on manipulating our Data
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_validate
import os
from sklearn.metrics import r2_score, mean_absolute_error,  mean_squared_error as MSE

# To evaluate end result we have
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV
import dvc.api
from  urllib.parse import urlparse
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import logging


import warnings

logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

path='../data/AdSmartABdata_browser.csv'
path1='../data/AdSmartABdata_platform.csv'
repo='/Users/apple/Desktop/AB-Testing-Ad-campaign-performance-P/.git/'
# version='v1'
version='v2'


data_url = dvc.api.get_url(
	path=path1,
	repo=repo,
	rev=version
    )
	
mlflow.set_experiment('mlops-abtest')

def eval_metric(actual, predic):
    rmse = np.sqrt(MSE(actual, predic))
    mae = mean_absolute_error(actual,predic)
    r2 = r2_score(actual,predic)
    return rmse,mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
	# Read data
    data = pd.read_csv(data_url, sep=",")
	
	# Log data params
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('inputs_cols', data.shape[1] )


    # X= data[['hour', 'device_make', 'browser', 'experiment']]
    X= data[['hour', 'device_make', 'platform_os', 'experiment']]
    # Define Y (This is the value we will predict)
    y = data["yes"]
    y

    X_train, X_rem, y_train, y_rem = train_test_split(X,y, 
                                                    train_size=0.7,random_state = 365)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, 
                                                    train_size=0.666,random_state=365)


    # Log an artifact (output file)
    cols_x=pd.DataFrame(list(X_train))
    cols_x.to_csv('features.csv',header=False,index=False)
    mlflow.log_artifact('features.csv')

    cols_x=pd.DataFrame(list(y_train))
    cols_x.to_csv('targert.csv',header=False,index=False)
    mlflow.log_artifact('targert.csv')

    #  initiate the classifier and train the model
    clf=RandomForestClassifier()
    kf = KFold(n_splits=5, shuffle=False)
    cv_results = cross_validate(
        estimator=clf,
        X=X_train,
        y=y_train,
        n_jobs=4,
        cv=kf,
        return_estimator=True,
    )
    print("%0.2f accuracy with a standard deviation of %0.2f" % (
        cv_results['test_score'].mean(),
        cv_results['test_score'].std(),
    ))