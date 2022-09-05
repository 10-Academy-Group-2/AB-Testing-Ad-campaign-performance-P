from flask import Flask,jsonify, request
import pickle
from sklearn.model_selection import train_test_split,cross_validate
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

@app.route("/train")
def train():
  
	# Read data
    data = pd.read_csv('../data/AdSmartABdata_platform.csv', sep=",")
	

    # X= data[['hour', 'device_make', 'browser', 'experiment']]
    X= data[['hour', 'device_make', 'platform_os', 'experiment']]
    # Define Y (This is the value we will predict)
    y = data["yes"]
    y

    X_train, X_rem, y_train, y_rem = train_test_split(X,y, 
                                                    train_size=0.7,random_state = 365)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, 
                                                    train_size=0.666,random_state=365)


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



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8088)