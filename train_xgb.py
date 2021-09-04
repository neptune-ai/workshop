import hashlib

import neptune.new as neptune
import pandas as pd
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# (neptune) create run
run = neptune.init(
    project="<WORKSPACE/PROJECT>",
    name="titanic-training",
    tags=["XGBoost"],
)

# load data
data = pd.read_csv("train.csv")

# (neptune) log feature and target names
target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
categorical_feature = ["Pclass", "Sex", "Cabin", "Embarked"]

run["data/target_name"] = target
run["data/features_names"] = features
run["data/categorical_features_names"] = categorical_feature

# (neptune) simple features analysis
women = data.loc[data.Sex == 'female']["Survived"]
run["data/analysis/women_survival_rate"] = sum(women)/len(women)

men = data.loc[data.Sex == 'male']["Survived"]
run["data/analysis/men_survival_rate"] = sum(men)/len(men)

# encode categorical features (OHE)
enc = OneHotEncoder(sparse=False)
enc_data = enc.fit_transform(data[categorical_feature])

data_cat = pd.DataFrame(enc_data)
data_num = data[list(set(features) - set(categorical_feature))]

X = pd.concat([data_num, data_cat], axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=98734)

# (neptune) log data version
run["data/train/version"] = hashlib.md5(X_train.to_numpy().copy(order="C")).hexdigest()
run["data/test/version"] = hashlib.md5(X_test.to_numpy().copy(order="C")).hexdigest()

# (neptune) log datasets sizes
run["data/train/size"] = len(X_train)
run["data/test/size"] = len(X_test)

# (neptune) log train sample
run["data/raw_sample"].upload(neptune.types.File.as_html(data.head(20)))

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)

# (neptune) create neptune callback
neptune_callback = NeptuneCallback(run=run, base_namespace="training", log_tree=[0, 1, 2, 3])

# define parameters
model_params = {
    "eta": 0.7,
    "gamma": 0.001,
    "max_depth": 5,
    "objective": "binary:logistic",
    "eval_metric": ["logloss"],
}

evals = [(dtrain, "train"), (dval, "valid")]
num_round = 57

# (neptune) train the model
xgb.train(
    params=model_params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evals,
    callbacks=[neptune_callback],
)
