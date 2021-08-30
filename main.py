import hashlib

import lightgbm as lgb
import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# (neptune) create run
run = neptune.init(
    project="<WORKSPACE/PROJECT>",
    name="titanic-training",
    tags=["LightGBM"]
)

# load data
data = pd.read_csv("train.csv")

# (neptune) log feature and target names
target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]

run["data/features_names"] = features
run["data/target_name"] = target

# (neptune) simple features analysis
women = data.loc[data.Sex == 'female']["Survived"]
run["data/analysis/women_survival_rate"] = sum(women)/len(women)

men = data.loc[data.Sex == 'male']["Survived"]
run["data/analysis/men_survival_rate"] = sum(men)/len(men)

# prepare data
data["Cabin"].fillna("unknown", inplace=True)
data["Embarked"].fillna("U", inplace=True)
data[["Sex", "Cabin", "Embarked"]] = OrdinalEncoder().fit_transform(data[["Sex", "Cabin", "Embarked"]])

y = data[target]
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=98734)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# (neptune) log data version
run["data/train/version"] = hashlib.md5(X_train.to_numpy().copy(order="C")).hexdigest()
run["data/test/version"] = hashlib.md5(X_test.to_numpy().copy(order="C")).hexdigest()

# (neptune) log datasets sizes
run["data/train/size"] = len(X_train)
run["data/test/size"] = len(X_test)

# (neptune) log train sample
run["data/raw_sample"].upload(neptune.types.File.as_html(data.head(20)))

# (neptune) create neptune callback
neptune_callback = NeptuneCallback(run=run, base_namespace="training")

# Define parameters
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": ["binary_logloss"],
    "num_leaves": 5,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 12,
}

# (neptune) train the model
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=50,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=["training", "testing"],
    categorical_feature=["Pclass", "Sex", "Cabin", "Embarked"],
    callbacks=[neptune_callback],
)

# (neptune) log summary metadata to the same run under the "lgbm_summary" namespace
y_pred = (gbm.predict(X_test) > 0.5) * 1

run["lgbm_summary"] = create_booster_summary(
    booster=gbm,
    log_trees=True,
    list_trees=[0, 1, 2, 3, 4],
    log_confusion_matrix=True,
    y_pred=y_pred,
    y_true=y_test,
)
