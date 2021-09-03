import hashlib
import pickle

import lightgbm as lgb
import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

base_namespace = "model_finetuning"

# (neptune) fetch project
project = neptune.get_project(name="<WORKSPACE/PROJECT>")

# (neptune) find your best LightGBM run
best_run_df = project.fetch_runs_table(owner="<USERNAME>>", tag="LightGBM").to_pandas()
best_run_df = best_run_df.sort_values(by=["training/testing/binary_logloss"])
best_run_id = best_run_df["sys/id"].values[0]

# (neptune) resume this run
run = neptune.init(
    project="<WORKSPACE/PROJECT>",
    run=best_run_id,
    monitoring_namespace=f"{base_namespace}/monitoring",
)

# (neptune-lightgbm integration) create neptune_callback to track LightGBM finetuning
neptune_callback = NeptuneCallback(run=run, base_namespace=base_namespace)

# (neptune) download model from the run
run["lgbm_summary/pickled_model"].download("lgbm.model")
with open("lgbm.model", "rb") as file:
    lgbm_model = pickle.load(file)

# load data
data = pd.read_csv("train.csv")

# feature and target names
target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]

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
run[base_namespace]["data/train/version"] = hashlib.md5(X_train.to_numpy().copy(order="C")).hexdigest()
run[base_namespace]["data/test/version"] = hashlib.md5(X_test.to_numpy().copy(order="C")).hexdigest()

# (neptune) log datasets sizes
run[base_namespace]["data/train/size"] = len(X_train)
run[base_namespace]["data/test/size"] = len(X_test)

# (neptune) log train sample
run[base_namespace]["data/raw_sample"].upload(neptune.types.File.as_html(data.head(20)))

# (neptune) create neptune callback
neptune_callback = NeptuneCallback(run=run, base_namespace=base_namespace)

# define parameters
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": ["binary_logloss"],
    "num_leaves": 7,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 9,
}

# (neptune) train the model
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=201,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=["training", "testing"],
    categorical_feature=["Pclass", "Sex", "Cabin", "Embarked"],
    callbacks=[neptune_callback],
    init_model=lgbm_model,
)

# (neptune) log summary metadata to the same run under the "lgbm_summary" namespace
y_pred = (gbm.predict(X_test) > 0.5) * 1

run[base_namespace]["lgbm_summary"] = create_booster_summary(
    booster=gbm,
    log_trees=True,
    list_trees=[0, 1, 2, 3, 4],
    log_confusion_matrix=True,
    y_pred=y_pred,
    y_true=y_test,
)

# (neptune) add tag "finetuned" at the end
run["sys/tags"].add("finetuned")
