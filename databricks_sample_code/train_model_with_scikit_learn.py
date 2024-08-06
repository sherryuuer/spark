"""
1, Visualize the data
2, Run a parallel hyperparameter sweep to train multiple models
3, Explore hyperparameter sweep results with MLflow
4, Register the best performing model in MLflow
5, Apply the registered model to another dataset using a Spark UDF
"""
from pyspark.sql.functions import struct
import xgboost as xgb
import mlflow.xgboost
from math import exp
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
import time
import spark
import cloudpickle
import sklearn
import numpy as np
import mlflow.sklearn
import mlflow.pyfunc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_registry_uri("databricks-uc")
CATALOG_NAME = "main"
SCHEMA_NAME = "default"

# Read data
white_wine = pd.read_csv(
    "/dbfs/databricks-datasets/wine-quality/winequality-white.csv",
    sep=";"
)
red_wine = pd.read_csv(
    "/dbfs/databricks-datasets/wine-quality/winequality-red.csv",
    sep=";"
)

# Merge data
red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
print(data.head)

# Visualize data
sns.displot(data.quality, kde=False)  # check the quality col

# Define quality to become 1 or 0
high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# Box plots 对于发现特征和二元分类（比如上面的高质量低质量指标）的相关性很有用
dims = (3, 4)

f, axes = plt.subplot(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
    if col == 'is_red' or col == 'quality':
        continue  # box plots cannot be used on indicator variables
    sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
    axis_j += 1
    if axis_j == dims[1]:
        axis_i += 1
        axis_j = 0

# Preprocess data
# Check for missing values
print(data.isna().any())

# Split data
X = data.drop(["quality"], axis=1)
y = data.quality
X_train, X_rem, y_train, y_rem = train_test_split(
    X,
    y,
    train_size=0.6,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rem,
    y_rem,
    test_size=0.5,
    random_state=42
)


# Train a baseline model: random forest model
# Wrapper model to return the probability of the class
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


with mlflow.start_run(run_name='untuned_random_forest'):
    n_estimators = 10
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=np.random.RandomState(123)
    )
    model.fit(X_train, y_train)

    # Predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_param('n_estimators', n_estimators)
    # Use the area under the ROC curve as a metric
    mlflow.log_metric('auc', auc_score)
    wrappedModel = SklearnModelWrapper(model)
    # Log the model with a signature that defines the schema of the model's inputs and outputs
    # When the model is deployed, this signature will be used to validate inputs
    # infer_signature is a function provided by MLflow
    # This function is used to automatically determine the schema of the input and output of a model
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

    # MLflow contains utilities to create a conda environment used to serve models
    # The necessary dependencies are added to a conda.yaml file which is logged along with the model
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__),
            "scikit-learn=={}".format(sklearn.__version__)
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        "random_forest_model",
        python_model=wrappedModel,
        conda_env=conda_env,
        signature=signature
    )

# Review the learned feature importances output by the model -> get col alcohol and density are importance
feature_importances = pd.DataFrame(
    model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# Register the model to Unity Catalog by programming or  UI
run_id = mlflow.search_runs(
    filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_quality"
model_version = mlflow.register_model(
    f"runs:/{run_id}/random_forest_model", model_name)
# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# Next, assign this model the "Champion" tag=alias, and load it into this notebook from Unity Catalog.
client = MlflowClient()
client.set_registered_model_alias(
    model_name,
    "Champion",
    model_version.version)

# Load the model by tag
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")
# Check the AUC
print(f"AUC: {roc_auc_score(y_test, model.predict(X_test))}")

# Experiment with a new model: xgboost
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, -3),
    'objective': 'binary:logistic',
    'seed': 42,  # Sed a seed for deterministic training
}


def train_model(params):
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow
    mlflow.xgboost.autolog()
    with mlflow.start_run(nested=True):
        train = xgb.DMatrix(data=X_train, label=y_train)
        validation = xgb.DMatrix(data=X_val, label=y_val)
        # Pass in the validation set so xgb can track an evaluation metric
        # XGBoost terminates training when the evaluation metric
        # is no longer improving
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            eval=[(validation, "validation")],
            early_stopping_rounds=50
        )
        validation_predictions = booster.predict(validation)
        auc_score = roc_auc_score(y_val, validation_predictions)
        mlflow.log_metric('auc', auc_score)

        signature = infer_signature(X_train, booster.predict(train))
        mlflow.xgboost.log_model(booster, "model", signature=signature)

        # Set the loss to -1*auc_score so fmin maximizes the auc_score
        return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}


# 更高的并行运算参数会提速，但是会降低超参搜索的质量，合理的并行数值是max_evals的平方根
# max_evals 是希望执行的超参数组合的总数目，将并行数设置为 max_evals 的平方根可以确保每个并行单元在合理时间内覆盖足够的搜索空间
spark_trials = SparkTrials(parallelism=10)

# Hyperopt 是一个用于超参数优化的 Python 库，它支持随机搜索和贝叶斯优化
# Hyperopt 的 fmin 函数用于在指定搜索空间内寻找目标函数的最小值
# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# Run called "xgboost_models"
with mlflow.start_run(run_name='xgboost_models'):
    best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=96,
        trials=spark_trials,
    )

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')
# 这里刷新了之前定义的model_name
new_model_version = mlflow.register_model(
    f"runs:/{best_run.run_id}/model",
    model_name
)
time.sleep(15)

# Assign the "Champion" alias to the new version
client.set_registered_model_alias(
    model_name,
    "Champion",
    new_model_version.version
)
# Client that call the load_model using the "Champion" alias now get the new model
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")
# The new model will achive a high score
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')


# Batch inference
# Evaluate the model on data stored in a Delta table, using Spark to run the computation in parallel
# To simulate a new corpus of data, save the existing X_train data to a Delta table
# In the real world, this would be a new batch fo data
spark_df = spark.createDataFrame(X_train)
table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_data"

(spark_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(table_name)
 )

# Load the model into a Spark UDF, so it can be applied to the Delta table
apply_model_udf = mlflow.pyfunc.spark_udf(
    spark, f"models:/{model_name}@Champion")

# Read the "new data" from the Unity Catalog table
new_data = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_data")

# Apply the model to the new data
# 构造成struct，{feature1, feature2, feature3}
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
    "prediction",
    apply_model_udf(udf_inputs)
)

# The xgboost function does not output probabilities by default
# so the predictions are not limited to the range [0, 1]
print(display(new_data))
