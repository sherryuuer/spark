"""
1, Create feature table and build training dataset for model
2, Modify feature table and use the updated table to create a new version of model
3, Use feature UI to determine how features relate to models
4, Perform batch scoring using automatic feature lookup
"""
from mlflow.tracking.client import MlflowClient
import pandas as pd
import spark
from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def addIdColumn(dataframe, id_column_name):
    """
    Add id column to dataframe
    增加一个id列
    """
    columns = dataframe.columns
    new_df = dataframe.withColumn(
        id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]


def renameColumns(df):
    """
    Rename columns to be compatible with Feature Engineering in UC
    列名修改将空格变更为下划线
    """
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(
            column, column.replace(' ', '_'))
    return renamed_df


# Load dataset
raw_data = spark.read.load(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    format="csv",
    sep=";",
    inferSchema="true",
    header="true"
)


# Run functions on raw data
renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# Drop target column 'quality' as it is not included in the feature table
features_df = df.drop('quality')
print(display(features_df))

# If need Create a new catalog with:
# spark.sql("CREATE CATALOG IF NOT EXISTS ml")
# spark.sql("USE CATALOG ml")
# Or reuse existing catalog:
spark.sql("USE CATALOG ml")
# Create a new schema(db) in the catalog
spark.sql("CREATE SCHEMA IF NOT EXSITS wine_db")
spark.sql("USE SCHEMA wine_db")
# Create a unique table name for each run
# this prevents errors if you run the notebook multiple times
table_name = f"ml.wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# Create the feature table
fe = FeatureEngineeringClient()
# get help for feature engineering client API functions
# help(fe.<functions_name>) ex help(fe.create_table)
fe.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="wine features"
)

# 上面的写法也可以变成先创建，然后再写入
# fe.create_table(
#     name=table_name,
#     primary_keys=["wine_id"],
#     schema=features_df.schema,
#     description="wine features"
# )

# fe.write_table(
#     name=table_name,
#     df=features_df,
#     mode="merge"
# )

# Train a model with Feature Engineering in Unity Catalog
# inference_data_df includes wine_id (primary key), quality (prediction target), and a real time feature
inference_data_df = df.select(
    "wine_id",
    "quality",
    (10 * rand()).alias("real_time_measurement")
)
print(display(inference_data_df))


def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter
    # all feature except primary keys are returned
    # 关于这个方法，参考官网例子，它主要是可以用于从多个表提取数据
    # https://docs.gcp.databricks.com/en/machine-learning/feature-store/train-models-with-feature-store.html
    model_feature_lookups = [FeatureLookup(
        table_name=table_name,
        lookup_key=lookup_key
    )]

    # fe.create_training_set looks up features in model_feature_lookups
    # that match the primary key from inference_data_df
    training_set = fe.create_training_set(
        df=inference_data_df,
        features_lookups=model_feature_lookups,
        label="quality",
        exclude_columns="wine_id"
    )
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set


# Create the train and test datasets
X_train, X_test, y_train, y_test, training_set = load_data(
    table_name, "wine_id")
X_train.head()

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = "ml.wine_db.wine_model"
client = MlflowClient()

try:
    # delete the model if already created
    client.delete_registered_model(model_name)
except:
    None

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
mlflow.sklearn.autolog(log_models=False)


def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    # fit and log model
    with mlflow.start_run() as run:
        rf = RandomForestRegressor(
            max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="wine_quality_prediction",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )


train_model(X_train, X_test, y_train, y_test, training_set, fe)


# Batch scoring
# helper function
def get_latest_model_version(model_name):

    latest_version = 1
    mlflow_client = MlflowClient()

    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int

    return latest_version


# For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("quality")
latest_model_version = get_latest_model_version(model_name)
predictions_df = fe.score_batch(
    model_uri=f"models:/{model_name}/{latest_model_version}",
    df=batch_input_df
)
print(display(predictions_df["wine_id", "prediction"]))

# Modify feature table
# 如果发现了新的feature，可以通过write table和merge mode来更新feature table
# Modify the dataframe containing the features
so2_cols = ["free_sulfur_dioxide", "total_sulfur_dioxide"]
new_features_df = (features_df.withColumn(
    "average_so2", expr("+".join(so2_cols)) / 2))
print(display(new_features_df))

# Update the feature table using fe.write_table with mode="merge"
fe.write_table(
    name=table_name,
    df=new_features_df,
    mode="merge"
)

# Read feature data from the feature tables use fe.read_table()
# Display most recent version of the feature table
# Note that features that were deleted in the current version still appear in the table but with value = null
print(display(fe.read_table(name=table_name)))


# Train a new model version using the updated feature table
def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(
        table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fe.create_training_set(
        df=inference_data_df,
        feature_lookups=model_feature_lookups,
        label="quality",
        exclude_columns="wine_id"
    )
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set


X_train, X_test, y_train, y_test, training_set = load_data(
    table_name, "wine_id")
X_train.head()


# Build a training dataset that will use the indicated key to lookup features.
def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    # fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(
            max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )


train_model(X_train, X_test, y_train, y_test, training_set, fe)

# For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("quality")  # Drop the label column
latest_model_version = get_latest_model_version(model_name)
predictions_df = fe.score_batch(
    model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)
print(display(predictions_df["wine_id", "prediction"]))
