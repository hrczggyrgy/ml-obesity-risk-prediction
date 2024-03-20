import numpy as np

# importing necessary libraries
import optuna
import wandb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from feature_engineering import create_features


train = pd.read_csv("merged_dataset.csv")

train = create_features(train)
# Separate the target variable
X = train.drop(["NObeyesdad", "id"], axis=1)
y = train["NObeyesdad"]

# Identify categorical and numerical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [
    cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
]

# Preprocessing for numerical data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model
rf_model = RandomForestClassifier(random_state=42)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf_model)])

# Encoding the target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Initialize W&B project
wandb.init(project="my-space", entity="herczeg-gyrgy", sync_tensorboard=True)

# Callback for logging Optuna optimization to W&B
def optuna_callback(study, trial):
    wandb.log({"Best Value": study.best_value, "Current Value": trial.value})


# Preprocessing for categorical data
categorical_cols = [
    cname
    for cname in X.columns
    if X[cname].dtype == "object" or X[cname].dtype.name == "category"
]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

# Configurations for hyperparameter optimization with W&B and Optuna integration
config = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": len(np.unique(y_train)),
}

wandb.config.update(config)

# Defining the objective function for Optuna study with WandB logging
def objective(trial):
    param = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "nthread": -1,
        "num_class": len(np.unique(y_train)),
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "max_leaves": trial.suggest_int("max_leaves", 32, 512),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "use_label_encoder": False,
    }

    wandb.config.update(param)  # Updating config in WandB with each trial's parameters

    X_preprocessed = preprocessor.fit_transform(X_train)  # Fixed line
    clf = xgb.XGBClassifier(**param, enable_categorical=True)

    score = cross_val_score(clf, X_preprocessed, y_train, cv=5, n_jobs=-1).mean()

    wandb.log({"score": score})  # Log the score

    return score


# Running Optuna optimization with W&B integration
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, callbacks=[optuna_callback])

# Printing best trial info
best_trial = study.best_trial

print(f"Best trial score: {best_trial.value}")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# Close the W&B run
wandb.finish()