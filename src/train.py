"""Model definitions, hyperparameter search spaces, and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor
from src.utils import RANDOM_STATE, MODELS_DIR


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict
    search_type: str = "grid"  # "grid" or "random"
    n_iter: int = 20


def get_model_specs() -> list[ModelSpec]:
    """Defines the candidate models and their search spaces for the comparison stage."""
    logistic_regression = ModelSpec(
        name="logistic_regression",
        estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid={
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        search_type="grid",
    )

    random_forest = ModelSpec(
        name="random_forest",
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": randint(3, 20),
            "classifier__min_samples_split": randint(2, 15),
            "classifier__min_samples_leaf": randint(1, 10),
            "classifier__max_features": ["sqrt", "log2", None],
        },
        search_type="random",
        n_iter=30,
    )

    xgboost = ModelSpec(
        name="xgboost",
        estimator=XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        param_grid={
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": randint(2, 10),
            "classifier__learning_rate": uniform(0.01, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        },
        search_type="random",
        n_iter=30,
    )

    return [logistic_regression, random_forest, xgboost]


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_majority_baseline() -> DummyClassifier:
    """Always predicts the majority class. The floor any real model must clear."""
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)


def build_credit_history_only_pipeline() -> Pipeline:
    """Logistic Regression using only `credit_history`, isolating how much of the
    full model's performance comes from the one feature that dominates it."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("credit_history", SimpleImputer(strategy="most_frequent"), ["credit_history"]),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE)),
        ]
    )


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> GridSearchCV | RandomizedSearchCV:
    """Runs the appropriate hyperparameter search and returns the fitted search object."""
    pipeline = build_pipeline(spec.estimator)

    if spec.search_type == "grid":
        search = GridSearchCV(
            pipeline,
            param_grid=spec.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=spec.param_grid,
            n_iter=spec.n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    search.fit(X_train, y_train)
    return search


def save_model(model: Any, name: str) -> str:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
