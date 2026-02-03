from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def build_model(model_key: str, random_state: int):
    k = model_key.lower().strip()

    if k == "logreg":
        return LogisticRegression(max_iter=500)

    if k == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        )

    if k == "gb":
        return GradientBoostingClassifier(random_state=random_state)

    if k == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if k == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_key: {model_key}")
