from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_model(model_key: str, random_state: int):
    key = model_key.lower()

    if key == "logreg":
        return LogisticRegression(max_iter=200)

    if key == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_key: {model_key}")
