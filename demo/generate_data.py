"""
Generate synthetic reference and production datasets for demo purposes.
"""
import numpy as np
import pandas as pd


def make_classification_data(n_ref: int = 1000, n_prod: int = 500, drift: bool = True, seed: int = 42):
    rng = np.random.default_rng(seed)
    features = ["age", "income", "score", "tenure", "category"]

    ref = pd.DataFrame({
        "age": rng.normal(35, 10, n_ref).clip(18, 80),
        "income": rng.lognormal(10.5, 0.5, n_ref),
        "score": rng.uniform(300, 850, n_ref),
        "tenure": rng.exponential(3, n_ref).clip(0, 20),
        "category": rng.choice(["A", "B", "C"], n_ref, p=[0.5, 0.3, 0.2]),
    })

    if drift:
        prod = pd.DataFrame({
            "age": rng.normal(42, 12, n_prod).clip(18, 80),       # age distribution shifted
            "income": rng.lognormal(10.2, 0.7, n_prod),            # income distribution changed
            "score": rng.uniform(300, 850, n_prod),                 # no drift
            "tenure": rng.exponential(5, n_prod).clip(0, 20),      # tenure shifted
            "category": rng.choice(["A", "B", "C"], n_prod, p=[0.3, 0.4, 0.3]),  # category drift
        })
    else:
        prod = pd.DataFrame({
            "age": rng.normal(35, 10, n_prod).clip(18, 80),
            "income": rng.lognormal(10.5, 0.5, n_prod),
            "score": rng.uniform(300, 850, n_prod),
            "tenure": rng.exponential(3, n_prod).clip(0, 20),
            "category": rng.choice(["A", "B", "C"], n_prod, p=[0.5, 0.3, 0.2]),
        })

    # Simulate predictions: degrade over time if drift
    def predict(df, noise=0.1):
        score = (df["score"] - 300) / 550 + rng.normal(0, noise, len(df))
        return (score > 0.5).astype(int)

    ref_labels = predict(ref, noise=0.05)
    ref_preds = predict(ref, noise=0.12)
    prod_labels = predict(prod, noise=0.05)
    prod_preds = predict(prod, noise=0.35 if drift else 0.12)  # degrade predictions if drift

    return ref, prod, ref_labels.values, ref_preds.values, prod_labels.values, prod_preds.values


if __name__ == "__main__":
    ref, prod, rl, rp, pl, pp = make_classification_data(drift=True)
    ref.to_csv("demo_reference.csv", index=False)
    prod.to_csv("demo_production.csv", index=False)
    pd.DataFrame({"true": rl, "pred": rp}).to_csv("demo_ref_predictions.csv", index=False)
    pd.DataFrame({"true": pl, "pred": pp}).to_csv("demo_prod_predictions.csv", index=False)
    print("Demo data generated.")
