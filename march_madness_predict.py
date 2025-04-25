#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def made_tournament(postseason):
    """
    Returns 1 if the team participated in the NCAA tournament (any stage),
    otherwise 0 (including NIT or no postseason).
    """
    if isinstance(postseason, str) and postseason != "NIT":
        return 1
    else:
        return 0

def train_model(historical_csv):
    """
    1) Loads and cleans historical data (cbb.csv).
    2) Trains and returns a Random Forest model for 'MADE_NCAA'.
    3) Prints an evaluation on a train/test split (for reference).
    """
    # Load historical data
    df = pd.read_csv(historical_csv)
    df.fillna(0, inplace=True)
    
    # Create binary target
    df['MADE_NCAA'] = df['POSTSEASON'].apply(made_tournament)
    
    # Drop columns we don't want as features
    # (TEAM, CONF, POSTSEASON, YEAR — if they exist)
    df.drop(['TEAM', 'CONF', 'POSTSEASON', 'YEAR'], axis=1, inplace=True, errors='ignore')
    
    # Extract features and target
    X = df.drop('MADE_NCAA', axis=1)
    y = df['MADE_NCAA']
    
    # Train/test split to evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Historical test accuracy: {acc:.3f}")
    print("[INFO] Classification Report (historical test set):")
    print(classification_report(y_test, y_pred))
    
    # View Feature Importances
    importances = rf.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    plt.figure()
    plt.title("Feature Importances (Historical Model)")
    plt.bar(range(len(sorted_importances)), sorted_importances, align="center")
    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90)
    plt.tight_layout()
    plt.show()
    
    return rf, X.columns

def predict_this_year(model, new_csv, training_columns):
    """
    Uses the trained model to predict for new data (e.g., 'this_year_cbb.csv').
    Returns a DataFrame with:
       TEAM, PREDICTED_MADE_NCAA, PREDICTED_PROB
    """
    # Load new data
    new_df = pd.read_csv(new_csv)
    new_df.fillna(0, inplace=True)
    
    # Keep a copy of the TEAM column for the output
    teams = new_df['TEAM'].values if 'TEAM' in new_df.columns else None
    
    # Drop the same columns we dropped when training
    drop_cols = ['TEAM', 'CONF', 'POSTSEASON', 'YEAR', 'MADE_NCAA']
    for col in drop_cols:
        if col in new_df.columns:
            new_df.drop(col, axis=1, inplace=True)
    
    # Ensure columns match the training set
    # If new_df is missing any columns from training, fill with 0
    for col in training_columns:
        if col not in new_df.columns:
            new_df[col] = 0
    
    # If the new_df has extra columns not used during training, drop them
    extra_cols = set(new_df.columns) - set(training_columns)
    if extra_cols:
        new_df.drop(list(extra_cols), axis=1, inplace=True)
    
    # Order columns the same as in training
    new_df = new_df[training_columns]
    
    # Predict probabilities and classes
    preds_proba = model.predict_proba(new_df)[:, 1]  # Probability of "1"
    preds_class = model.predict(new_df)
    
    # Build results DataFrame
    results = pd.DataFrame({
        'TEAM': teams,
        'PREDICTED_MADE_NCAA': preds_class,
        'PREDICTED_PROB': preds_proba
    })
    
    # Sort by probability descending
    results.sort_values('PREDICTED_PROB', ascending=False, inplace=True)
    
    return results

def main():
    # File paths (modify as needed)
    historical_csv = "cbb.csv"            # Past data
    this_year_csv  = "this_year_cbb.csv"  # This year's data
    
    # Train a model on historical data
    print("[INFO] Training model on historical data...")
    rf_model, train_cols = train_model(historical_csv)
    
    # Predict on this year's data
    print("[INFO] Predicting for this year's data...")
    this_year_results = predict_this_year(rf_model, this_year_csv, train_cols)
    
    # Print the top 68 teams explicitly with name + predicted probability
    top68 = this_year_results.head(68)
    print("\n[INFO] Top 68 teams most likely to make the NCAA:")
    for idx, row in top68.iterrows():
        print(f"{row['TEAM']}: "
              f"Predicted_Made_NCAA={row['PREDICTED_MADE_NCAA']} "
              f"(Prob={row['PREDICTED_PROB']:.2f})")
    
    # Save the full results to a CSV
    this_year_results.to_csv("this_year_predictions.csv", index=False)
    print("\n[INFO] Full predictions saved to 'this_year_predictions.csv'.")

if __name__ == "__main__":
    main()
