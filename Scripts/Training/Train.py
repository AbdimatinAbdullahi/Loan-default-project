import os
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation-data", default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--model-dir", default=os.environ["SM_MODEL_DIR"])
    args = parser.parse_args()


    # Loading the data
    train_data = pd.read_csv(os.path.join(args.train_data, "train.csv"))
    validation_data = pd.read_csv(os.path.join(args.validation_data, "validation.csv"))

    # Extracting information from the data
    X_train = train_data.drop("Default", axis=1)
    y_train = train_data["Default"]
    X_val = validation_data.drop("Default", axis=1)
    y_val = validation_data["Default"]

    # Training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluating the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_val)
    precision = precision_score(y_pred=y_pred, y_true=y_val)
    recall = recall_score(y_pred=y_pred, y_true=y_val)
    print(f"Evaluation metrics received: Recall: {recall}, Precision: {precision}, Accuracy: {accuracy}")

    # How to save the above metrics so that will be later used for deployment stage?
    
    # Saving the trained model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()