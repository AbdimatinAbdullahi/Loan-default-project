import os
import tarfile
import boto3
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to load the model from s3

def load_model_from_s3(model_dir):

    print("Extracting the model")
    model_path = os.path.join(model_dir, "model.tar.gz")
    # Extract model.tar.gz file
    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_dir)
    
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def main():

    # Data from Evaluation Jobs
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-data", type=str)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--output-evaluation", type=str)
    args = parser.parse_args()

    print("Loading the data")
    df = pd.read_csv(os.path.join(args.validation_data, "validation.csv"))
    print(df.head(1))
    X_test = df.drop("Default", axis=1)
    y_test = df["Default"]

    # Load the train model
    model = load_model_from_s3(model_dir=args.model_dir)

    # Make Predictions on Test data
    predictions = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="binary")  # Adjust for multiclass if needed
    recall = recall_score(y_test, predictions, average="binary")
    f1 = f1_score(y_test, predictions, average="binary")


    evaluation_metrics = {
        "metrics": [
            {"name": "accuracy", "value": accuracy},
            {"name": "precision", "value": precision},
            {"name": "recall", "value": recall},
            {"name": "f1_score", "value": f1}  # Changed "F1-Score" to "f1_score" for consistency
        ]
    }


    os.makedirs(args.output_evaluation, exist_ok=True) # Create the output directory of it doesnt exist
    evaluation_path = os.path.join(args.output_evaluation, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation_metrics, f)
    
    print(f"Accuracy: {accuracy}, Recall: {recall}, f1: {f1}, Precision: {precision}")
    print(f"Model evaluation metrics: {evaluation_metrics}")

if __name__ == "__main__":
    main()