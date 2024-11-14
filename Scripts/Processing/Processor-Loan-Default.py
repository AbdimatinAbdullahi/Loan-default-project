import os
import sklearn
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--output-train", type=str)
    parser.add_argument("--output-validation", type=str)
    parser.add_argument("--output-test", type=str)
    args = parser.parse_args()

    input_path = os.path.join(args.input_data, "Loan_default.csv")
    df = pd.read_csv(input_path)
    
    # Performing Label Encoding
    educ_en = LabelEncoder()
    emplysts = LabelEncoder()
    mrtlsts = LabelEncoder()
    hsmortg = LabelEncoder()
    hsdpt = LabelEncoder()
    lprps = LabelEncoder()
    hscsgnr = LabelEncoder()
    
    df["education"] = educ_en.fit_transform(df["Education"])
    df["employmentType"] = emplysts.fit_transform(df["EmploymentType"])
    df["MaritalStatus"] = mrtlsts.fit_transform(df["MaritalStatus"])
    df["hasMortgage"] = hsmortg.fit_transform(df["HasMortgage"])
    df["hasDependents"] = hsdpt.fit_transform(df["HasDependents"])
    df["loanPurpose"] = lprps.fit_transform(df["LoanPurpose"])
    df["hasCoSigner"] = hscsgnr.fit_transform(df["HasCoSigner"])
    
    # Dropping columns
    cols_to_drop = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner", "LoanID"]
    df.drop(cols_to_drop, axis=1, inplace=True)
    cols = ["Default"] + [col for col in df.columns if col != "Default"]

    # Spliting the datasets into Train, Validation and Test
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validation, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Saving the datasets to specified Location
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_validation, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)

    train.to_csv(os.path.join(args.output_train, "train.csv"), index=False)
    validation.to_csv(os.path.join(args.output_validation, "validation.csv"), index=False)
    test.to_csv(os.path.join(args.output_test, "test.csv"), index=False)


if __name__ == "__main__":
    main()