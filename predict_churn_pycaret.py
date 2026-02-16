import pandas as pd
from pycaret.classification import load_model, predict_model

def predict_churn(df: pd.DataFrame, model_name: str = "churn_pycaret_model") -> pd.DataFrame:
    """
    Takes a pandas dataframe and returns churn predictions:
    - prediction_label: Yes/No
    - prediction_score: probability of churn (for the positive class)
    """
    model = load_model(model_name)
    preds = predict_model(model, data=df)
    return preds[["prediction_label", "prediction_score"]]

if __name__ == "__main__":
    # Load the new data provided by the assignment
    new_df = pd.read_csv("/Users/taniaaguilar/Desktop/Desktop - Tania’s MacBook Air/Regis University/Intro to DS/HW/Week 5/new_churn_data.csv", index_col="customerID")

    # Predict + print results
    results = predict_churn(new_df)
    print(results)
    
