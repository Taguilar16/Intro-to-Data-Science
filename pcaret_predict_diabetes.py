import pandas as pd
import pickle
from pycaret.classification import load_model, predict_model


def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='Patient number')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model("pycaret_model")
    predictions = predict_model(model, data=df)
    return predictions


if __name__ == "__main__":
    df = load_data('new_diabetes_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
