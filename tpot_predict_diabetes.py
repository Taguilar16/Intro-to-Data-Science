import pandas as pd
import pickle
import tpot

def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='Patient number')
    return df


def make_predictions(df):
    """
    Uses the tpot best model to make predictions on data in the df dataframe.
    """
    with open('tpot_diabetes_pipeline.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)
        predictions = loaded_pipeline.predict(df)
    return predictions


if __name__ == "__main__":
    df = load_data('new_diabetes_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
