import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df=pd.read_csv(file_path)

    return df

def split_data(df, target="target"):  
    x=df.drop(columns=[target])
    y=df[target]
    return train_test_split(x, y, test_size=0.2, random_state=42)

