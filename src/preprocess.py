# from sklearn.datasets import load_iris
# import pandas as pd

# data = load_iris(as_frame=True)
# df = data.frame

# df.to_csv("data/iris.csv", index=False)

import pandas as pd
from sklearn.model_selection import train_test_split

# Prétraitement
def preprocess(path) : 
    
    df = pd.read_csv(path)
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

