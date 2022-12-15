import pandas as pd

# For the CFA notebook we do not want stochastic outcomes from the random selection of the test/train/cv datasets
# We use this function to create deterministic outcomes in the use-case
def train_test_split_deterministic(X: pd.DataFrame, 
                     y: pd.DataFrame, 
                     test_size: float = 0.3, 
                     random_state = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    
  # Split the data at this point.
  train_row_end = int(test_size * X.shape[0])

  X_train = X.iloc[0:train_row_end,:]
  y_train = y.iloc[0:train_row_end]

  X_test = X.iloc[train_row_end+1:,:]
  y_test = y.iloc[train_row_end+1:]
  
  return X_train, X_test, y_train, y_test
  
  
