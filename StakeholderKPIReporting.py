# package for working with tabular data
import pandas as pd 
import numpy as np

# Package for charting
import matplotlib.pyplot as plt
import seaborn as sns #charts

# package for working with tabular data
import pandas as pd 
import numpy as np

# Package for charting
import matplotlib.pyplot as plt
import seaborn as sns #charts

import sklearn.metrics as metrics

# Performance metrics...
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import warnings
import shap

class StakeholderKPIReporting():

  @staticmethod
  def kpi_review_customer_business_compliance(mdl: object,
                                    X_test: pd.DataFrame,
                                    y_test: pd.DataFrame,
                                    y_hat: pd.DataFrame) -> (float, float):   
      '''
      Args:
        mdl: sklearn classifier model object
        X_test: X variables, columns are features, rows are instances
        y_test: actual target variable {1,0}
        y_hat: prediction of target
        
      Returns:
        Accuracy, Precision

      '''
      
      print(f"Accuracy train: {mdl.score(X_test,y_test):.4f}, cross-validation: ",
        f"{mdl.score(X_test,y_test):.4f}")
      print(f"Precision train: {precision_score(y_test, y_hat, average=None)[1]:.4f}, cross-validation: ",
        f"{precision_score(y_test,y_hat, average=None)[0]:.4f}")
      
      return mdl.score(X_test,y_test), precision_score(y_test,y_hat, average=None)[1]  

  # Analyst KPI: ROC Curve, f1, precision and accuracy of y_hat from a classifier
  # Compute micro-average ROC curve and ROC area
  @staticmethod
  def kpi_review_analyst(mdl: object,
                        X: np.array, 
                        y: np.array,
                        y_hat: np.array) -> (float, float, float):    
    '''
    Args:
        mdl: sklearn classifier model object
        X: X variables, columns are features, rows are instances
        y: actual target variable {1,0}
        y_hat: prediction of target
        
      Returns:
        f1, prec, rec: F1 score, precision, recall.

      '''

    #sanity
    if X.shape[0] != y.shape[0]:
      raise TypeError('Bad parameter: X.shape[0] != y.shape[0]')
    if y.shape[0] != y_hat.shape[0]:
      raise TypeError('Bad parameter: y_test.shape[0] != y_hat.shape[0]')
    #if (y.dtypes != y_hat.dtypes):
    #  raise TypeError('Bad parameter: y_test.dtypes != y_hat.dtypes')

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      # F1, precision, recall...  
      prec = precision_score(y_true=y[:], y_pred=y_hat[:])
      rec = recall_score(y_true=y[:], y_pred=y_hat[:])
      f1 = f1_score(y_true=y[:], y_pred=y_hat[:])

      print(prec)

      # ROC Curve
      metrics.plot_roc_curve(mdl, X, y) 
      fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
      plt.title('Credit Decisions ROC Curve')
      plt.show()
    return f1, prec, rec
  
  @staticmethod
  def kpi_review_analyst2(shap_values: np.array):    
    # plot the feature importance
    shap.plots.bar(shap_values=shap_values, max_display=30, show=False)
    plt.title("Feature Importance: Credit-Use Case Feature Importance")
    plt.show()

    # shap summary plot
    shap.summary_plot(shap_values, show=False)
    plt.title("Beeswarm: Credit-Use Case Feature Importance and Dependency")
    plt.show()
    return

  # Return shap values for the classifier chosen
  @staticmethod
  def classifier_shap_vals(max_mdl: object,
                              X_test: pd.DataFrame,
                              X_test_protected: pd.DataFrame,
                              speed_up: bool = True) -> np.array:
    #sanity
    if max_mdl is None:
      raise TypeError('max_mdl is None')
    if X_test_protected.shape[0] != X_test.shape[0]:
      raise TypeError('X_test_protected.shape[0] != X_test.shape[0]')

    #Ignore warnings...
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      # Speed up...
      if speed_up:
        if X_test.shape[0] > 50:
          # create a list of randomly picked indices, one for each row
          size_of_data = 25
          # create a list of randomly picked indices, one for each row
          idx_bool = np.full((X_test.shape[0]), False)
          idx = np.random.randint(size_of_data, size=X_test.shape[0])
          # replace "False" by "True" at given indices
          idx_bool[idx] = True

          # Reduced set to run shap on
          X_test = X_test.loc[idx_bool].copy(deep=True)
          X_test_protected_reduced = X_test_protected.loc[idx_bool]

      # Instantiate an explainer object for our chosen classifier...
      if type(max_mdl).__name__ == 'DecisionTreeClassifier': # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'KNeighborsClassifier': # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'GaussianNB': # GOOD
        explainer = shap.Explainer(max_mdl.predict,X_train)
        shap_values = explainer(X_train)
      elif type(max_mdl).__name__ == 'LogisticRegression':  # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'MLPClassifier': # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'RandomForestClassifier': # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'LinearDiscriminantAnalysis': # GOOD
        masker = shap.maskers.Independent(data = X_test)
        explainer = shap.LinearExplainer(max_mdl, masker = masker)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'QuadraticDiscriminantAnalysis': # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test.values)
      elif type(max_mdl).__name__ == 'AdaBoostClassifier':  # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'GradientBoostingClassifier':
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'SVC':  # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)
      elif type(max_mdl).__name__ == 'NuSVC':  # GOOD
        explainer = shap.Explainer(max_mdl.predict, X_test)
        shap_values = explainer(X_test)

      return shap_values, explainer, X_test, X_test_protected_reduced
