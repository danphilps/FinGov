import sklearn.metrics as metrics

# Performance metrics...
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
    if X.shape[0] != y_test.shape[0]:
      raise TypeError('Bad parameter: X.shape[0] != y_test.shape[0]')
    if y.shape[0] != y_hat.shape[0]:
      raise TypeError('Bad parameter: y_test.shape[0] != y_test_hat.shape[0]')
    if (y.dtype != y_hat.dtype):
      raise TypeError('Bad parameter: y_test.dtypes != y_test_hat.dtypes')

    # F1, precision, recall...  
    prec = precision_score(y_true=y[:], y_pred=y_hat[:])
    rec = recall_score(y_true=y[:], y_pred=y_hat[:])
    f1 = f1_score(y_true=y[:], y_pred=y_hat[:])

    print(prec)

    # ROC Curve
    metrics.plot_roc_curve(mdl, X, y) 
    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
    plt.title ='Credit Decisions ROC Curve'
    plt.show()
    return f1, prec, rec
  
  @staticmethod
  def kpi_review_analyst2(shap_values: np.array):    
    # plot the feature importance
    shap.plots.bar(shap_values=shap_values, max_display=30, show=False)
    plt.title = "Feature Importance: Credit-Use Case Feature Importance"
    plt.show()

    # shap summary plot
    shap.summary_plot(shap_values, show=False)
    plt.title = "Beeswarm: Credit-Use Case Feature Importance and Dependency"
    plt.show()
    return

  # Return shap values for the classifier chosen
  @staticmethod
  def classifier_shap_vals(max_mdl: object,
                            X_test: pd.DataFrame) -> np.array:
    #sanity
    if max_mdl is None:
      raise TypeError('max_mdl is None')

    # Instantiate an explainer object for our chosen classifier...
    if type(max_mdl).__name__ == 'DecisionTreeClassifier':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'GaussianNB':
      explainer = shap.KernelExplainer(max_mdl.predict, X_test.values)
      shap_values = explainer.shap_values(X_test.values)
    elif type(max_mdl).__name__ == 'LogisticRegression':
      explainer = shap.explainers.Permutation(max_mdl.predict, X_test)
      shap_values = explainer(X_test)
    elif type(max_mdl).__name__ == 'MLPClassifier':
      explainer = shap.KernelExplainer(max_mdl.predict, X_test.values)
      shap_values = explainer.shap_values(X_test.values)
    elif type(max_mdl).__name__ == 'RandomForestClassifier':
      explainer = shap.Explainer(max_mdl.predict, X_test)
      shap_values = explainer(X_test)
    elif type(max_mdl).__name__ == 'LinearDiscriminantAnalysis':
      masker = shap.maskers.Independent(data = X_test.values)
      explainer = shap.LinearExplainer(max_mdl, masker = masker)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'QuadraticDiscriminantAnalysis':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'AdaBoostClassifier':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'GradientBoostingClassifier':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'KNeighborsClassifier':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'SVC':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    elif type(max_mdl).__name__ == 'NuSVC':
      explainer = shap.Explainer(max_mdl.predict, X_test.values)
      shap_values = explainer(X_test.values)
    
    return shap_values
