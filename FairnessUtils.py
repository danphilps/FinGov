# package for working with tabular data
import pandas as pd 
import numpy as np

# Package for charting
import matplotlib.pyplot as plt
import seaborn as sns #charts
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import math

import warnings

class FairnessUtils():

  # Fairness: run the model on different groups, and get precision, accuracy, f1 and so on, for each model run/group
  # Calcluates these stats for the predictions of a trained model (mod) for each category 
  # in a given column (category_col_name) in the data set (X_test). 
  @staticmethod
  def fairness_stats_get (mod: object, 
                        X_test: pd.DataFrame, 
                        y_test: pd.DataFrame, 
                        X_test_category_col: pd.DataFrame,
                        y_approval_threshold: float = 0.5) -> pd.DataFrame:
    
    '''    
    Args:
        mod: sklearn model, trained without the category_col_name, and ready to test for biases.
        X_test: X data, including the category_col_name you want to examine 
        y_test: y data, including the category_col_name you want to examine
        X_test_category_col: column - corresponding to X_test and y_test in which categories are contained we want to test for fairness
        y_approval_threshold: We are forecasting the probability of default (0=0% probability of default, 1=100% probability of default), this is the probability threshold over which we offer loans
        
    Returns:
        df_stats: record of the accuracy (etc) of the model on each category. Examine this for fairness...###
    Author:
      Madhu Nagarajan, Dan Philps
    '''

    #Sanity
    if mod is None:
      raise TypeError('mod has not been instantiated or trained')
    if X_test.shape[0] != y_test.shape[0]:
      raise TypeError('X_test.shape[0] != y_test.shape[0]')
    if X_test.shape[0] != X_test_category_col.shape[0]:
      raise TypeError('X_test.shape[0] != X_test_category_col.shape[0]:')
    if (y_approval_threshold < 0) | (y_approval_threshold > 1):
      raise TypeError('(approval_threshold < 0) | (approval_threshold > 1)')

    # Ini
    df_stats = pd.DataFrame()
    stats_cols = []

    # Get categories in our test column
    categories = pd.Series(X_test_category_col).unique()

    # Type conversion
    #if type(X_test_category_col) != pd.Series:
    X_test_category_col = pd.Series(X_test_category_col)
    X_test_category_col = X_test_category_col.values
    
    # Loop through each of the categories in the category_col
    # Eg male (=0) and female (=1)
    # test acccuracy/precision/recall for each cat
    for cat in categories:

        #Filter on the cat
        cat_rows = (X_test_category_col == cat)
        #X...  
        X_test_cat = X_test.loc[cat_rows]
        #y...
        y_test_cat = y_test.loc[cat_rows]      
        
        # Predict the probability of default, and decide who to offer credit to, for specific population groups
        y_test_cat_hat_pred_proba = mod.predict_proba(X_test_cat.values)
        y_test_cat_hat = (y_test_cat_hat_pred_proba[:,0] < y_approval_threshold).astype('int')

        # Calc and record fairness analytics for each cat, record in df_stats
        TN, FP, FN, TP = confusion_matrix(y_test_cat, y_test_cat_hat, labels = [0,1]).ravel()
        fnr = FN/(FN+TP)
        fdr = FP/(FP+TP)
        fpr = FP/(FP+TN)
        npv = TN/(TN+FN)

        cat_row = pd.Series({'cat': cat,
                            'cat_proportion': np.divide(float(X_test_cat.shape[0]),float(X_test.shape[0])), 
                            'accuracy': accuracy_score(y_true=y_test_cat, y_pred=y_test_cat_hat),  
                            'precision': precision_score(y_true=y_test_cat, y_pred=y_test_cat_hat), 
                            'recall': recall_score(y_true=y_test_cat, y_pred=y_test_cat_hat), 
                            'fnr': fnr, 'fdr': fdr, 'fpr': fpr, 'npv': npv})

        # Build record of accuracy and so on, of each category in the category_col
        df_stats = pd.concat([df_stats, cat_row], axis=1)
        stats_cols.append(cat)
      
      # Predict the probability of default, and decide who to offer credit to, for all data
    y_test_pred_proba = mod.predict_proba(X_test.values)
    y_test_hat = (y_test_pred_proba[:,0] < y_approval_threshold).astype('int')

    # Calc and record fairness analytics for each cat, record in df_stats
    TN, FP, FN, TP = confusion_matrix(y_test, y_test_hat, labels = [0,1]).ravel()
    fnr = FN/(FN+TP)
    fdr = FP/(FP+TP)
    fpr = FP/(FP+TN)
    npv = TN/(TN+FN)

    cat_row = pd.Series({'cat': "All",
                        'cat_proportion': 1, 
                        'accuracy': accuracy_score(y_true=y_test, y_pred=y_test_hat),  
                        'precision': precision_score(y_true=y_test, y_pred=y_test_hat), 
                        'recall': recall_score(y_true=y_test, y_pred=y_test_hat), 
                        'fnr': fnr, 'fdr': fdr, 'fpr': fpr, 'npv': npv})

    # Build record of accuracy and so on, of each category in the category_col
    df_stats = pd.concat([df_stats, cat_row], axis=1)
    stats_cols.append(cat)      

    # Set up df_stats with column names and an index
    df_stats = df_stats.transpose()
    df_stats.columns = cat_row.index
    df_stats = df_stats.set_index(df_stats['cat'])

    return df_stats

  # Find the best threshold for 
  @staticmethod
  def decision_threshold_find_optimal(mod: object, 
                        X_test: pd.DataFrame, 
                        y_test: pd.DataFrame, 
                        X_test_category_col: pd.DataFrame,
                        majority_class: str = 'male',
                        fairness_metric: str = 'precision',
                        threshold_metric: str = 'recall',
                        threshold_min_max: list = [50,80],              
                        show_charts: bool = True) -> float:   
    ''' 
    Args:
        mod: sklearn model, trained without the category_col_name, and ready to test for biases.
        X_test: X data, including the category_col_name you want to examine 
        y_test: y data, including the category_col_name you want to examine
        X_test_category_col: pd.DataFrame,
        majority_class: string - the value of the majority class against which the other population groups are compared with (e.g. ["Male"])
        fairness_metric: array with the fairness metrics to compare e.g. ["precision"]
        threshold_metric: Lender metric to optimize on...
        show_charts: ...
    Returns:
        (A plot charting the fairness metric values to the various population groups...)
        optimal_threshold: a float with value showing the "best" cut off for threshold_metric, while satifying fairness_metric based on the category X_test_category_col and the majority_class
    
    Author:
      Madhu Nagarajan, Dan Philps
    '''
    
    # Sanity
    if type(threshold_min_max) != list:
      raise TypeError('type(threshold_min_max) != list')
    if len(threshold_min_max) != 2:
      raise TypeError('threshold_min_max needs 2 elements: from and to')
    if threshold_min_max[0] > threshold_min_max[1]:
      raise TypeError('threshold_min_max[0] > threshold_min_max[1]')
    if threshold_min_max[1] <=1:
      raise TypeError('threshold_min_max[1] should be in the 0-100 range not 0-1')
        
    # Ini
    high_threshold = -999
    high_maximization_metric = -999
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # To supress warnings in notebooks - remove this if running in any other context
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
    
      #Try with multiple threshold values from 0.5 to 1.0.
      df_stats_per_iteration = None
      for a_threshold in range (30, 100, 1):
          fair_model = True

          #get the model metrics for a speicific threshold values
          df_stats = FairnessUtils.fairness_stats_get (mod, X_test, y_test, X_test_category_col, a_threshold/100)

          # record results for each iteration
          if df_stats_per_iteration is None:
            df_stats_per_iteration = pd.DataFrame(np.zeros((0,df_stats.shape[0])))
            df_stats_per_iteration.columns = df_stats.index
            df_cats_per_iteration = None

          #get the metric to compare for the majority class (e.g. Male)
          majority_class_metric  = df_stats.loc[df_stats["cat"] == majority_class, fairness_metric].astype('float64')[0]
          compare_metric = 0
                    
          majority_class_metric_threshold  = df_stats.loc[df_stats["cat"] == majority_class, threshold_metric].astype('float64')[0]
          compare_metric_threshold = 0

          # Store values for the charts later
          if df_cats_per_iteration is None:
            df_cats_per_iteration = pd.DataFrame(np.zeros((0,df_stats['cat'].shape[0])))
            df_cats_per_iteration.columns = df_stats['cat'].values

          #Iterate through the various values for the selected group
          cats_per_iteration = None
          for cat in df_stats['cat'].values:
              #ignore the category values of All and the majority class. obtain the fairness metric for the other population groups
              fairness_val = df_stats.loc[df_stats["cat"]==cat][fairness_metric].astype('float64')[0]
              threhold_val = df_stats.loc[df_stats["cat"]==cat][fairness_metric].astype('float64')[0]

              if cat not in ["All", majority_class]:
                #Ensure the metric for all non majority classes are within limits, one sided ensures that the non majority classes are not worse off
                if (majority_class_metric * 0.8 > fairness_val):  
                  if (majority_class_metric_threshold * 0.8 > threhold_val):  

                    #if any metric is below limit, then set the model as not fair
                    fair_model = 'False'
                    #and try the next threshold

              # record results...
              if cats_per_iteration is None:
                cats_per_iteration = list()
              cats_per_iteration.append(fairness_val)

          # metric to maximize!
          current_maximization_metric = df_stats.loc[df_stats["cat"]=="All"][threshold_metric].astype('float64')[0]

          #if the model is found fair for all population groups (other than the majority one), then check if the model has a higher maximization metric. if so save the threshold value
          if fair_model == True:
              # Only save the highest within the bounds specified:
              if (a_threshold > threshold_min_max[0]) & (a_threshold < threshold_min_max[1]):
                if current_maximization_metric > high_maximization_metric:
                    high_maximization_metric = current_maximization_metric
                    high_threshold = a_threshold

          # record results...
          df_stats_per_iteration.loc[a_threshold] = df_stats[threshold_metric].T.values
          df_cats_per_iteration.loc[a_threshold] = cats_per_iteration

      if high_maximization_metric > 0:
          df_stats = FairnessUtils.fairness_stats_get (mod, X_test, y_test, X_test_category_col, high_threshold/100)
          FairnessUtils.plot_fairness_charts(df_stats, majority_class, fairness_metric, threshold_metric)
          opt_threshold = high_threshold/100
      else:
          opt_threshold = np.nan

      # show training curve
      if show_charts:
        # map colors...
        def get_cmap(n, name='hsv'):
          '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
          RGB color; the keyword argument name must be a standard mpl colormap name.'''
          return plt.cm.get_cmap(name, n)
        cmap = get_cmap(10)  

        plt.title('Learning curve for Credit Approvals Model: (threshold_metric:' + threshold_metric + ', while monitoring fairness metric ' +  fairness_metric +  ')')
        plt.figure(figsize=(15,10))
        for j in range(df_cats_per_iteration.shape[1]): 
          Y_val = df_cats_per_iteration.iloc[:,j].values
          # Add some text for labels, title and custom x-axis tick labels, etc.
          plt.plot(df_cats_per_iteration.index, Y_val, label='Fairness: ' + df_cats_per_iteration.columns[j], color=cmap(j))

        # each col
        for j in range(df_stats_per_iteration.shape[1]): 
          Y_val = df_stats_per_iteration.iloc[:,j].values
          # Add some text for labels, title and custom x-axis tick labels, etc.
          plt.plot(df_stats_per_iteration.index, Y_val, label='Threshold: ' + df_stats_per_iteration.columns[j], color=cmap(j), linestyle='--')

        plt.axvline(opt_threshold*100,color='black', label='Optimum threshold')
        plt.xlabel('Loans refused at what probability of default (%)?')
        plt.ylabel('Measure of Threshold and Fairness')
        plt.legend()
        plt.show()

      # Print the optimal threshold....
      print('Optimal threshold: ' + str(opt_threshold))

    return opt_threshold

       
  @staticmethod
  def plot_fairness_feature_importance(X_test_protected: pd.DataFrame,
                                       prot_char: str,
                                       shap_values: object):
    #Sanity
    if prot_char not in X_test_protected.columns:
      raise TypeError('prot_char not in X_test_protected.columns')
    if shap_values is None:
      raise TypeError('Ini shap_values using the shap package')

    # Feature importance by protected characteristic.. different treatment?
    for prot_char in X_test_protected:
      # Extract the protected classes.
      curr_prot_cats = X_test_protected[prot_char].astype(str).to_list()

      # Plot the feature importance
      shap.plots.bar(shap_values.cohorts(curr_prot_cats).abs.mean(0), show=False)
      plt.title= "Bias Check: Feature Importance of protected group: " + prot_char
      plt.show()
    return
    # Ini
    df_stats = pd.DataFrame()
    stats_cols = []

    # Get categories in our test column
    categories = pd.Series(X_test_category_col).unique()

    # Type conversion
    #if type(X_test_category_col) != pd.Series:
    X_test_category_col = pd.Series(X_test_category_col)
    X_test_category_col = X_test_category_col.values
    
    # Loop through each of the categories in the category_col
    # Eg male (=0) and female (=1)
    # test acccuracy/precision/recall for each cat
    for cat in categories:

        #Filter on the cat
        cat_rows = (X_test_category_col == cat)
        #X...  
        X_test_cat = X_test.loc[cat_rows]
        #y...
        y_test_cat = y_test.loc[cat_rows]      
        
        # Predict the probability of default, and decide who to offer credit to, for specific population groups
        y_test_cat_hat_pred_proba = mod.predict_proba(X_test_cat.values)
        y_test_cat_hat = (y_test_cat_hat_pred_proba[:,0] < y_approval_threshold).astype('int')

        # Calc and record fairness analytics for each cat, record in df_stats
        TN, FP, FN, TP = confusion_matrix(y_test_cat, y_test_cat_hat, labels = [0,1]).ravel()
        fnr = FN/(FN+TP)
        fdr = FP/(FP+TP)
        fpr = FP/(FP+TN)
        npv = TN/(TN+FN)

        cat_row = pd.Series({'cat': cat,
                            'cat_proportion': np.divide(float(X_test_cat.shape[0]),float(X_test.shape[0])), 
                            'accuracy': accuracy_score(y_true=y_test_cat, y_pred=y_test_cat_hat),  
                            'precision': precision_score(y_true=y_test_cat, y_pred=y_test_cat_hat), 
                            'recall': recall_score(y_true=y_test_cat, y_pred=y_test_cat_hat), 
                            'fnr': fnr, 'fdr': fdr, 'fpr': fpr, 'npv': npv})

        # Build record of accuracy and so on, of each category in the category_col
        df_stats = pd.concat([df_stats, cat_row], axis=1)
        stats_cols.append(cat)
      
      # Predict the probability of default, and decide who to offer credit to, for all data
    y_test_pred_proba = mod.predict_proba(X_test.values)
    y_test_hat = (y_test_pred_proba[:,0] < y_approval_threshold).astype('int')

    # Calc and record fairness analytics for each cat, record in df_stats
    TN, FP, FN, TP = confusion_matrix(y_test, y_test_hat, labels = [0,1]).ravel()
    fnr = FN/(FN+TP)
    fdr = FP/(FP+TP)
    fpr = FP/(FP+TN)
    npv = TN/(TN+FN)

    cat_row = pd.Series({'cat': "All",
                        'cat_proportion': 1, 
                        'accuracy': accuracy_score(y_true=y_test, y_pred=y_test_hat),  
                        'precision': precision_score(y_true=y_test, y_pred=y_test_hat), 
                        'recall': recall_score(y_true=y_test, y_pred=y_test_hat), 
                        'fnr': fnr, 'fdr': fdr, 'fpr': fpr, 'npv': npv})

    # Build record of accuracy and so on, of each category in the category_col
    df_stats = pd.concat([df_stats, cat_row], axis=1)
    stats_cols.append(cat)      

    # Set up df_stats with column names and an index
    df_stats = df_stats.transpose()
    df_stats.columns = cat_row.index
    df_stats = df_stats.set_index(df_stats['cat'])

    return df_stats 

  # Plot the fairness charts
  @staticmethod
  def plot_fairness_charts (df_stats: pd.DataFrame, 
                        majority_class: str = 'Female',
                        fairness_metric: str = 'precision',
                        threshold_metric: str = 'recall',
                        y_approval_threshold: float = 0.5) -> pd.DataFrame:
    
      '''    
      Args:
          df_stats: record of the accuracy (etc) of the model on each category.
          majority_class: string (must be a string) - the value of the majority class against which the other population groups are compared with (e.g. ["Male"])
          fairness_metric: array with the fairness metrics to compare e.g. ["precision"]
          threshold_metric: Lender metric to optimize on...
          y_approval_threshold: Refuse loans with a probabiloyty of default over this value...
      Returns:
          A plot charting the fairness metric values to the various population groups...
      
      Author:
        Madhu Nagarajan
      '''
      #Sanity
      if fairness_metric not in df_stats.columns:
        raise TypeError('fairness_metric is not in df_stats.iloc[:,0]')
      if threshold_metric not in df_stats.columns:
        raise TypeError('threshold_metric is not in df_stats.iloc[:,0]')
      if majority_class.isnumeric():
        raise TypeError('non numeric descriptors of classes only')

      bars_to_plot = [fairness_metric,threshold_metric]
      N = len(bars_to_plot)
      X_val = df_stats["cat"].values.tolist()

      
      # map colors...
      def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
      cmap = get_cmap(10)
    
      width = 0.9  # the width of the bars
      plt.title = 'Fairness: Maximising threshold Metric (' + threshold_metric + '), while monitoring fairness metric p (' +  threshold_metric +  ')'

      fig, ax = plt.subplots(nrows=1, ncols=N ,  figsize=(8,6))
      i = 0
      for ametric in bars_to_plot: 
        #Get the metric corresponding to the majority - e.g. the recall corresponding to Male group
        majority_class_metric  = df_stats.loc[df_stats["cat"] == majority_class, ametric].astype('float64')

        #Y values to plot are the metrics of population groups, get them from df_stats
        Y_val = df_stats[ametric].values.tolist()
        
        ax[i].clear()
        
        #The plot displays a range that is +/- 20% from the metric for the majority class
        ax[i].axhline(y=majority_class_metric.values[0]*0.8,color='red', label='Upper unfairness bound')
        ax[i].axhline(y=majority_class_metric.values[0],color='green', label='Fairness parity')
        ax[i].axhline(y=majority_class_metric.values[0]*1.2,color='red', label='Lower unfairness bound')
        ax[i].legend(loc='lower center')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[i].set_ylabel('%')
        ax[i].set_xticklabels(X_val, rotation = 45, ha="right")
        ax[i].title.set_text(ametric)
        ax[i].bar(X_val, Y_val, width, label=ametric, color=cmap(i))

        i += 1
      plt.show()
