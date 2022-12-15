from StakeholderKPIReporting import StakeholderKPIReporting

# package for working with tabular data
import pandas as pd 
import numpy as np

# Package for charting
import matplotlib.pyplot as plt
import seaborn as sns #charts

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

# Classsifiers from sklearn
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import sklearn.metrics as metrics

# Performance metrics...
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import warnings

class GovernanceUtils():

  # Func to wrap up running these selected classification learners...
  # NOTE: to test the performance of the learners out-of-sample, we should use a cross-validation dataset
  # this is a hold back dataset and we will use our testing data to do this, in this case. 
  def auto_classifier_selection(X_train: pd.DataFrame, 
                              X_cross_validation: pd.DataFrame, 
                              y_train: pd.DataFrame, 
                              y_cross_validation: pd.DataFrame, 
                              selection_criteria: str = 'precision',
                              balance_method: str = '') -> (object, list, list, list):
      
      '''
      Args:
        X_train: DataFrame with training data for classifier, columns are features, rows are instances
        X_cross_validation: cross validation data matching above shape, used in model selection
        y_train: training data target variable {1,0}, instances are rows.
        y_cross_validation: test data target variable {1,0}, instances are rows, used in model selection
        selection_criteria: str value, one of ['precision','recall','accuracy','f1']
        balance_method: 'smote' or '' for nothing

      Returns:
        max_mdl: sklearn model object performing "best"
        all_mdls: list of sklearn classifier objects trained
        all_mdls_desc: list of description of the above model objects (elements corresponding to the above also)
        all_mdls_prec: list of precision scores of the above model objects (elements corresponding to the above also)
        
      Author:
        Dan Philps
      '''
      #sanity
      if X_train.shape[0] != y_train.shape[0]:
        raise TypeError('Bad parameter: X_train.shape[0] != y_train.shape[0]')
      if X_cross_validation.shape[0] != y_cross_validation.shape[0]:
        raise TypeError('Bad parameter: X_train.shape[0] != y_train.shape[0]')
      if (X_train.dtypes != X_cross_validation.dtypes).sum() != 0:
        raise TypeError('Bad parameter: X_train.dtype != X_cross_validation.dtype')
      #if (y_train.dtypes != y_cross_validation.dtypes):
      #  raise TypeError('Bad parameter: y_train.dtype != y_cross_validation.dtype')
      if selection_criteria not in ['precision','recall','accuracy','f1']:
        raise TypeError("selection_criteria not in ['precision','recall','accuracy','f1']")

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #Balance training data....
        
        if balance_method == 'smote':
          # Generate SMOTE samples and use this to train
          upsampler_smote = SMOTE()
          X_upsampled_smote, y_upsampled_smote = upsampler_smote.fit_resample(X_train.values, y_train.values)
        else:
          # No up or down sampling
          X_upsampled_smote = X_train.values
          y_upsampled_smote = y_train.values
         
        X_train_cols = X_train.columns #retain cols so we can meaninglfully use XAI

        sclr = StandardScaler()
        sclr.fit(X_train.values) # scale to 0 mean and std dev 1 on training data

        X_train = sclr.fit_transform(X_upsampled_smote) # scale both sets:
        X_cross_validation = sclr.fit_transform(X_cross_validation)

        # reinstate cols so we can meaninglfully use XAI
        X_train = pd.DataFrame(X_train)
        X_train.columns = X_train_cols

        # These are the classifiers we will select from...
        #dtc = DecisionTreeClassifier(max_depth=5) #If we allow endless depth we overfit
        gnb = GaussianNB()
        lr = LogisticRegression(max_iter=2000,random_state=0)
        mlp = MLPClassifier(max_iter=2000,random_state=1, early_stopping=True) # MLP will tend to overfit unless we stop early   
        rf = RandomForestClassifier(max_depth=3,random_state=0) # << artibitrary parameters, consider hyper parameter tuning.
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()
        ada = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        knn = KNeighborsClassifier(n_neighbors=3) # << artibitrary parameters, consider hyper parameter tuning.
        svc = SVC(kernel="rbf", C=0.025, probability=True) # << artibitrary parameters, consider hyper parameter tuning.
        nsvc = NuSVC(probability=True)

        all_mdls = [gnb, lr, mlp, rf, lda, qda, ada, gbc, knn, svc, nsvc]
        all_mdls_desc = ['gnb', 'lr', 'mlp', 'rf', 'lda', 'qda', 'ada', 'gbc', 'knn', 'svc', 'nsvc']
        all_mdls_perf = []
        
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")

          # Loop through each classifer and record the "best"...
          max_perf = 0
          for mdl in all_mdls:
              #Fit model
              mdl.fit(X_upsampled_smote,y_upsampled_smote)  
              y_train_hat = mdl.predict(X_upsampled_smote)
              y_cross_validation_hat = mdl.predict(X_cross_validation)       

              # Output model selection information....Analytics calculated wrt default or y=1... Print score
              print(mdl)
              
              # Selection based on cross-validation set, ie out of sample data not used in training
              if selection_criteria == 'precision':
                this_cv_perf = precision_score(y_cross_validation,y_cross_validation_hat, average=None)[1]
                #
                print(f"Precision train: {precision_score(y_upsampled_smote, y_train_hat, average=None)[1]:.4f}, cross-validation: ",
                f"{precision_score(y_cross_validation,y_cross_validation_hat, average=None)[1]:.4f}")
              elif selection_criteria == 'recall':
                this_cv_perf = recall_score(y_cross_validation,y_cross_validation_hat, average=None)[1]
                #
                print(f"Precision train: {recall_score(y_upsampled_smote, y_train_hat, average=None)[1]:.4f}, cross-validation: ",
                f"{recall_score(y_cross_validation,y_cross_validation_hat, average=None)[1]:.4f}")
              elif selection_criteria == 'accuracy':
                this_cv_perf = mdl.score(X_cross_validation,y_cross_validation)
                #
                print(f"Precision train: {mdl.score(y_upsampled_smote, y_train_hat)[1]:.4f}, cross-validation: ",
                f"{mdl.score(y_cross_validation,y_cross_validation_hat)[1]:.4f}")
              elif selection_criteria == 'f1':
                this_cv_perf = f1_score(y_cross_validation,y_cross_validation_hat, average=None)[1]
                #
                print(f"Precision train: {f1_score(y_upsampled_smote, y_train_hat, average=None)[1]:.4f}, cross-validation: ",
                f"{f1_score(y_cross_validation,y_cross_validation_hat, average=None)[1]:.4f}")
                
              if this_cv_perf > max_perf:
                  max_perf = this_cv_perf
                  max_mdl = mdl
             
              #Save the F1 score of this model...
              all_mdls_perf.append(this_cv_perf)

          # The best....
          #Fit...
          max_mdl.fit(X_upsampled_smote,y_upsampled_smote)
          y_train_hat = max_mdl.predict(X_upsampled_smote)
          y_cross_validation_hat = max_mdl.predict(X_cross_validation)

          # Analytics calculated wrt default or y=1... Print score
          print('\nWinner\n', type(max_mdl))        
          print(f"Accuracy train: {max_mdl.score(X_train,y_upsampled_smote):.4f}, cross-validation: ",
            f"{max_mdl.score(X_cross_validation,y_cross_validation):.4f}")
          print(f"Precision train: {precision_score(y_upsampled_smote, y_train_hat, average=None)[1]:.4f}, cross-validation: ",
            f"{precision_score(y_cross_validation,y_cross_validation_hat, average=None)[1]:.4f}")
          print(f"Recall train: {recall_score(y_upsampled_smote, y_train_hat, average=None)[1]:.4f}, cross-validation: ",
            f"{recall_score(y_cross_validation,y_cross_validation_hat, average=None)[1]:.4f}")
          print(f"F1 train: {f1_score(y_upsampled_smote, y_train_hat, average=None)[1]:.4f}, cross-validation: ",
            f"{f1_score(y_cross_validation,y_cross_validation_hat, average=None)[1]:.4f}")

          #Print confusion matrix...
          cf_matrix = confusion_matrix(y_cross_validation, y_cross_validation_hat, labels=[0, 1]) 
          cf_matrix_norm = cf_matrix.astype('float')

          ax = sns.heatmap(cf_matrix_norm, annot=True, cmap='Blues', fmt='g')
          ax.set_title('Confusion Matrix\n\n');
          ax.set_xlabel('\nPredicted Values')
          ax.set_ylabel('Actual Values ');
          plt.show()
      
      #sanity
      if max_mdl is None:
        raise TypeError('Bad return: max_mdl is None')

      return max_mdl, all_mdls, all_mdls_desc, all_mdls_perf

  # Prepare data for predictions
  def norm_X(X_train: pd.DataFrame,
              X_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Args:
      X_train: base normalization scheme on this data.
      X_test: Normalize this data based on the normalization scheme
        
    Returns:
        X_test: normalized
        
    Author:
        Dan Philps
    '''
    
    # Scale and transform the data for training
    sclr = StandardScaler()
    sclr.fit(X_train) # scale to 0 mean and std dev 1 on training data

    X_train_cols = X_train.columns
    X_train = sclr.fit_transform(X_train) # scale both sets:
    X_test_normalized = sclr.fit_transform(X_test)

    X_train = pd.DataFrame(X_train)
    X_train.columns = X_train_cols

    X_test_normalized = pd.DataFrame(X_test_normalized)
    X_test_normalized.columns = X_train_cols

    return X_test_normalized
    
  # Check to run on Challenger and live models that encapsulates key KPIs
  def challenger_review_live(live_mod: object,
                            challenger_mod: object,
                            X_test: pd.DataFrame,
                            y_test: pd.DataFrame,
                            precision_fault_threshold: float = 0.1,
                            recall_fault_threshold: float = 0.1) -> str:
    '''
    Args:
      live_mod: live model, trained and ready to go.
      challenger_mod: challenger model to compare to live, trained and ready to go.
      X_test: Test data matching above shape
      y_test: test data target variable {1,0}, instances are rows.
      precision_fault_threshold: if challenger is > this must better than live, throw an exception
      recall_fault_threshold: if challenger is > this must better than live, throw an exception
        
    Returns:
        err: error message
        
    Author:
        Dan Philps
    '''
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      
      # Run models and compare
      y_hat_live = live_mod.predict(X_test)
      y_hat_challenger = challenger_mod.predict(X_test)
    
      # Compare the precsision of live and challenger
      live_recall, live_prec = StakeholderKPIReporting.kpi_review_customer_business_compliance(live_mod, X_test, y_test, y_hat_live)
      challenger_recall, challenger_prec = StakeholderKPIReporting.kpi_review_customer_business_compliance(challenger_mod, X_test, y_test, y_hat_challenger)

      # Simple test
      err = ''
      if (challenger_prec - live_prec > precision_fault_threshold):
          err = 'Precision fault threshold breached! Challenger model achieving materially better precision than live - consider retraining live models.'
          print('Precision fault threshold breached! Challenger model achieving materially better precision than live - consider retraining live models.')

      if (challenger_recall - live_recall > recall_fault_threshold):
          err = 'Recall fault threshold breached! Challenger model achieving materially better recall than live - consider retraining live models.'
          print('Recall fault threshold breached! Challenger model achieving materially better recall than live - consider retraining live models.')

      # Bar chart of prec and recall
      plt.bar(['live_prec', 'challenger_prec'], [live_prec, challenger_prec], color = 'b')
      plt.bar(['live_recall', 'challenger_recall'], [live_recall, challenger_recall], color = 'r')
      plt.title('Bar chart of Precision and Recall')
      plt.show()

      # ROC Curve
      y_hat_prob_live = live_mod.predict_proba(X_test)[:, 1]
      y_hat_prob_challenger = challenger_mod.predict_proba(X_test)[:, 1]

      plt.title("Credit Decisions ROC Curve")

      fpr, tpr, _ = metrics.roc_curve(y_test, y_hat_prob_live)
      plt.plot(fpr,tpr,label='Live model')

      fpr, tpr, _ = metrics.roc_curve(y_test, y_hat_prob_challenger)
      plt.plot(fpr,tpr,label='Challenger model')

      plt.legend(['Live', 'Challenger'])
      plt.title('ROC Curves: Live vs Challenger')
      plt.show()

    return err

  # Create a challenger model
  # requires a list of ready trained models (all_mdls) and corresponding descrtiptions 
  # (all_mdls_desc) and performance (all_mdls_prec). Ensembled the better performers
  # and uses the resulting model as a challenger 
  @staticmethod
  def challenger_ensemble_run(all_mdls: list, 
                    all_mdls_desc: list,
                    all_mdls_prec: list,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame, 
                    X_test: pd.DataFrame) -> (np.array, object):
    '''
    Args:
      all_mdls: a list of sklearn classifiers, trained and ready to go.
      all_mdls_desc: list of description of the above model objects (elements corresponding to the above also)       
      all_mdls_prec: list of out-of-sample prec scores for each of the above model objects, used to elmininate the poor performers from the challenger (elements corresponding to the above also)       
      X_train: DataFrame with training data for classifier, columns are features, rows are instances
      X_test: Test data matching above shape
      y_train: training data target variable {1,0}, instances are rows.
      y_test: test data target variable {1,0}, instances are rows.
        
    Returns:
        y_hat: numpy array containing predictions from the challenger
        object: sklearn model object containing the challenger model.
        
    Author:
        Dan Philps
    '''

    #Sanity
    if X_train.shape[0] != y_train.shape[0]:
        raise TypeError('Bad parameter: X_train.shape[0] != y_train.shape[0]')
    if X_test.shape[1] != X_train.shape[1]:
      raise TypeError('Bad parameter: X_train.shape[0] != y_train.shape[0]')
  
   # Supress warnings
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
   
      # Only use classifiers that have generated an above median F1 score out of sample.
      min_prec_to_use = np.median(all_mdls_prec)

      # Prepare our models for ensembling
      challenger_models = []
      for i in range(0, all_mdls_desc.__len__()):
        if all_mdls_prec[i] > min_prec_to_use:
          challenger_models.append((all_mdls_desc[i], all_mdls[i]))

      # Instantiate ...
      vc = VotingClassifier(estimators=challenger_models, voting='soft')

      # Fit on the training data to train your live model... 
      challenger_mdl = vc.fit(X_train, y_train.values)

      # Challenge! Compare the results
      y_hat = challenger_mdl.predict(X_test)
    
    return y_hat, challenger_mdl


  # Check for imbalances, charts pie of imbalances in the y variable 
  # wrapped as a func as we will use it a few times..
  @staticmethod
  def imbalanced_y_check(y: pd.Series) -> bool:  
    '''
      Args:
          y: Dataframe of only the y variable
        
      Returns:
        bool: True if imbalanced, False, if not imbalanced.

      '''

    print('Dataset Balanced?')
    print(y.value_counts())

    # Convert to df...
    df_y = pd.DataFrame(y)
    class_col = y.name
    df_y.groupby(df_y[class_col]).size().plot(kind='pie', y=class_col, label = "Type",  autopct='%1.1f%%')

    #Rule of thumb... highest frequency class < 70% of observations
    imbalanced = False
    perc_split = y.value_counts() / df_y.shape[0]
    if np.max(perc_split) >= 0.7:
      print('Imbalanced y variable!')
      imbalanced = True
    
    return imbalanced

  # Population Stability Index (PSI) can be applied to the input features or variables, also known as Characteristic 
  # Stability Index (CSI), as well as the output of a scoring model, a model whose score may indicate the probability 
  # of fraud, or probability of default. PSI captures the shift in the population distribution of values. If the score
  # distribution has shifted, one should then look to see what feature(s) or variable(s) is causing the shift.
  # A shift in the distribution of input features or features, or output score distribution could imply that the model 
  # may need retrained. The common interpretation of PSI, which comes from the orignal work on credit models, is as follows: 
  # PSI < 0.1: no significant population change, PSI < 0.2: moderate population change and PSI >= 0.2: significant population
  # change.
  #https://www.quora.com/What-is-population-stability-index
  @staticmethod
  def data_drift_psi(X_train: pd.DataFrame,                  
                  X:  pd.DataFrame, 
                  buckettype: str='bins', 
                  buckets: int =10, 
                  axis: int =0, 
                  single_variable: bool=False, 
                  show_results: bool = True) -> np.ndarray:
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values - both features and target
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, 
       quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal 
       single_variable: True if only passing in one column of data, like the target variable
       show_results: print out the results...
       
    Returns:
       psi_values: ndarray of psi values for each variable
       
    Author:
       Matthew Burke, Augustine Backer, Dan Philps
       github.com/mwburke
       worksofchart.com
    '''
    # sanity
    if single_variable == False:
        if X_train.shape[1] != X.shape[1]:
          raise TypeError('X_train.shape != X.shape')

    # Ini data   
    ar_X_train = X_train.to_numpy()
    ar_X = X.to_numpy()

    def psi(expected_array: np.array,        
            actual_array: np.array,        
            buckets: int,
            show_results: bool = True) -> np.ndarray:
        '''Calculate the PSI for a single variable 
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
           show_results: print out results...
           
        Returns:
           psi_value: calculated PSI value
        
        Author: Augustine Backer
        '''
        
        def scale_range (input,
                         min, 
                         max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
    
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
    
        def sub_psi(e_perc: float, 
                    a_perc: float) -> float:
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.001

                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)
        
        psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)
    
    # The shape function returns the dimension of the array, with 1 being one variable being examined
    # psi_values - a psi value will be calculated for each variable in the array
    if len(ar_X_train.shape) == 1:
        psi_values = np.empty(len(ar_X_train.shape))
    else:
        psi_values = np.empty(ar_X_train.shape[axis])
    
    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(ar_X_train, X, buckets)
        elif axis == 0:
            psi_values[i] = psi(ar_X_train[:,i], ar_X[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(ar_X_train[i,:], ar_X[i,:], buckets)
    
    # Print out the features CSI values
    if show_results == True:
      print("The feature CSI values are:")
      columns_features = X_train.columns
      i = 0
      for col in columns_features:

          if psi_values[i] > 0.2:
                  print(col, "*************** CSI value is over 0.2 = ",psi_values[i])
          elif psi_values[i] > 0.1:
                  print(col, "*************** CSI value is over 0.1 = ",psi_values[i])
          else:
                  print(col, "CSI is OK = ",psi_values[i])
          i += 1
      
    return(psi_values)


  # Kolmogorov-Smirnov (KS) is used to measure the performance of classification models. More accurately, KS is a measure of 
  # the degree of separation between the positive and negative distributions, for example deafult vs. non-default. K-S ranges
  # from 0% to 100%, and the higher the KS value is, the better the model is at separating the positive and negative 
  # distributions. The KS statistic for two samples is simply the greatest distance between their two cummulative 
  # distribution functions, so if we measure the distance between the positive and negative class distributions. 
  # A KS of 0.6 or higher is considered good, associated with a low p-value.
  # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test https://towardsdatascience.com/evaluating-classification-models-with-kolmogorov-smirnov-ks-test-e211025f5573
  from scipy.stats import ks_2samp
  @staticmethod
  def ks_test_result(model_score_target: pd.DataFrame,
                    score, 
                    target) -> float:
      '''
      Args:
          model_score_target: dataframe of the model output and binary target (1=event, 0=non event), with the colunns 
          labeled to match the target_column and score_column strings passed in
          target: string of target column headerin the dataframe, example "y"
          score: string of the score column header, or model output, typically [0,100], where the higher the score the higher
          the likelihood of an event, for example "score"
        
      Returns:
        ks_statistic: the KS value for the cummulative distribution function of the scores of events and non-events
      
      Author:
        Augustine Backer
      '''
      ks=ks_2samp(model_score_target.loc[model_score_target[target]==0,score], 
                  model_score_target.loc[model_score_target[target]==1,score])
      
      print(f"KS: {ks.statistic:.4f} (p-value: {ks.pvalue:.3e})")

      return ks.statistic

  ########################################################################################################################

  # By itself, Kolmogorov-Smirnov (KS) is a good measure of the degree of separation between the positive and negative 
  # distributions that the model can achieve, for example deafult vs. non-default. However, it is important to monitor any 
  # changes in KS from the baseline period and current monitoring period. This is typically done by looking at the relative 
  # percentage change. For example if KS in the baseline is 80%, or 0.80, and in the current period it is 0.60, then 
  # (0.60 - 0.80)/).80 = -25%. The inner and outer thresholds for this Key Performance Indicator would be -15% to -25%.
  @staticmethod
  def data_drift_ks_test(model_score_target_baseline, model_score_target_current, score, target):
      '''
      Args:
          model_score_target_baseline: dataframe of the model output during baseline and binary target (1=event, 0=non event), with the colunns 
          labeled to match the target_column and score_column strings passed in
          model_score_target_current: datafrmae of the model output during current monitoring period
          target: string of target column header in the dataframe, example "y"
          score: string of the score coumn header, or model output, typically [0,100], where the higher the score the higher
          the likelihood of an event, for example "score"
        
      Returns:
        ks_relative_percent_change: the KS value for the cummulative distribution function of the scores of events and non-events
      '''
      ks_baseline=ks_2samp(model_score_target_baseline.loc[model_score_target_baseline[target]==0,score], 
                  model_score_target_baseline.loc[model_score_target_baseline[target]==1,score])
      ks_current=ks_2samp(model_score_target_current.loc[model_score_target_current[target]==0,score], 
                  model_score_target_current.loc[model_score_target_current[target]==1,score])
      
      print(f"KS Baseline: {ks_baseline.statistic:.4f} (p-value: {ks_baseline.pvalue:.3e})")
      print(f"KS Monitoring Period: {ks_current.statistic:.4f} (p-value: {ks_current.pvalue:.3e})")
      
      ks_relative_percent_change = (ks_current.statistic - ks_baseline.statistic)/ks_baseline.statistic

      return ks_relative_percent_change
