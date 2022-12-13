![image](https://user-images.githubusercontent.com/55665698/207409340-eb3aab6d-4271-4808-ac59-9d458107b34e.png)



# FinGov
Contains utility functions for implementing Machine Learning Governance and Fairness in a Finance process

## Governance Framework and Utility Function Overview

A Governance framework ensures models in an organisation achieve all their key stakeholder requirements satisfactorily, and in a safe, verifiable way. In this section we introduce functions we will use lter in thje use-case to control and monitor model development process to ensure our aims are achieved.

Manys steps in a Governance Framework are qualitative, requiring professionals to assess, specify, approve or reject stages in model development. However, quantitative tools can be a powerful utility, allowing professionals to control and monitor a process, and reach judgements about model design, stability, and efficacy. 

We discuss the 5 stages of model development and the utility functions that can be used to support the Governance Framework:

  

#### Stage1: Business Analysis
We first define our stakeholder KPIs, which should be systematically defined. We introduce example functions that go some way to representing stakeholder KPIs, with visualizations, statistical tests and checks where appropriate. 

>> Class StakeholderKPIReporting contains various examples of reporting functions that can be used to convey to model stakeholders, key performance indicators and other important information.

#### Stage2: Data Process
Exploratory data analysis goes some way to examining the quality and nature of the data, looking at distributions, correlations, imbalances in the data. We use some utility functions to support this.

>> GovernanceUtils contains function to examine data for imbalances

#### Stage3: Model Design and Development
From a governance point of view, model design and development is more qualitative, and requires good practice, statitically and in terms of the code implementation. Good commenting is essential, sanity checking of input and return values is advised, and in Python clear parameter declaration and control of source code, and code versions is essential too. 
We also need to ensure that the outcomes of our model are fair to different population groups, as well as having a good precision to protect the business from loan losses. We will introduce functions to ensure fairness.

>> GovernanceUtils contains function to select sklearn classifiers based on recall/precision/f1, and generates an ensemlbe challenger model from this process.
>> FairnessUtils() checks the fairness of model outcomes; allows selection of the optimal classifier based on performance criteria for the lender, and a fairness constraint.

#### Stage4: Model Deployment
Model deployment involes multiple stages of testing and authorization. We propose a challenger model to conduct part of this process, which is also used in the monitoring and reporting stage also.

>> Class StakeholderKPIReporting contains various examples of reporting functions that can be used to convey to model stakeholders, key performance indicators and other important information.
>> FairnessUtils() monitors fairness in outcomes between majority and minority classes.

#### Stage5: Monitoring/Reporting
During live running of the models, monitoring of data drift is essential, and for additional safety a challenger model can be run in parallel to the live model, to ensure the live model is functioning well with respect to stakeholder KPIs.

>> GovernanceUtils contains challenger model, fit and predict functions to compare live model performance; data drift functions to ensure the distributions of the input data are similar to the training data.
