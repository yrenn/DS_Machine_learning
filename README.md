# DS_Machine_Learning
This project creates a data modeler to build and test models' accuracy using Python.


# Introductions of each model
HistGradientBoostingClassifier

This estimator has native support for missing values (NaNs). 
During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with missing values are assigned to the left or right child consequently. 
If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples.

RandomForestClassifier

A random forest is a meta-estimator that fits a number of decision tree classifiers  on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

DecisionTreeClassifier

It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.



# lesson learned
1. We use the mean value to fill in the missing value to improve the data quality.
2. We test some common data models and find the accuracy of each model. The model 'DecisionTreeClassifier' has 100% accuracy and others have 40% and 60% accuracy.
3. Basically, the common steps of creating a data modeler are initializing the DataModeler, preparing a data frame, imputing missing values if exist, fitting the model, and making a set of predictions with the model.
