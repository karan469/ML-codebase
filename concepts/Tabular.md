# ML Concepts related to tabular datasets
*For example, Decision Trees, Random Forests, Gradient Boosting Trees, SVM*

## 1. Decision Trees
A decision tree is a tree where each node represents a feature(attribute), each link(branch) represents a decision(rule) and each leaf represents an outcome(categorical or continues value).

![image](https://user-images.githubusercontent.com/33158202/118827242-d374e700-b8d9-11eb-8ae6-6884a16dd7fe.png)

### Algorithm:<br>
  1. Compute the entropy for data-set:<br>
  2. For every attribute/feature:<br>
       &nbsp;&nbsp;&nbsp;&nbsp; 1. Calculate entropy for all categorical values<br>
       &nbsp;&nbsp;&nbsp;&nbsp; 2. Take average information entropy for the current attribute<br>
       &nbsp;&nbsp;&nbsp;&nbsp; 3. Calculate gain for the current attribute<br>
  3. Pick the highest gain attribute.<br>
  4. Repeat until we get the tree we desired.<br>

At  each  node,  you  should  select  the  attribute  which  results  in  maximum  decrease  inthe entropy of the class variable (i.e.  has the highest mutual information with respect to the classvariable).  This problem has some attributes as integers and some as boolean.  For handling continuousattributes, you should use the following procedure.  At any given internal node of the tree, a numericalattribute is considered for a two way split by calculating the median attribute value from the datainstances coming to that node, and then computing the information gain if the data was split basedon whether the numerical value of the attribute is greater than the median or not.

## 2. Ensemble

*Collection of predictors come together to give final predictions*

    
  1. **Bagging**: Model averaging technique (weighted avg, majority vote etc) used on a set of multiple predictors, trained on random subsample/bootstrap of dataset.     <br>Each model has different observations and thus, this reduces variance.<br>Example: Random Forests.<br>

  2. **Boosting**: Predictors are not made independently, but sequentially. But we have to choose a stopping criteria unless it can lead to overfitting.

![image](https://user-images.githubusercontent.com/33158202/118827036-a1638500-b8d9-11eb-9a62-019e216783fd.png)

  3. **AdaBoost vs GBM**: AdaBoost reweighs samples which were misclassified in the previous weak learner. GBM -> This approach trains learners based upon minimising the loss function of a learner (i.e., training on the residuals of the model). The calculated contribution of each tree is based on minimising the overall error of the strong learner.
