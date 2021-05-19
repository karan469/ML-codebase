# ML Concepts related to tabular datasets
### For example, Decision Trees, Random Forests, Gradient Boosting Trees, SVM

## 1. Decision Trees
A decision tree is a tree where each node represents a feature(attribute), each link(branch) represents a decision(rule) and each leaf represents an outcome(categorical or continues value).

![image](https://user-images.githubusercontent.com/33158202/118825326-35cce800-b8d8-11eb-9f0c-f59648fe1b1e.png)

### Algorithm:<br>
  1. Compute the entropy for data-set:<br>
  2. For every attribute/feature:<br>
       &nbsp;&nbsp;&nbsp;&nbsp; 1. Calculate entropy for all categorical values<br>
       &nbsp;&nbsp;&nbsp;&nbsp; 2. Take average information entropy for the current attribute<br>
       &nbsp;&nbsp;&nbsp;&nbsp; 3. Calculate gain for the current attribute<br>
  3. Pick the highest gain attribute.<br>
  4. Repeat until we get the tree we desired.<br>
