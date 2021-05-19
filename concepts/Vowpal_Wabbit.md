# vw-varinfo      Summarize features of a training-set using VW

  **Input**:          A vw training set file<br>
  **Output**:         A list of features, their VW hash values, min/max
                  values, regressor weights, and distance from
                  the best constant.

Algorithm:
  1)  Collect all variables and their ranges from training-set
  2)  Train with VW to determine regressor weights
  3)  Build a test-set with a single example including all variables
  4)  run VW with --audit on 3) to map variable names to hash values
      and weights.
  5)  Output collected information about the input variables.
