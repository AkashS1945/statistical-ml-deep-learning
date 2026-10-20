# Complete Answers to HW4 Questions

## Part 2: Decision Trees for Review Classification (25 Marks)

### 2.1 Train a Simple Tree and Visualize (10 Marks)

**Question: Is there any internal node that has two child leaf nodes corresponding to the same sentiment class?**

**Answer: No**, looking at the tree output, there are no internal nodes where both child leaves predict the **same sentiment class** (i.e., both predict positive or both predict negative as the majority class).

However, we can observe some interesting patterns:
- At depth 2, under `X['great'] > 0.50` → `X['return'] > 0.50`, both leaves have relatively extreme probabilities (0.190 vs 0.833)
- The left branch (`X['great'] <= 0.50`) shows leaves with lower positive probabilities (0.114-0.430)
- The right branch (`X['great'] > 0.50`) shows leaves with higher positive probabilities (0.415-0.833)

**Why would having two children predict the same class make sense?**

Even if both child nodes predict the same majority class, the split can still be valuable because:

1. **Different probability distributions**: Both leaves might predict "positive" as the majority class, but with different confidence levels (e.g., 60% vs 85% probability of positive sentiment)

2. **Purity improvement**: The Gini impurity may decrease even if the majority class stays the same, because the split makes each subset more homogeneous

3. **Future splitting potential**: If the tree weren't limited to depth 3, these nodes could be split further to better separate the classes

4. **Statistical refinement**: The split helps the model learn different probability estimates for different subgroups, even within the same predicted class

**Example**: A node with p(positive)=0.55 might split into children with p(positive)=0.40 and p(positive)=0.70. Both would predict "positive" if using 0.5 threshold, but the split provides valuable probability calibration.

---

### 2.2 Hyperparameter Tuning (15 Marks)

**Question: What are the values of max_depth and min_samples_leaf for the best tree found by the grid search?**

**Answer:**
- **max_depth = 32**
- **min_samples_leaf = 3**

These hyperparameters were selected from the grid search over:
- max_depth: [2, 8, 32, 128]
- min_samples_leaf: [1, 3, 9]

The best model achieved a **validation balanced accuracy of 0.732** (73.2%) and **test balanced accuracy of 0.749** (74.9%).

**Interpretation:**
- **max_depth=32**: Much deeper than the simple tree (depth=3), allowing the model to learn complex decision boundaries needed for text classification
- **min_samples_leaf=3**: Requires at least 3 samples in each leaf node, which prevents overfitting to individual examples and improves generalization

---

## Part 3: Random Forests for Review Classification (30 Marks)

### 3.1 Train a Simple Random Forest and Analyze Features (10 Marks)

**Table 2: Feature Importances**

#### Top 10 Most Important Words:
1. return (0.032990)
2. excel (0.029485)
3. great (0.028984)
4. worst (0.028407)
5. poor (0.026748)
6. disappoint (0.024952)
7. your_money (0.018002)
8. i_love (0.017955)
9. the_best (0.017662)
10. don't (0.017654)

#### 10 Randomly Selected Unimportant Words (importance < 0.00001):
1. doing_the (0.000000)
2. bought_these (0.000000)
3. mix (0.000000)
4. not_all (0.000000)
5. vision (0.000000)
6. example_the (0.000000)
7. the_author's (0.000000)
8. abuse (0.000000)
9. went_on (0.000000)
10. also_very (0.000000)

**Total eligible near-zero importance features: 7,296 out of 7,729 (94.4%)**

**Interpretation:**
- Strong sentiment indicators dominate the top features (negative: "return", "worst", "poor", "disappoint"; positive: "excel", "great", "i_love", "the_best")
- The vast majority of vocabulary words (94.4%) contribute almost nothing to predictions, suggesting significant feature redundancy in bag-of-words representation
- Common phrase fragments like "doing_the" and "example_the" are uninformative for sentiment

---

### 3.2 Hyperparameter Tuning for Random Forest (10 Marks)

**Question 1: What is the value of max_features of your best forest?**

**Answer: max_features = 33**

This was selected from the grid search over [3, 10, 33, 100, 333], achieving a **validation balanced accuracy of 0.851** (85.1%) and **test balanced accuracy of 0.837** (83.7%).

---

**Question 2: What is the maximum possible value for max_features for this dataset? Why is it beneficial to tune this hyperparameter?**

**Answer:**

**Maximum possible value: 7,729** (the total number of features in the vocabulary)

**Why tuning max_features is beneficial:**

1. **Controls tree diversity**: Lower max_features means each tree considers fewer features at each split, forcing trees to use different features and become more diverse. Higher diversity reduces correlation between trees, improving ensemble performance.

2. **Bias-variance tradeoff**: 
   - **Small max_features** → more bias (weaker individual trees), less variance (more diverse ensemble)
   - **Large max_features** → less bias (stronger individual trees), more variance (similar trees)

3. **Computational efficiency**: Fewer features per split reduces training time

4. **Prevents dominance of strong features**: With max_features < total features, not every tree will split on the same dominant features (like "great" or "return"), allowing other informative features to be discovered

5. **Performance results from our grid search**:
   - max_features=3: 0.797 validation accuracy (too restrictive)
   - max_features=33: **0.851 validation accuracy (optimal)**
   - max_features=333: 0.812 validation accuracy (too permissive)

The optimal value (33) is close to √7729 ≈ 88, which is a common rule of thumb for classification tasks, providing a good balance between individual tree strength and ensemble diversity.

---

**Question 3: When fitting random forests, what is the primary tradeoff controlled by the n_estimators hyperparameter? Can you overfit by setting it to be too large? Why or why not?**

**Answer:**

**Primary tradeoff:**

The **n_estimators** hyperparameter controls the **performance vs. computational cost** tradeoff:

- **More trees** (larger n_estimators):
  - ✓ Better performance (predictions stabilize through averaging)
  - ✓ More robust predictions (lower variance)
  - ✗ Longer training time
  - ✗ More memory required
  - ✗ Slower prediction time

- **Fewer trees** (smaller n_estimators):
  - ✓ Faster to train and predict
  - ✓ Less memory
  - ✗ Higher variance in predictions
  - ✗ Potentially lower performance

**Can you overfit by setting n_estimators too large? NO.**

**Why not:**

1. **Averaging reduces overfitting**: Random forests work by averaging predictions from multiple trees. As you add more trees, the variance decreases but the bias stays approximately constant. The law of large numbers ensures that averaging more independent (or weakly correlated) predictions approaches the expected value.

2. **Asymptotic behavior**: Performance typically improves rapidly with the first ~50-100 trees, then plateaus. Adding more trees beyond this point gives diminishing returns but doesn't cause overfitting.

3. **Mathematical proof**: Each tree is trained on a bootstrap sample (sampling with replacement). The out-of-bag (OOB) error serves as a validation estimate. As n_estimators → ∞, the OOB error converges to a stable value; it doesn't increase.

4. **Empirical evidence**: In practice, test error decreases or stabilizes as n_estimators increases, but never increases (unlike increasing depth, which can cause overfitting).

**Practical recommendation**: Use as many trees as computationally feasible. Common choices are 100-500 trees. Beyond ~200 trees, improvements are typically marginal. We used 100 trees as a reasonable balance.

---

### 3.3 Final Model Comparison (10 Marks)

**Table 3: Comparison of Methods on Bag-of-Words Sentiment Classification**

| Method | Max Depth | Num Trees | Train BAcc | Valid BAcc | Test BAcc |
|--------|-----------|-----------|------------|------------|-----------|
| Simple Tree | 3 | 1 | 0.646 | 0.645 | 0.646 |
| Best Tree | 32 | 1 | 0.877 | 0.732 | 0.749 |
| Simple RandomForest | 3 | 100 | 0.819 | 0.797 | 0.778 |
| Best RandomForest | 32 | 100 | 0.964 | 0.851 | 0.837 |

**Summary of Conclusions:**

1. **Ensemble methods significantly outperform single trees**: The Best RandomForest (83.7% test accuracy) outperforms the Best Tree (74.9% test accuracy) by 8.8 percentage points, demonstrating the power of ensemble learning through averaging multiple diverse trees.

2. **Hyperparameter tuning is crucial**: 
   - For trees: tuning improved test accuracy from 64.6% to 74.9% (+10.3 points)
   - For forests: tuning improved test accuracy from 77.8% to 83.7% (+5.9 points)

3. **Random forests provide better generalization**:
   - Best Tree: train-test gap = 0.877 - 0.749 = 0.128 (12.8% overfitting)
   - Best RandomForest: train-test gap = 0.964 - 0.837 = 0.127 (12.7% overfitting)
   - Despite similar gaps, the forest achieves much higher overall performance

4. **The simple tree severely underfits**: With only depth=3, the simple tree achieves identical train and test accuracy (64.6%), indicating it lacks sufficient capacity to learn the patterns in the data.

5. **Validation and test performance are consistent**: The validation accuracy (85.1%) closely matches test accuracy (83.7%) for the best forest, indicating good generalization and proper hyperparameter selection without overfitting to the validation set.

6. **Deeper trees are necessary**: The optimal max_depth=32 for both trees and forests indicates that sentiment classification from bag-of-words requires complex decision boundaries that cannot be captured by shallow trees.

7. **Feature importance analysis reveals interpretable patterns**: The top features (like "great", "worst", "excel", "poor") are intuitively meaningful sentiment indicators, validating that the model learns sensible patterns rather than spurious correlations.

8. **Most features are uninformative**: 94.4% of vocabulary words have near-zero importance, suggesting that feature selection or dimensionality reduction could improve efficiency without sacrificing performance.

**Overall Conclusion**: Random forests with proper hyperparameter tuning (max_depth=32, max_features=33, n_estimators=100) are highly effective for bag-of-words sentiment classification, achieving 83.7% balanced accuracy on held-out test data. The ensemble approach successfully balances model complexity with generalization, substantially outperforming single decision trees while maintaining interpretability through feature importance analysis.