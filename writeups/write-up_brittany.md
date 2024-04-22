<!-- ============================================================================================================== -->
---
<!-- ============================================================================================================== -->
---





<!-- ============================================================================================================== -->
---


Methodology:

Data Preprocessing:
- Loaded the preprocessed dataset from a CSV file.
- Removed rows not belonging to the Gulf Coast basin.
- Dropped unnecessary columns.
- Handled missing values.

Model Training:
- Utilized Support Vector Regression (SVR) model.
- Conducted grid search with cross-validation to find the best hyperparameters.
- Split the data into training and testing sets.
- Standardized features by removing the mean and scaling to unit variance.

Model Evaluation:
- Calculated various performance metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean - - - - Absolute Error (MAE), R-squared (R2), and Explained Variance.
- Analyzed the percentage of predictions within different tolerance levels.
- Examined the minimum and maximum percentage differences between predicted and actual values.
- Evaluated quantile regression loss at different percentiles.
- Conducted k-fold cross-validation to assess model robustness.

Result Visualization:
- Generated descriptive plots including actual vs. predicted values, histograms of predicted Li concentration values, histograms of residuals, and Q-Q plots of residuals.

Data Export:
- Saved individual result DataFrames to CSV files.
- Concatenated the results for further analysis.

<!-- ============================================================================================================== -->
---

SVR Regression Model:

Support Vector Regression (SVR) is a supervised learning algorithm used for regression tasks. It works by finding a hyperplane in an N-dimensional space (where N is the number of features) that best fits the data. SVR differs from traditional regression models by focusing on points within a certain margin around the predicted line (or hyperplane in higher dimensions), rather than fitting all data points precisely.

Results Metrics:

Mean Squared Error (MSE):
- Measures the average of the squares of errors between predicted and actual values. Lower values indicate better model performance.

Root Mean Squared Error (RMSE):
- Represents the square root of the MSE. RMSE provides the estimate of the standard deviation of residuals. It is interpretable in the same units as the target variable.

Mean Absolute Error (MAE):
- Computes the average of the absolute differences between predicted and actual values. It is less sensitive to outliers compared to MSE.

R-squared (R2):
- Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating a better fit.

Explained Variance:
- Represents the proportion of variance in the target variable explained by the model. Higher values indicate better performance.

Percentage of Predictions within Tolerance Levels:
- Measures the percentage of predictions falling within certain percentage ranges of the actual values. Higher percentages indicate better accuracy.

Quantile Regression Loss:
- Evaluates the model's performance at different quantiles of the target variable's distribution.

<!-- ============================================================================================================== -->
---

Key Findings:

Variability in Model Performance: There is significant variability in the performance of SVR models across different basins and data input types. For instance, the Appalachian basin shows relatively lower quantile losses and cross-validation MSE compared to other basins.

Effect of Kernel and Regularization Parameters: The choice of kernel function (e.g., rbf, poly, sigmoid) and regularization parameter (C) significantly impacts model performance. In some cases, the best-performing kernel varies across basins.

Impact of Input Data Type: Models trained with principal component analysis (PCA) input data exhibit different performance characteristics compared to models trained without PCA. PCA-based models generally show higher quantile losses and cross-validation MSE, suggesting that PCA may not always be beneficial for improving model accuracy.

Outlier Detection: Some basins, such as Fort Worth and Illinois, exhibit unusually high percentage differences and quantile losses, indicating potential issues with model performance or data quality in these regions.

Conclusion:

The performance of SVR models in predicting reservoir properties varies significantly across different geological basins and input data types. While certain basins demonstrate relatively accurate predictions with low quantile losses and cross-validation MSE, others exhibit higher levels of variability and uncertainty. Further investigation into the underlying geological characteristics and data preprocessing methods may help improve model accuracy and robustness.