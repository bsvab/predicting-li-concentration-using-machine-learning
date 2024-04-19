# Geochemical Data Analysis and Lithium Concentration Prediction Using Machine Learning Techniques

## Introduction

In this report, we present a comprehensive analysis of geochemical data of produced water samples, obtained from the U.S. Geological Survey (https://www.sciencebase.gov/catalog/item/64fa1e71d34ed30c2054ea11). Our objective is to apply statistical and machine learning techniques to predict Lithium (Li) concentrations. This analysis encompasses data preprocessing, Principal Component Analysis (PCA), clustering analysis, regression analysis, and hypothesis testing.

## Data Preparation

The raw dataset was subjected to several preprocessing steps to ensure its readiness for analysis:

- **General Cleaning**: Irrelevant columns, such as identifiers and categorical data, were removed. This step was crucial to focus the analysis on quantitative geochemical features.
- **Well Type Filtering**: To ensure our study focused on the high-salinity produced water typical in oil and gas operations, we excluded well samples from coal, injection, and geothermal plays. This approach keeps our analysis relevant to the typical environmental and production challenges in the oil and gas industry.
- **Basin Categorization**: Well samples were categorized based on the basin they belong to. This categorization helps to group samples by geographical and geological features.
- **Technical Cleaning**: In the dataset preprocessing phase, observations with Total Dissolved Solids (TDS) levels below 10,000 parts per million (ppm) were excluded. This threshold was established based on the consideration that waters with TDS levels below this mark are typically characterized as brackish and are often subjected to treatment processes for various uses, including agricultural and industrial applications. Conversely, waters with TDS exceeding 10,000 ppm are generally associated with greater environmental challenges, requiring more rigorous management strategies to mitigate potential adverse ecological impacts. By focusing on samples with higher TDS levels, the analysis targets the subset of produced water that is more likely to raise environmental and disposal concerns, thus providing a more relevant framework for examining geochemical data in the context of environmental science and hydrology.

## Machine Learning Models and Scenarios
We employed various machine learning models including Gradient Boosting, Random Forest, Neural Networks, K-Nearest Neighbor, and Support Vector Regression across two scenarios:

- **Scenario 1 (Imputed Dataset)**: Utilized an enhanced dataset with imputed missing values.
- **Scenario 2 (PCA-Transformed Dataset)**: Applied PCA to reduce dimensionality, focusing on the first 10 principal components capturing over 90% of data variance.

### Imputation Strategy
  For geochemical datasets, where the variables often exhibit complex interdependencies, simplistic imputation methods such as substitution by mean, median, or extremities like minimum and maximum values may not adequately capture the intrinsic variability and can potentially introduce biases. To address the missing data in our dataset, we utilized an Iterative Imputer, employing a RandomForestRegressor as the estimator. This advanced imputation technique considers the entire variable distribution and the stochastic nature of the dataset, thereby preserving the inherent multivariate relationships among geochemical parameters. It is especially critical in PCA, which requires a complete dataset as missing values can significantly distort the principal components derived from the analysis.

## Geochemical Data Analysis

### Clustering Analysis

A clustering analysis using the K-Means algorithm was conducted, revealing distinct spatial clusters in the data:

- **Spatial Clustering**: The latitude and longitude were used to identify six clusters, potentially indicating geochemical similarities based on location.

![Distance Clustering Plot](/images/geomaps/distance_clustering_plot.png)
*Figure 1: Spatial clustering of well samples.*

- **Geo-visualization**: A geographic map with clustered data points was created using Folium, offering insights into spatial patterns related to geochemical features.

![Distance Cluster Map](/images/geomaps/distance_clustering_map.PNG)
*Figure 2: Map visualization of clustered well samples.*

### Regression Analysis

We explored the relationships between Li concentration and other geochemical parameters using regression analysis:

- **Li Concentration vs. TDS**: A statistically significant positive correlation was identified, implying a relationship between increased TDS and Li concentration.
- **Li Concentration vs. Depth**: Analysis showed a positive correlation, suggesting that depth might influence Li concentration in the wells.

### Hypothesis Testing

Statistical tests were conducted to verify the significance of the relationships observed:

- **Li and TDS**: The strong correlation was confirmed to be statistically significant, indicating a genuine relationship across the dataset.
- **Li and Depth**: The positive correlation was statistically significant, albeit with a lower correlation coefficient compared to the Li-TDS correlation.


## Heatmap Analysis
Through heatmap analysis, we have examined the correlations between Lithium (Li) concentrations and various geochemical parameters across multiple basins. The following section elucidates the distinct geochemical signatures observed in each basin and discusses the implications of these findings for predicting Lithium concentrations.

### Basin-Specific Correlations with Lithium
Our heatmap analysis yielded a wealth of insights, pinpointing the top three geochemical features most correlated with Lithium in each basin:

- **Appalachian Basin**: Strontium (Sr) and Barium (Ba) emerged as the most correlated features with Lithium, with correlation coefficients of 0.557 and 0.529 respectively.

- **Permian Basin**: Zinc (Zn) displayed a remarkably high correlation with Lithium at 0.774, followed by Sr and Sodium (Na).

- **Oklahoma Platform Basin**: Here, Sr was again prominent with a correlation of 0.529, alongside Potassium (K) and Calcium (Ca), with correlations of 0.425 and 0.365.

- **Gulf Coast Basin**: Boron (B) and Bromine (Br) showed strong correlations with Lithium at 0.634 and 0.632, respectively, while Ca also demonstrated a notable correlation of 0.546.

- **Williston Basin**: Exhibited extremely high correlations with Barium and Boron at around 0.98 for both, indicating a significant geochemical interplay with Lithium.

- **Michigan Basin**: Zinc displayed a perfect correlation with Lithium, followed by Br and Ca with strong correlations of 0.845 and 0.755.

- **Pacific Basin**: Featured Bicarbonate (HCO3) as the most correlated parameter with Lithium at 0.428, suggesting geochemical processes where HCO3 may be a determinant in Lithium concentration.

- **Illinois Basin**: Highlighted K as the leading correlating feature with a high coefficient of 0.845, with Ca and Ba also showing strong relationships with Lithium.

- **Great Plains Basin**: Indicated that Ba, Br, and Sr are significant contributors to Lithium variability, with correlation coefficients ranging from 0.559 to 0.585.

- **Anadarko Basin**: Sulfate (SO4) was the most correlated with Lithium at 0.592, along with notable correlations with K and Ba.

- **Rocky Mountain Basin**: Presented Na, TDS, and Chloride (Cl) as the most correlated with Lithium, though the coefficients were modest, all hovering around 0.3.

- **Fort Worth Basin**: Iron Total (FeTot) had a perfect correlation with Lithium, while Ca and Ba also showed very strong correlations, with coefficients above 0.86.

### Implications for Machine Learning
The disparities in geochemical correlations with Lithium across basins underscore the complexity of geochemical interactions within distinct geological settings. In some basins, certain elements exhibit extremely strong, if not perfect, correlations with Lithium, suggesting a potential for direct predictive modeling of Lithium concentrations based on these elements.

In other basins, such as the Rocky Mountain, the correlations are less pronounced, which may necessitate a more nuanced approach to modeling that incorporates a broader array of features or perhaps the development of basin-specific prediction models.

For machine learning applications, these findings inform feature selection, allowing us to tailor our predictive models to include the most relevant geochemical parameters for each basin. The highlighted correlations will serve as the foundation for the feature sets in regression analysis and other predictive modeling techniques aimed at estimating Lithium concentrations.

By concentrating on the most influential parameters as indicated by the heatmap analysis, we aim to enhance model accuracy and interpretability while mitigating the risk of multicollinearity and overfitting.

## Principal Component Analysis (PCA)
Principal Component Analysis (PCA) was applied to the geochemical data from each basin separately. The findings from this dimensionality reduction technique provided insights into the underlying data structure and the interrelations among geochemical variables, as summarized below for each basin:

### Anadarko Basin
- **Cumulative Explained Variance**: Indicates a substantial amount of the variance is captured by the first 10 principal components, showing that they are significant in representing the dataset.
- **Biplot and Loadings Plot**: Highlights the geochemical variables like SO4, K, and Mg that contribute prominently to the variance within the basin.
- **PCA Scatter Plot**: The spread of data points predominantly along the first principal component suggests that it captures a significant variation within the basin.
- **Scree Plot**: Demonstrates a quick decline in the explained variance ratio, indicating that 90% of the information is concentrated in the first few principal components.
### Appalachian Basin
- **Cumulative Explained Variance**: The curve suggests that first 10 principal components are sufficient to explain more than 90% of the variance in the data.
- **Biplot and Loadings Plot**: Shows variables such as Ba, Na, and Sr exerting a strong influence on the components, which are indicative of the geochemical composition in the basin.
- **PCA Scatter Plot**: Reveals a distribution of samples that is wide along the first principal component, suggesting significant variability across the geochemical signatures.
- **Scree Plot**: The steep slope of the initial components followed by a leveling off indicates that the dataset's dimensionality can be effectively reduced without substantial loss of information.
### Fort Worth Basin
- **Cumulative Explained Variance**: The plot shows that a significant portion of the total variance (more than 90%) is explained by the first 10 components, with a plateau suggesting additional components add less information.
- **Biplot and Loadings Plot**: Illustrates that geochemical parameters like TDS, Mg, and certain ions are influential in the dataset, contributing strongly to the first two components.
- **PCA Scatter Plot**: Demonstrates variability in geochemical composition, with samples spread primarily along the first principal component.
- **Scree Plot**: Indicates that the majority of the variance is captured by the first few components, with a steep decline in additional explained variance thereafter.
### Great Plains Basin
- **Cumulative Explained Variance**: Similar to other basins, the first 10 principal components account for the bulk of the variance, suggesting a strong pattern within the data.
- **Biplot and Loadings Plot**: Shows that specific variables, including TDS, Ba, and K, are prominent, reflecting their significant role in the geochemical variance across samples.
- **PCA Scatter Plot**: Reveals a clustering along the first principal component, indicative of a dominant geochemical signature or factor within the basin.
- **Scree Plot**: The explained variance decreases rapidly after the initial components, reinforcing the idea that the first few components can represent the dataset effectively.
### Gulf Coast Basin
- **Cumulative Explained Variance**: The first 10 principal components capture more than 90% of the variance, with the curve plateauing, indicating that the rest of the components add minimal information.
- **Biplot and Loadings Plot**: Indicated which variables have the strongest influence on the components. Key geochemical parameters such as TDS and specific ions like Br and B are prominently featured.
- **PCA Scatter Plot**: The spread of data points suggests variability in geochemical signatures across the samples, though no clear distinct groupings are observed when considering only the first two components.
- **Scree Plot**: Illustrated a steep drop after the first few components, suggesting that a small number of components can be used to capture the majority of the data variance.
### Illinois Basin
- **Cumulative Explained Variance**: Demonstrated a similar pattern to the Gulf Coast basin, with early 10 components explaining most variance.
- **Biplot and Loadings Plot**: Highlighted elements such as K, Ca, and Ba as major contributors to variance in the first two principal components.
- **PCA Scatter Plot**: Showed a distribution of data points along the first principal component with less spread along the second, indicating that the first component captures a significant variation aspect.
- **Scree Plot**: Showed that additional components beyond the first few contribute incrementally less to the explained variance.
### Michigan Basin
- **Cumulative Explained Variance**: The curve showed that a considerable amount of variance is explained by the initial components.
- **Biplot and Loadings Plot**: Zn, Br, and Ca were prominent, suggesting their strong association with the variability in this basin.
- **PCA Scatter Plot**: Displayed a wide dispersion of data points along the first principal component, which may be reflective of the varied geochemical environment of the Michigan basin.
- **Scree Plot**: Reinforced the significance of the first couple of components, after which the explained variance ratio decreases more gradually.
### Oklahoma Platform Basin
- **Cumulative Explained Variance**: The variance explained by the principal components shows a strong start, with a rapid accumulation within the first few components before leveling off. This suggests that most of the geochemical variability can be captured by the initial principal components.
- **Biplot and Loadings Plot**: The biplot reveals that geochemical parameters such as K, SO4, and Mg have a pronounced impact on the variance, with these variables pointing in the direction of the vectors, indicating their strong influence on the components.
- **PCA Scatter Plot**: The scatter of samples predominantly along the first principal component indicates a significant geochemical gradient, with less variability explained by the second component. The dispersion pattern might suggest distinct geochemical processes influencing the composition of the samples.
- **Scree Plot**: The scree plot exhibits a sharp decrease after the first couple of principal components, reinforcing the idea that the most meaningful information is concentrated in the initial components, beyond which the marginal gain in explained variance significantly drops.
### Pacific Basin
- **Cumulative Explained Variance**: A smooth curve indicated that the variance is well explained by the PCA components, with a clear point of diminishing returns.
- **Biplot and Loadings Plot**: Indicates that geochemical parameters like sodium (Na) and total dissolved solids (TDS) are among the variables with the strongest influence on the components. This is evident from their vector lengths and directions, which imply how these variables correlate with the principal components and with each other.
- **PCA Scatter Plot**: Suggested heterogeneity in the geochemical data with the first two components revealing a spread in sample composition.
- **Scree Plot**: The variance explained by subsequent components drops off, indicating the possibility of a lower-dimensional representation.
### Permian Basin
- **Cumulative Explained Variance**: The first few components accounted for a large portion of the variance, with a plateau indicating fewer components could be used without significant loss of information.
- **Biplot and Loadings Plot**: Illustrated the geochemical variables that are most impactful, with certain elements like Zn and SO4 having significant loadings.
- **PCA Scatter Plot**: The scatter of points mostly along the first principal component suggests that it captures a key variance aspect within the basin's geochemical data.
- **Scree Plot**: Showed a steep initial slope, indicating that the first few principal components are the most informative.
### Rocky Mountain Basin
- **Cumulative Explained Variance**: The curve suggested that initial components significantly capture the geochemical data variance.
- **Biplot and Loadings Plot**: Showed the dominance of geochemical parameters such as SO4 and Br in the dataset variability.
- **PCA Scatter Plot**: The spread of points across the first component highlighted the heterogeneity of the geochemical signatures.
- **Scree Plot**: Confirmed that the first few components contain most of the informative variance.
### Williston Basin
- **Cumulative Explained Variance**: Exhibits a steep curve, indicating that initial components are crucial in explaining the variance.
- **Biplot and Loadings Plot**: Highlights the dominance of geochemical indicators like Na and Cl.
- **PCA Scatter Plot**: Shows a concentration of samples along the first component, suggestive of a key variance factor.
- **Scree Plot**: The explained variance by each subsequent component drops markedly after the initial few components.

## Predictive Modeling

### Gradient Boosting for Lithium Concentration Prediction

Employing Gradient Boosting Regression, we derived predictions for Li concentrations, relying on performance metrics for model validation and leveraging visual plots for a comparative assessment of actual vs. predicted values. This approach was chosen due to its robustness in handling nonlinear relationships and its ability to handle the variance explained by the principal components effectively.

### Methodology

Gradient Boosting Regressor from scikit-learn, configured with 100 estimators, a 0.1 learning rate, and a maximum depth of 3, was employed to balance complexity with performance. Both the imputed dataset and the PCA-transformed dataset were split into training and testing sets, allocating 20% for evaluation.

### Model Training and Evaluation

For each basin, the model was trained on the known Lithium concentrations and evaluated using the Mean Squared Error (MSE), R-squared (R2), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Explained Variance. These metrics provide a comprehensive view of the model's accuracy and its ability to generalize to new data.

- **MSE and RMSE** offer measures of the average error magnitude, with RMSE giving more weight to larger errors.
- **MAE** provides a straightforward average of error magnitudes.
- **R2** indicates the proportion of variance in the dependent variable predictable from the independent variables.
- **Explained Variance** measures how well our model accounts for the variation in the dataset.

### Scenario 1 (Imputed Data)
Each basin’s model was trained on known Li concentrations and evaluated against a set of metrics—Mean Squared Error (MSE), R-squared (R2), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Explained Variance—to quantify accuracy and predictive power.

#### Anadarko Basin
- The actual vs. predicted scatter plot displays a tight clustering of points around the diagonal, indicative of the model’s predictive accuracy.
- The distribution histogram for new samples skews significantly towards lower Li concentrations, suggesting the prevalence of lower concentration levels within this basin.
#### Appalachian Basin
- The scatter plot reveals a fairly linear relationship between predicted and actual values, although some variance is noted, particularly at higher concentration levels.
- Predicted values for new samples exhibit a wide distribution, signaling diverse Li concentrations within the basin.
#### Fort Worth Basin
- The scatter plot illustrates a strong alignment for lower concentrations, while a few outliers suggest exceptional cases beyond the model's typical prediction scope.
- The distribution of predicted concentrations for new samples indicates a concentration of predictions within the low to moderate range.
#### Great Plains Basin
- The scatter plot shows that the model performs well, accurately predicting most of the Li concentrations, though a few outliers exist.
- The histogram of predicted values for new samples features a multimodal distribution, implying various geochemical profiles within the basin.
#### Gulf Coast Basin
- The scatter plot indicates an excellent match between predicted and actual Li concentrations, underscoring the model's ability to capture the geochemical dynamics of the basin.
- The histogram for new sample predictions portrays a high frequency of lower concentrations and a tail extending to higher values, hinting at the potential for both common low Lithium zones and rarer high Lithium pockets.

### Illinois Basin
- Actual vs. Predicted: Predictions were in close agreement with actual measurements, although there was a tendency to underestimate at higher Lithium levels.
- Predicted Distribution: A spread of predicted values indicates varied Lithium presence across new samples.
### Michigan Basin
- Actual vs. Predicted: The model performance appeared robust for lower concentrations, with divergence occurring at higher concentration levels.
- Predicted Distribution: New samples exhibited a wide range of predicted Lithium concentrations, with a notable peak at lower values.
### Oklahoma Platform Basin
- Actual vs. Predicted: Demonstrated good concordance between predicted and actual values with minimal variance.
- Predicted Distribution: The histogram reflected a prevalence of moderate Lithium concentrations, with less frequent occurrences of extreme values.
### Pacific Basin
- Actual vs. Predicted: The data showed a tight cluster of predictions closely following the line of unity, which suggests high accuracy.
- Predicted Distribution: Predicted concentrations for new samples displayed a prominent peak at lower values, with fewer predictions of high Lithium concentrations.
### Permian Basin
- Actual vs. Predicted: A consistent model performance was observed, particularly for lower actual Lithium concentrations, with some scatter at higher concentrations.
- Predicted Distribution: The distribution was left-skewed, indicating a predominance of lower Lithium concentrations in the new sample predictions.
### Rocky Mountain Basin
- Predictions for lower concentrations align well, but there's deviation at higher actual concentrations. 
- The histogram for new samples peaks at lower concentrations, suggesting this as the common Lithium level.

### Williston Basin
- The actual vs. predicted plot demonstrates a strong correlation, indicating reliable model predictions. 
- The distribution for new samples peaks at lower concentrations, likely indicative of the basin's typical Lithium distribution.

### Scenario 2 (PCA-Transformed Data)
For each basin, predictions were visualized against actual values to assess the model's accuracy visually. These plots highlight the alignment between predicted and actual concentrations, with a perfect prediction falling along the diagonal line.

Additionally, for basins where Lithium values were missing, the model provided predictions, which were saved and visualized in a histogram to understand the distribution of predicted values.

 here are short interpretations for each basin:
#### Anadarko Basin
- The actual vs. predicted plot shows a concentration of predictions closely hugging the line of perfect agreement, indicating a model with a tight fit for most data points.
- The distribution of predicted concentrations for new samples skews heavily towards lower concentrations, suggesting that Lithium-rich areas in this basin might be less common.
#### Appalachian Basin
- Predictions are generally in line with the actual values, with a spread indicating some variance in model accuracy.
- Predicted values for new samples show a wide distribution, implying varied Lithium concentrations across the basin.
#### Fort Worth Basin
- A close match between actual and predicted values, with outliers potentially indicating unique geochemical scenarios or model limitations.
- The distribution plot suggests most new samples are predicted to have low to moderate Lithium concentrations.
#### Great Plains Basin
- The actual vs. predicted values indicate a strong model performance for the majority of the data with some outliers.
- New sample predictions show a multimodal distribution, hinting at different geochemical environments within the basin.
#### Gulf Coast Basin
- Actual vs. predicted concentrations are closely aligned, suggesting the model captures the basin's geochemistry well.
- Predicted Lithium concentrations for new samples have a high frequency at lower concentrations, with a long tail towards higher values, indicating potential areas of high Lithium presence.
#### Illinois Basin
- Predictions align well with actual concentrations, though the model appears to slightly underestimate at higher Lithium concentrations.
- New sample predictions are somewhat evenly distributed, with multiple peaks, possibly reflecting diverse geochemical conditions.
#### Michigan Basin
- The model seems to perform well for lower concentrations but shows some divergence at higher values.
- Predicted new sample concentrations present a broad range, suggesting that while low Lithium concentrations are common, pockets of higher concentration exist.
#### Oklahoma Platform Basin
- The plot displays a good correlation between predicted and actual values with a minor spread.
- Predicted Lithium concentration distribution indicates a higher frequency of moderate Lithium values, with fewer samples predicted to have very low or high concentrations.
#### Pacific Basin
- Actual vs. predicted Lithium concentrations show a tight grouping along the line of perfect prediction.
- Distribution of predicted concentrations for new samples is multi-peaked, suggesting distinct geochemical zones within the basin.
#### Permian Basin
- There is a solid correlation for lower concentrations, with some deviation at higher values.
- The predicted concentration distribution for new samples is highly left-skewed, indicating that higher Lithium concentrations are rare in this basin.
#### Rocky Mountain Basin
- For the Rocky Mountain Basin, the actual vs. predicted plot shows that the model predicts lower concentrations well but deviates at higher actual concentrations. 
- The predicted distribution has a prominent peak at -lower concentrations, indicating a consistency in the predicted Lithium levels for most new samples.
#### Williston Basin
- The actual vs. predicted Lithium concentrations in the Williston Basin display a strong correlation along the diagonal, suggesting the model's predictions are reliable. 
- The distribution of predicted values for new samples shows a strong peak at lower concentrations, tapering off for higher values, which might be representative of the prevalent Lithium distribution in this basin.

These interpretations provide a quick snapshot of the model's performance and the nature of Lithium distribution in each basin, as per the provided plots. It's important to consider these alongside geochemical insights for a comprehensive understanding of the factors influencing Lithium levels in these regions.

### Comparative Analysis
The GB model's ability to predict Lithium concentrations varies greatly depending on the basin, which suggests that basin-specific features and geochemical signatures significantly influence model performance. 
Only the Williston Basin without PCA and the Michigan Basin without PCA meet the criteria for good predictions with R^2 scores above 0.75. The Pacific Basin without PCA is just on the threshold and may also be considered an adequate model. The remaining scenarios for other basins fail to meet the 0.75 threshold for R^2, which means they are not considered good predictors of Lithium concentrations according to the specified criterion. This emphasizes the need for using different modeling approaches for these basins to improve prediction quality.

Performance Summary with R^2 Threshold of 0.75:

- Appalachian Basin: Both scenarios fall below the threshold, but the model without PCA is closer to being acceptable.
- Permian Basin: Both scenarios show poor predictions with R^2 values well below the threshold.
- Oklahoma Platform: Neither scenario meets the threshold; predictions are better without PCA but still inadequate.
- Gulf Coast Basin: Both scenarios are below the threshold; however, without PCA, the performance is closer to the threshold.
- Williston Basin: The model without PCA exceeds the threshold, indicating good predictions. With PCA, the model performs poorly.
- Michigan Basin: Predictions without PCA are acceptable, as the R^2 is above the threshold. With PCA, the model is insufficient.
- Pacific Basin: The model without PCA exceeds the threshold, showing good predictive performance. The PCA scenario performs poorly.
- Illinois Basin: Both scenarios do not meet the threshold; model performance is poor.
- Great Plains Basin: Both scenarios are well below the threshold, indicating very poor predictions.
- Anadarko Basin: The PCA scenario is closer to the threshold, but both are below 0.75, indicating poor predictions.
- Rocky Mountain Basin: Neither scenario reaches the threshold; however, the PCA scenario is closer to being acceptable.
- Fort Worth Basin: Both scenarios are below the threshold, suggesting poor predictions.



### Random Forest for Lithium Concentration Prediction

### Methodology
The Random Forest models were implemented using the scikit-learn library, optimized for geochemical data analysis to predict lithium concentrations. Each model configuration was adapted to handle different data scenarios effectively:
- **Random Forest Setup**: The model utilized a Random Forest Regressor, an ensemble learning method known for its robustness and accuracy in regression tasks. The forest comprised multiple decision trees, whose parameters were fine-tuned using hyperparameter optimization.
- **Hyperparameter Tuning**: A grid search was conducted to find the optimal model settings, utilizing a parameter grid that included:

    - ***Number of Estimators***: Options of 100, 200, 300 trees to explore the effect of forest size on performance.
    - ***Maximum Depth***: Tree depths of None (fully grown trees), 10, and 20 to control overfitting.
    - ***Minimum Samples Split***: Minimum number of samples required to split a node set at 2, 5, and 10, impacting the model's sensitivity to data variance.
    - ***Minimum Samples Leaf***: Minimum number of samples required at a leaf node, tested with 1, 2, and 4, which helps in stabilizing the predictions across diverse data points.

For data handling, two scenarios were utilized:

- **PCA-Transformed Data**: The models were fed the first 10 principal components, representing over 90% of the variance in the dataset, to reduce dimensionality and focus on the most significant features.
- **Imputed Data Without PCA**: The models used the full set of imputed geochemical features to capture all potential interactions and provide a comprehensive basis for prediction.

Both datasets were divided into training and testing sets, with 80% of the data allocated for model training and 20% reserved for model evaluation.

### Scenario 1 (Imputed Data)
#### Anadarko Basin:
 - A relatively good fit was observed with some predictions close to the perfect agreement line, but with variability at higher concentrations.
 - Most predicted new samples were within lower concentration ranges.
#### Appalachian Basin:
 - The model showed a general alignment with actual values, suggesting a reasonably accurate prediction with variability.
#### Fort Worth Basin:
 - There was a close match between actual and predicted values for the lower concentration range, with sparse data at higher concentrations.
#### Great Plains Basin:
 - The model performance indicated a good fit for the majority of the data, with some discrepancies at higher concentrations.
#### Gulf Coast Basin: 
 - Predictions closely aligned with actual concentrations, indicating a good model fit across various concentrations.
#### Illinois Basin:
 - The model predictions were in line with the actual values, with minor underestimation at higher concentrations.
#### Michigan Basin:
 - The model performed well, particularly at lower concentrations, with some spread at higher concentrations.
#### Oklahoma Platform Basin:
 - There was a general correlation between predicted and actual values, with a concentration of predictions within the moderate range.
#### Pacific Basin:
 - Tight grouping along the line of perfect prediction was observed, indicating good model performance.
#### Permian Basin:
 - A solid correlation was observed for lower concentrations, while higher values showed some deviation.
#### Rocky Mountain Basin:
- There was a general correlation between predicted and actual values, with a concentration of predictions within the moderate range.
#### Williston Basin:
 - Tight grouping along the line of perfect prediction was observed, indicating good model performance.

### Scenario 2 (PCA-transformed Data)

In this scenario, we used Principal Component Analysis (PCA) as a pre-processing step to reduce the dimensionality of the feature space before applying Random Forest regression.
Anadarko Basin:

The Actual vs. Predicted plot shows that lower concentrations of lithium are well-predicted, but higher concentrations show greater variance.
The predicted distribution histogram indicates a high frequency of samples with low predicted lithium concentration, suggesting the model predicts lower concentrations more frequently.

#### Appalachian Basin:
- The model predictions are closely aligned with the actual values, particularly at lower concentrations, indicating good model performance.
- The distribution histogram shows a smooth decrease in frequency as the predicted lithium concentration increases.
#### Fort Worth Basin:
- There is a tight clustering of points along the line of perfect prediction in the Actual vs. Predicted plot, which suggests excellent model performance.
- The distribution plot reveals a fairly uniform spread across the range of predicted concentrations.
#### Great Plains Basin:
- Predictions are scattered at higher lithium concentrations, which may indicate potential outliers or variance in the data.
- The histogram shows a wide distribution of predicted concentrations, with a peak at lower values.
#### Gulf Coast Basin:
- The Actual vs. Predicted scatter plot suggests a consistent over-prediction for lower actual lithium values.
- The histogram of predicted concentrations is heavily skewed towards the lower end, with a long tail towards higher concentrations.
#### Illinois Basin:
- The Actual vs. Predicted plot shows a general agreement between actual and predicted values, with some deviations at higher concentrations.
- The predicted lithium concentrations are most frequently on the lower end, as seen in the histogram.
#### Michigan Basin:
- The model shows a good fit for lower concentrations but some divergence at higher concentrations in the Actual vs. Predicted plot.
- The distribution plot indicates a multi-modal frequency, with several peaks across the predicted concentration range.
#### Oklahoma Platform:
- The Actual vs. Predicted plot reveals that predictions are quite consistent with actual values for this basin.
- Predicted concentrations have a high frequency at the lower end with a gradual decrease, showing a right-skewed distribution.
#### Pacific Basin:
- The Actual vs. Predicted scatter plot shows a compact clustering, especially at lower concentration levels.
- The distribution histogram indicates that the majority of predictions are for lower concentrations, with fewer predictions for higher concentrations.
#### Permian Basin:
- The Actual vs. Predicted plot shows a decent model fit with some outliers present.
- The distribution histogram has a clear peak at the lower end, indicating the model's tendency to predict lower lithium concentrations.
#### Rocky Mountain Basin:
- The Actual vs. Predicted plot indicates that the model predicts lower concentrations well, but there's variability in higher concentrations.
- The histogram shows a high frequency of predictions at lower concentrations, with a steep decline towards the higher end.
#### Williston Basin:
- There is a good alignment between the actual and predicted values in the scatter plot, indicating strong model performance.
- The histogram shows a relatively even distribution of predicted concentrations across the range.

### Comparative Analysis
- Gulf Coast, Williston, and Michigan Basins show robust model performance across both PCA-transformed and imputed data scenarios, with each consistently surpassing or nearing the 0.75 R^2 threshold. These basins benefit significantly from both types of data processing, indicating strong predictive capabilities that are likely due to more homogeneous geochemical properties or better data coverage.

- Rocky Mountain and Anadarko Basins exhibit improvement when using imputed data without PCA compared to PCA-transformed data. This suggests that retaining the full set of geochemical features without the reduction by PCA might be capturing essential variances better in these basins.

- Great Plains Basin and a few others show consistently poor performance in both models, indicating challenges that might be due to data quality issues, such as limited data variability or skewness towards lower lithium concentrations.

## Neural Network for Lithium Concentration Prediction

### Methodology
The Neural Network models were constructed using Keras, with a configuration optimized for geochemical data analysis. Each model comprised an input layer, two hidden layers, and an output layer. The first hidden layer had 128 neurons, and the second had 64 neurons, both utilizing the ReLU activation function for non-linearity. The output layer consisted of a single neuron for predicting lithium concentrations. The Adam optimizer was employed with a learning rate of 0.001, and the loss function used was mean squared error to minimize the prediction error.

For data handling, two scenarios were utilized:

- **PCA-Transformed Data**: The models were fed the first 10 principal components, representing over 90% of the variance in the dataset, to reduce dimensionality and focus on the most significant features.
- **Imputed Data Without PCA**: The models used the full set of imputed geochemical features to capture all potential interactions and provide a comprehensive basis for prediction.

Both datasets were divided into training and testing sets, with 80% of the data used for training and 20% reserved for testing, to evaluate model performance and generalizability.

### Model Performance Across Geological Basins
The performance analysis of Neural Network models on PCA-transformed and imputed data without PCA reveals distinct patterns that underscore the influence of data preprocessing techniques on predictive accuracy in geological settings. Focusing on the R^2 value, which assesses the proportion of variance in lithium concentrations that the model can predict, we establish a threshold of 0.75 to evaluate effective predictive performance. Here’s a detailed examination of how each scenario fares across various basins:

### Predictive Performance
Models that exceed the R^2 threshold of 0.75 demonstrate a robust ability to predict lithium concentrations, suggesting that the models have successfully captured the key geochemical interactions required for accurate predictions:

- Anadarko Basin (PCA-Transformed Data): Achieving an R^2 of 0.81, the model effectively uses the principal components to capture essential data variability, indicating that the PCA approach is well-suited for this basin’s geochemical data structure.
- Fort Worth Basin (PCA-Transformed Data): With an R^2 of 0.87, this basin shows excellent model performance, suggesting that the dominant geochemical features influencing lithium concentrations are well-represented by the top principal components.
- Gulf Coast Basin (PCA-Transformed Data): An R^2 of 0.85 highlights the model's capability to efficiently utilize reduced dimensions to predict lithium concentrations accurately, reinforcing the effectiveness of PCA in maintaining crucial information while reducing noise.
- Appalachian Basin: Experiencing an R^2 of 0.60 with PCA and even lower without PCA, these results suggest that neither data handling strategy adequately captures the complex geochemical interactions necessary for accurate lithium prediction in this region.
- Great Plains Basin (PCA-Transformed Data): The near-zero R^2 value indicates a failure in the model's ability to generalize the data's underlying patterns, possibly due to the inherently complex geochemical background that requires more data.

