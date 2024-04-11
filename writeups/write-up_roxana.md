# Geochemical Data Analysis and Lithium Concentration Prediction Using Machine Learning Techniques

## Introduction

In this report, we present a comprehensive analysis of geochemical data of produced water samples, obtained from the U.S. Geological Survey (https://www.sciencebase.gov/catalog/item/64fa1e71d34ed30c2054ea11). Our objective is to apply statistical and machine learning techniques to predict Lithium (Li) concentrations. This analysis encompasses data preprocessing, Principal Component Analysis (PCA), clustering analysis, regression analysis, and hypothesis testing.

## Data Preparation

The raw dataset was subjected to several preprocessing steps to ensure its readiness for analysis:

- **General Cleaning**: Irrelevant columns, such as identifiers and categorical data, were removed. This step was crucial to focus the analysis on quantitative geochemical features.
- **Well Type Filtering**: To ensure our study focused on the high-salinity produced water typical in oil and gas operations, we excluded well samples from coal, injection, and geothermal plays. This approach keeps our analysis relevant to the typical environmental and production challenges in the oil and gas industry.
- **Basin Categorization**: Well samples were categorized based on the basin they belong to. This categorization helps to group samples by geographical and geological features.
- **Technical Cleaning**: In the dataset preprocessing phase, observations with Total Dissolved Solids (TDS) levels below 10,000 parts per million (ppm) were excluded. This threshold was established based on the consideration that waters with TDS levels below this mark are typically characterized as brackish and are often subjected to treatment processes for various uses, including agricultural and industrial applications. Conversely, waters with TDS exceeding 10,000 ppm are generally associated with greater environmental challenges, requiring more rigorous management strategies to mitigate potential adverse ecological impacts. By focusing on samples with higher TDS levels, the analysis targets the subset of produced water that is more likely to raise environmental and disposal concerns, thus providing a more relevant framework for examining geochemical data in the context of environmental science and hydrology.
- **Imputation Strategy**:  For geochemical datasets, where the variables often exhibit complex interdependencies, simplistic imputation methods such as substitution by mean, median, or extremities like minimum and maximum values may not adequately capture the intrinsic variability and can potentially introduce biases. To address the missing data in our dataset, we utilized an Iterative Imputer, employing a RandomForestRegressor as the estimator. This advanced imputation technique considers the entire variable distribution and the stochastic nature of the dataset, thereby preserving the inherent multivariate relationships among geochemical parameters. It is especially critical in PCA, which requires a complete dataset as missing values can significantly distort the principal components derived from the analysis.

## Clustering Analysis

A clustering analysis using the K-Means algorithm was conducted, revealing distinct spatial clusters in the data:

- **Spatial Clustering**: The latitude and longitude were used to identify six clusters, potentially indicating geochemical similarities based on location.

![Distance Clustering Plot](/images/geomaps/distance_clustering_plot.png)
*Figure 1: Spatial clustering of well samples.*

- **Geo-visualization**: A geographic map with clustered data points was created using Folium, offering insights into spatial patterns related to geochemical features.

![Distance Cluster Map](/images/geomaps/distance_clustering_map.PNG)
*Figure 2: Map visualization of clustered well samples.*

## Regression Analysis

We explored the relationships between Li concentration and other geochemical parameters using regression analysis:

- **Li Concentration vs. TDS**: A statistically significant positive correlation was identified, implying a relationship between increased TDS and Li concentration.
- **Li Concentration vs. Depth**: Analysis showed a positive correlation, suggesting that depth might influence Li concentration in the wells.

## Test Hypothesis

Statistical tests were conducted to verify the significance of the relationships observed:

- **Li and TDS**: The strong correlation was confirmed to be statistically significant, indicating a genuine relationship across the dataset.
- **Li and Depth**: The positive correlation was statistically significant, albeit with a lower correlation coefficient compared to the Li-TDS correlation.


## Heatmap Analysis
Through heatmap analysis, we have examined the correlations between Lithium (Li) concentrations and various geochemical parameters across multiple basins. The following section elucidates the distinct geochemical signatures observed in each basin and discusses the implications of these findings for predicting Lithium concentrations.

### Basin-Specific Correlations with Lithium
Our heatmap analysis yielded a wealth of insights, pinpointing the top three geochemical features most correlated with Lithium in each basin:

- **Appalachian Basin**: Strontium (Sr), Barium (Ba), and Total Dissolved Solids (TDS) emerged as the most correlated features with Lithium, with correlation coefficients of 0.557, 0.529, and 0.273, respectively.

- **Permian Basin**: Zinc (Zn) displayed a remarkably high correlation with Lithium at 0.774, followed by Sr and Sodium (Na) with coefficients of 0.236 and 0.194.

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

### Interpretation and Implications for Machine Learning
The disparities in geochemical correlations with Lithium across basins underscore the complexity of geochemical interactions within distinct geological settings. In basins like the Michigan and Fort Worth, certain elements exhibit extremely strong, if not perfect, correlations with Lithium, suggesting a potential for direct predictive modeling of Lithium concentrations based on these elements.

In other basins, such as the Rocky Mountain, the correlations are less pronounced, which may necessitate a more nuanced approach to modeling that incorporates a broader array of features or perhaps the development of basin-specific prediction models.

For machine learning applications, these findings inform feature selection, allowing us to tailor our predictive models to include the most relevant geochemical parameters for each basin. The highlighted correlations will serve as the foundation for the feature sets in regression analysis and other predictive modeling techniques aimed at estimating Lithium concentrations.

By concentrating on the most influential parameters as indicated by the heatmap analysis, we aim to enhance model accuracy and interpretability while mitigating the risk of multicollinearity and overfitting.

## Machine Learning Implications

- **Feature Correlation**: In predictive modeling, particularly within machine learning frameworks, the presence of highly correlated features can lead to redundancy and overfitting, commonly referred to as multicollinearity. Our exploratory data analysis indicates significant collinearity among several geochemical parameters; for instance, potassium (K) displays a strong positive correlation with boron (B), while calcium (Ca) is similarly correlated with strontium (Sr), and total dissolved solids (TDS) with chloride (Cl). To mitigate the risk of data leakage and ensure model robustness, it is prudent to consider the inclusion of only one variable from each correlated pair in the feature set. This practice enhances the generalization ability of the predictive model and improves interpretability by minimizing the effect of multicollinearity.
- **PCA for Dimensionality Reduction**: PCA can serve as a feature engineering step, reducing the dimensionality and removing multicollinearity, which is advantageous for machine learning models.


## Principal Component Analysis (PCA)
Principal Component Analysis (PCA) was applied to the geochemical data from each basin separately. The findings from this dimensionality reduction technique provided insights into the underlying data structure and the interrelations among geochemical variables, as summarized below for each basin:

### Anadarko Basin
- **Cumulative Explained Variance**: Indicates a substantial amount of the variance is captured by the initial principal components, showing that they are significant in representing the dataset.
- **Biplot and Loadings Plot**: Highlights the geochemical variables like SO4, K, and Mg that contribute prominently to the variance within the basin.
- **PCA Scatter Plot**: The spread of data points predominantly along the first principal component suggests that it captures a significant variation within the basin.
- **Scree Plot**: Demonstrates a quick decline in the explained variance ratio, indicating that most of the information is concentrated in the first few principal components (three components).
### Appalachian Basin
- **Cumulative Explained Variance**: The curve suggests that a few principal components are sufficient to explain a large proportion of the variance in the data.
- **Biplot and Loadings Plot**: Shows variables such as Ba, Na, and Sr exerting a strong influence on the components, which are indicative of the geochemical composition in the basin.
- **PCA Scatter Plot**: Reveals a distribution of samples that is wide along the first principal component, suggesting significant variability across the geochemical signatures.
- **Scree Plot**: The steep slope of the initial components followed by a leveling off indicates that the dataset's dimensionality can be effectively reduced without substantial loss of information (two components).
### Fort Worth Basin
- **Cumulative Explained Variance**: The plot shows that a significant portion of the total variance is explained by the initial components, with a plateau suggesting additional components add less information.
- **Biplot and Loadings Plot**: Illustrates that geochemical parameters like TDS, Mg, and certain ions are influential in the dataset, contributing strongly to the first two components.
- **PCA Scatter Plot**: Demonstrates variability in geochemical composition, with samples spread primarily along the first principal component.
- **Scree Plot**: Indicates that the majority of the variance is captured by the first few components (two components), with a steep decline in additional explained variance thereafter.
### Great Plains Basin
- **Cumulative Explained Variance**: Similar to other basins, the initial principal components account for the bulk of the variance, suggesting a strong pattern within the data.
- **Biplot and Loadings Plot**: Shows that specific variables, including TDS, Ba, and K, are prominent, reflecting their significant role in the geochemical variance across samples.
- **PCA Scatter Plot**: Reveals a clustering along the first principal component, indicative of a dominant geochemical signature or factor within the basin.
- **Scree Plot**: The explained variance decreases rapidly after the initial components, reinforcing the idea that a small number of components  (two components) can represent the dataset effectively.
### Gulf Coast Basin
- **Cumulative Explained Variance**: The first few principal components capture a significant portion of the variance, with the curve plateauing, indicating that the rest of the components add minimal information.
- **Biplot and Loadings Plot**: Indicated which variables have the strongest influence on the components. Key geochemical parameters such as TDS and specific ions like Br and B are prominently featured.
- **PCA Scatter Plot**: The spread of data points suggests variability in geochemical signatures across the samples, though no clear distinct groupings are observed when considering only the first two components.
- **Scree Plot**: Illustrated a steep drop after the first few components, suggesting that a small number of components (two components) can be used to capture the majority of the data variance.
### Illinois Basin
- **Cumulative Explained Variance**: Demonstrated a similar pattern to the Gulf Coast basin, with early components explaining most variance.
- **Biplot and Loadings Plot**: Highlighted elements such as K, Ca, and Ba as major contributors to variance in the first two principal components.
- **PCA Scatter Plot**: Showed a distribution of data points along the first principal component with less spread along the second, indicating that the first component captures a significant variation aspect.
- **Scree Plot**: Showed that additional components beyond the first few contribute (two components) incrementally less to the explained variance.
### Michigan Basin
- **Cumulative Explained Variance**: The curve showed that a considerable amount of variance is explained by the initial components.
- **Biplot and Loadings Plot**: Zn, Br, and Ca were prominent, suggesting their strong association with the variability in this basin.
- **PCA Scatter Plot**: Displayed a wide dispersion of data points along the first principal component, which may be reflective of the varied geochemical environment of the Michigan basin.
- **Scree Plot**: Reinforced the significance of the first couple of components (two components), after which the explained variance ratio decreases more gradually.
### Oklahoma Platform Basin
- **Cumulative Explained Variance**: The variance explained by the principal components shows a strong start, with a rapid accumulation within the first few components before leveling off. This suggests that most of the geochemical variability can be captured by the initial principal components.
- **Biplot and Loadings Plot**: The biplot reveals that geochemical parameters such as K, SO4, and Mg have a pronounced impact on the variance, with these variables pointing in the direction of the vectors, indicating their strong influence on the components.
- **PCA Scatter Plot**: The scatter of samples predominantly along the first principal component indicates a significant geochemical gradient, with less variability explained by the second component. The dispersion pattern might suggest distinct geochemical processes influencing the composition of the samples.
- **Scree Plot**: The scree plot exhibits a sharp decrease after the first couple of principal components, reinforcing the idea that the most meaningful information is concentrated in the initial components (three components), beyond which the marginal gain in explained variance significantly drops.
### Pacific Basin
- **Cumulative Explained Variance**: A smooth curve indicated that the variance is well explained by the PCA components, with a clear point of diminishing returns.
- **Biplot and Loadings Plot**: Indicates that geochemical parameters like sodium (Na) and total dissolved solids (TDS) are among the variables with the strongest influence on the components. This is evident from their vector lengths and directions, which imply how these variables correlate with the principal components and with each other.
- **PCA Scatter Plot**: Suggested heterogeneity in the geochemical data with the first two components revealing a spread in sample composition.
- **Scree Plot**: The variance explained by subsequent components (three components) drops off, indicating the possibility of a lower-dimensional representation.
### Permian Basin
- **Cumulative Explained Variance**: The first few components accounted for a large portion of the variance, with a plateau indicating fewer components could be used without significant loss of information.
- **Biplot and Loadings Plot**: Illustrated the geochemical variables that are most impactful, with certain elements like Zn and SO4 having significant loadings.
- **PCA Scatter Plot**: The scatter of points mostly along the first principal component suggests that it captures a key variance aspect within the basin's geochemical data.
- **Scree Plot**: Showed a steep initial slope, indicating that the first few principal components (three components) are the most informative.
### Rocky Mountain Basin
- **Cumulative Explained Variance**: The curve suggested that initial components significantly capture the geochemical data variance.
- **Biplot and Loadings Plot**: Showed the dominance of geochemical parameters such as SO4 and Br in the dataset variability.
- **PCA Scatter Plot**: The spread of points across the first component highlighted the heterogeneity of the geochemical signatures.
- **Scree Plot**: Confirmed that the first few components (three components) contain most of the informative variance.
### Williston Basin
- **Cumulative Explained Variance**: Exhibits a steep curve, indicating that initial components are crucial in explaining the variance.
- **Biplot and Loadings Plot**: Highlights the dominance of geochemical indicators like Na and Cl.
- **PCA Scatter Plot**: Shows a concentration of samples along the first component, suggestive of a key variance factor.
- **Scree Plot**: The explained variance by each subsequent component drops markedly after the initial few components (three components).