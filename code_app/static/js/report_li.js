// function to insert content for the "detailed report" tab

// define the function
function insertHTML() {

    let newHTML = `
        
    <h1>Lithium Concentration Prediction Using Machine Learning Techniques</h1>

    <h2 style="color:#996600">Table of Contents</h2>
    
    <ul>
      <li><a href="#1-introduction">1. Introduction</a></li>
      <li><a href="#2-data-preparation">2. Data Preparation</a></li>
      <li><a href="#3-imputation-strategy">3. Imputation Strategy</a></li>
      <li><a href="#4-heatmap-analysis">4. Heatmap Analysis</a></li>
      <li><a href="#5-principal-component-analysis-pca">5. Principal Component Analysis (PCA)</a></li>
      <li><a href="#6-predictive-modeling">6. Predictive Modeling</a></li>
      <li><a href="#7-conclusion">7. Conclusion</a></li>
      <li><a href="#8-technologies">8. Technologies</a></li>
      <li><a href="#9-data-sources">9. Data Sources</a></li>
      <li><a href="#10-contributors">10. Contributors</a></li>
    </ul>
    
    <h2 style="color:#996600">1. Introduction</h2>
    
    <p>In this report, we present an analysis of geochemical data of produced water samples, obtained from the <a href="https://www.sciencebase.gov/catalog/item/64fa1e71d34ed30c2054ea11" target="_blank">U.S. Geological Survey</a>. Our objective is to apply statistical and machine learning techniques to predict Lithium (Li) concentrations in produced water from oil and gas production. Data preprocessing encompasses Principal Component Analysis (PCA), iterative imputation techniques to handle null values, and spearman correlation heatmaps. The machine learning models utilized in this project include Gradient Boosting (GB), Random Forest (RF), Neural Networks (NN), K-Nearest Neighbor (KNN), and Support Vector Regression (SVR).</p>
    
    <h2 style="color:#996600">2. Data Preparation</h2>
    
    <p>The raw dataset was subjected to several preprocessing steps to ensure its readiness for analysis:</p>
    
    <ul>
      <li><strong>General Cleaning</strong>: Irrelevant columns, such as identifiers and categorical data, were removed. This step was crucial to focus the analysis on quantitative geochemical features.</li>
      <li><strong>Well Type Filtering</strong>: To ensure our study focused on the high-salinity produced water typical in oil and gas operations, we excluded well samples from coal, injection, and geothermal plays. This approach keeps our analysis relevant to the typical environmental and production challenges in the oil and gas industry.</li>
      <li><strong>Basin Categorization</strong>: Well samples were categorized based on the basin they belong to. This categorization helps to group samples by geographical and geological features.</li>
      <li><strong>Technical Cleaning</strong>: In the dataset preprocessing phase, observations with Total Dissolved Solids (TDS) levels below 10,000 parts per million (ppm) were excluded. This threshold was established based on the consideration that waters with TDS levels below this mark are typically characterized as brackish and are often subjected to treatment processes for various uses, including agricultural and industrial applications. Conversely, waters with TDS exceeding 10,000 ppm are generally associated with greater environmental challenges, requiring more rigorous management strategies to mitigate potential adverse ecological impacts. By focusing on samples with higher TDS levels, the analysis targets the subset of produced water that is more likely to raise environmental and disposal concerns, thus providing a more relevant framework for examining geochemical data in the context of environmental science and hydrology.</li>
    </ul>
    
    <p>See the below table with the statistical info for the data after general and technical cleaning.</p>
    
    <img src="${img_data_table}" alt="Cleaned Data Info">
    <br><br>

    <h2 style="color:#996600">3. Amazon RDS</h2>
    <p>The platform selected for our database management is Amazon Relational Database Service (Amazon RDS), which was determined to be optimally suited for handling our extensive datasets. Following the successful establishment of the database, we employed the Psycopg2 and SQLAlchemy libraries within Python to facilitate data loading procedures. The initial step involved the configuration of a connection between pgAdmin and Amazon RDS. Subsequent to this, we proceeded to create distinct tables corresponding to each dataset.The data, originally stored in CSV files, was imported into Pandas DataFrames. Finally, we systematically uploaded the data from each DataFrame into the appropriate tables in Amazon RDS.This structured data was subsequently retrieved from the database for use in various machine learning models and for the creation of our dashboard.</p>

    <p>For a visual representation of this process, please refer to the flowchart below.</p>

    <img src="${img_flowchart}" alt="Flowchart">
    <br><br>
    
    <h2 style="color:#996600">3. Imputation Strategy</h2>
    
    <p>For geochemical datasets, where the variables often exhibit complex interdependencies, simplistic imputation methods such as substitution by mean, median, or extremities like minimum and maximum values may not adequately capture the intrinsic variability and can potentially introduce biases. To address the missing data in our dataset, we utilized an Iterative Imputer, employing a RandomForestRegressor as the estimator. This advanced imputation technique considers the entire variable distribution and the stochastic nature of the dataset, thereby preserving the inherent multivariate relationships among geochemical parameters. It is especially critical in PCA, which requires a complete dataset as missing values can significantly distort the principal components derived from the analysis. PCA is discussed further below.</p>
    
    <h2 style="color:#996600">4. Heatmap Analysis</h2>
    
    <p>Through heatmap analysis, we have examined the correlations between Lithium (Li) concentrations and various geochemical parameters across multiple basins. The following section elucidates the distinct geochemical signatures observed in each basin and discusses the implications of these findings for predicting Lithium concentrations.</p>
    
    <h3 style="color:gray">Basin-Specific Correlations with Lithium</h3>
    
    <ul>
      <li><strong>Appalachian Basin</strong>: Strontium (Sr) and Barium (Ba) emerged as the most correlated features with Lithium, with correlation coefficients of 0.557 and 0.529 respectively.</li>
      <li><strong>Permian Basin</strong>: Zinc (Zn) displayed a remarkably high correlation with Lithium at 0.774, followed by Sr and Sodium (Na).</li>
      <li><strong>Oklahoma Platform Basin</strong>: Here, Sr was again prominent with a correlation of 0.529, alongside Potassium (K) and Calcium (Ca), with correlations of 0.425 and 0.365.</li>
      <li><strong>Gulf Coast Basin</strong>: Boron (B) and Bromine (Br) showed strong correlations with Lithium at 0.634 and 0.632, respectively, while Ca also demonstrated a notable correlation of 0.546.</li>
      <li><strong>Williston Basin</strong>: Exhibited extremely high correlations with Barium and Boron at around 0.98 for both, indicating a significant geochemical interplay with Lithium.</li>
      <li><strong>Michigan Basin</strong>: Zinc displayed a perfect correlation with Lithium, followed by Br and Ca with strong correlations of 0.845 and 0.755.</li>
      <li><strong>Pacific Basin</strong>: Featured Bicarbonate (HCO3) as the most correlated parameter with Lithium at 0.428, suggesting geochemical processes where HCO3 may be a determinant in Lithium concentration.</li>
      <li><strong>Illinois Basin</strong>: Highlighted K as the leading correlating feature with a high coefficient of 0.845, with Ca and Ba also showing strong relationships with Lithium.</li>
      <li><strong>Great Plains Basin</strong>: Indicated that Ba, Br, and Sr are significant contributors to Lithium variability, with correlation coefficients ranging from 0.559 to 0.585.</li>
      <li><strong>Anadarko Basin</strong>: Sulfate (SO4) was the most correlated with Lithium at 0.592, along with notable correlations with K and Ba.</li>
      <li><strong>Rocky Mountain Basin</strong>: Presented Na, TDS, and Chloride (Cl) as the most correlated with Lithium, though the coefficients were modest, all hovering around 0.3.</li>
      <li><strong>Fort Worth Basin</strong>: Iron Total (FeTot) had a perfect correlation with Lithium, while Ca and Ba also showed very strong correlations, with coefficients above 0.86.</li>
    </ul>
    
    <h3 style="color:gray">Implications for Understanding the Dataset</h3>
    
    <p>The disparities in geochemical correlations with Lithium across basins underscore the complexity of geochemical interactions within distinct geological settings. In some basins, certain elements exhibit extremely strong, if not perfect, correlations with Lithium, suggesting a potential for direct predictive modeling of Lithium concentrations based on these elements.</p>
    
    <p>In other basins, such as the Rocky Mountain, the correlations are less pronounced, which may necessitate a more nuanced approach to modeling that incorporates a broader array of features or perhaps the development of basin-specific prediction models.</p>
    
    <h2 style="color:#996600">5. Principal Component Analysis (PCA)</h2>

    <p>Principal Component Analysis (PCA) was applied to the geochemical data from each basin separately. The findings from this dimensionality reduction technique provided insights into the underlying data structure and the interrelations among geochemical variables, as summarized below for each basin:</p>

    <h3 style="color:gray">Anadarko Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: Indicates a substantial amount of the variance is captured by the first 10 principal components, showing that they are significant in representing the dataset.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Highlights the geochemical variables like SO4, K, and Mg that contribute prominently to the variance within the basin.</li>
    <li><strong>PCA Scatter Plot</strong>: The spread of data points predominantly along the first principal component suggests that it captures a significant variation within the basin.</li>
    <li><strong>Scree Plot</strong>: Demonstrates a quick decline in the explained variance ratio, indicating that 90% of the information is concentrated in the first few principal components.</li>
    </ul>

    <h3 style="color:gray">Appalachian Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The curve suggests that first 10 principal components are sufficient to explain more than 90% of the variance in the data.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Shows variables such as Ba, Na, and Sr exerting a strong influence on the components, which are indicative of the geochemical composition in the basin.</li>
    <li><strong>PCA Scatter Plot</strong>: Reveals a distribution of samples that is wide along the first principal component, suggesting significant variability across the geochemical signatures.</li>
    <li><strong>Scree Plot</strong>: The steep slope of the initial components followed by a leveling off indicates that the dataset's dimensionality can be effectively reduced without substantial loss of information.</li>
    </ul>

    <h3 style="color:gray">Fort Worth Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The plot shows that a significant portion of the total variance (more than 90%) is explained by the first 10 components, with a plateau suggesting additional components add less information.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Illustrates that geochemical parameters like TDS, Mg, and certain ions are influential in the dataset, contributing strongly to the first two components.</li>
    <li><strong>PCA Scatter Plot</strong>: Demonstrates variability in geochemical composition, with samples spread primarily along the first principal component.</li>
    <li><strong>Scree Plot</strong>: Indicates that the majority of the variance is captured by the first few components, with a steep decline in additional explained variance thereafter.</li>
    </ul>

    <h3 style="color:gray">Great Plains Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: Similar to other basins, the first 10 principal components account for the bulk of the variance, suggesting a strong pattern within the data.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Shows that specific variables, including TDS, Ba, and K, are prominent, reflecting their significant role in the geochemical variance across samples.</li>
    <li><strong>PCA Scatter Plot</strong>: Reveals a clustering along the first principal component, indicative of a dominant geochemical signature or factor within the basin.</li>
    <li><strong>Scree Plot</strong>: The explained variance decreases rapidly after the initial components, reinforcing the idea that the first few components can represent the dataset effectively.</li>
    </ul>

    <h3 style="color:gray">Gulf Coast Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The first 10 principal components capture more than 90% of the variance, with the curve plateauing, indicating that the rest of the components add minimal information.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Indicated which variables have the strongest influence on the components. Key geochemical parameters such as TDS and specific ions like Br and B are prominently featured.</li>
    <li><strong>PCA Scatter Plot</strong>: The spread of data points suggests variability in geochemical signatures across the samples, though no clear distinct groupings are observed when considering only the first two components.</li>
    <li><strong>Scree Plot</strong>: Illustrated a steep drop after the first few components, suggesting that a small number of components can be used to capture the majority of the data variance.</li>
    </ul>

    <h3 style="color:gray">Illinois Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: Demonstrated a similar pattern to the Gulf Coast basin, with early 10 components explaining most variance.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Highlighted elements such as K, Ca, and Ba as major contributors to variance in the first two principal components.</li>
    <li><strong>PCA Scatter Plot</strong>: Showed a distribution of data points along the first principal component with less spread along the second, indicating that the first component captures a significant variation aspect.</li>
    <li><strong>Scree Plot</strong>: Showed that additional components beyond the first few contribute incrementally less to the explained variance.</li>
    </ul>

    <h3 style="color:gray">Michigan Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The curve showed that a considerable amount of variance is explained by the initial components.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Zn, Br, and Ca were prominent, suggesting their strong association with the variability in this basin.</li>
    <li><strong>PCA Scatter Plot</strong>: Displayed a wide dispersion of data points along the first principal component, which may be reflective of the varied geochemical environment of the Michigan basin.</li>
    <li><strong>Scree Plot</strong>: Reinforced the significance of the first couple of components, after which the explained variance ratio decreases more gradually.</li>
    </ul>

    <h3 style="color:gray">Oklahoma Platform Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The variance explained by the principal components shows a strong start, with a rapid accumulation within the first few components before leveling off. This suggests that most of the geochemical variability can be captured by the initial principal components.</li>
    <li><strong>Biplot and Loadings Plot</strong>: The biplot reveals that geochemical parameters such as K, SO4, and Mg have a pronounced impact on the variance, with these variables pointing in the direction of the vectors, indicating their strong influence on the components.</li>
    <li><strong>PCA Scatter Plot</strong>: The scatter of samples predominantly along the first principal component indicates a significant geochemical gradient, with less variability explained by the second component. The dispersion pattern might suggest distinct geochemical processes influencing the composition of the samples.</li>
    <li><strong>Scree Plot</strong>: The scree plot exhibits a sharp decrease after the first couple of principal components, reinforcing the idea that the most meaningful information is concentrated in the initial components, beyond which the marginal gain in explained variance significantly drops.</li>
    </ul>

    <h3 style="color:gray">Pacific Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: A smooth curve indicated that the variance is well explained by the PCA components, with a clear point of diminishing returns.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Indicates that geochemical parameters like sodium (Na) and total dissolved solids (TDS) are among the variables with the strongest influence on the components. This is evident from their vector lengths and directions, which imply how these variables correlate with the principal components and with each other.</li>
    <li><strong>PCA Scatter Plot</strong>: Suggested heterogeneity in the geochemical data with the first two components revealing a spread in sample composition.</li>
    <li><strong>Scree Plot</strong>: The variance explained by subsequent components drops off, indicating the possibility of a lower-dimensional representation.</li>
    </ul>

    <h3 style="color:gray">Permian Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The first few components accounted for a large portion of the variance, with a plateau indicating fewer components could be used without significant loss of information.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Illustrated the geochemical variables that are most impactful, with certain elements like Zn and SO4 having significant loadings.</li>
    <li><strong>PCA Scatter Plot</strong>: The scatter of points mostly along the first principal component suggests that it captures a key variance aspect within the basin's geochemical data.</li>
    <li><strong>Scree Plot</strong>: Showed a steep initial slope, indicating that the first few principal components are the most informative.</li>
    </ul>

    <h3 style="color:gray">Rocky Mountain Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: The curve suggested that initial components significantly capture the geochemical data variance.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Showed the dominance of geochemical parameters such as SO4 and Br in the dataset variability.</li>
    <li><strong>PCA Scatter Plot</strong>: The spread of points across the first component highlighted the heterogeneity of the geochemical signatures.</li>
    <li><strong>Scree Plot</strong>: Confirmed that the first few components contain most of the informative variance.</li>
    </ul>

    <h3 style="color:gray">Williston Basin</h3>

    <ul>
    <li><strong>Cumulative Explained Variance</strong>: Exhibits a steep curve, indicating that initial components are crucial in explaining the variance.</li>
    <li><strong>Biplot and Loadings Plot</strong>: Highlights the dominance of geochemical indicators like Na and Cl.</li>
    <li><strong>PCA Scatter Plot</strong>: Shows a concentration of samples along the first component, suggestive of a key variance factor.</li>
    <li><strong>Scree Plot</strong>: The explained variance by each subsequent component drops markedly after the initial few components.</li>
    </ul>

    <br>

    <h2 style="color:#996600">6. Predictive Modeling</h2>

    <h3 style="color:gray">Input Data Scenarios</h3>

    <p>We employed the machine learning models across two scenarios:</p>

    <ul>
    <li><strong>Scenario 1 (Imputed Dataset)</strong>: Utilized an enhanced dataset with imputed missing values.</li>
    <li><strong>Scenario 2 (PCA-Transformed Dataset)</strong>: Applied PCA to reduce dimensionality, focusing on the first 10 principal components capturing over 90% of data variance.</li>
    </ul>

    <h3 style="color:gray">Gradient Boosting for Lithium Concentration Prediction</h3>

    <p>Employing Gradient Boosting Regression, we derived predictions for Li concentrations, relying on performance metrics for model validation and leveraging visual plots for a comparative assessment of actual vs. predicted values. This approach was chosen due to its robustness in handling nonlinear relationships and its ability to handle the variance explained by the principal components effectively.</p>

    <h4><ins>Methodology</ins></h4>

    <p>Gradient Boosting Regressor from scikit-learn, configured with 100 estimators, a 0.1 learning rate, and a maximum depth of 3, was employed to balance complexity with performance. Both the imputed dataset and the PCA-transformed dataset were split into training and testing sets, allocating 20% for evaluation.</p>

    <h4><ins>Model Training and Evaluation</ins></h4>

    <p>For each basin, the model was trained on the known Lithium concentrations and evaluated using the Mean Squared Error (MSE), R-squared (R2), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Explained Variance. These metrics provide a comprehensive view of the model's accuracy and its ability to generalize to new data.</p>

    <ul>
    <li><strong>MSE and RMSE</strong>: offer measures of the average error magnitude, with RMSE giving more weight to larger errors.</li>
    <li><strong>MAE</strong>: provides a straightforward average of error magnitudes.</li>
    <li><strong>R2</strong>: indicates the proportion of variance in the dependent variable predictable from the independent variables.</li>
    <li><strong>Explained Variance</strong>: measures how well our model accounts for the variation in the dataset.</li>
    </ul>

    <h4><ins>Predictive Performance</ins></h4>

    <p>The following plots display the performance metrics of the machine learning models in each basin for both scenarios:</p>

    <img src="${img_performance_gb}" alt="Performance Metrics">
    <br><br>

    <p>Generally, models trained on the full set of imputed data without PCA perform better than their PCA-transformed counterparts. This trend suggests that the reduction of dimensionality through PCA might lead to the loss of critical information necessary for making accurate predictions in certain basins.
    Basins with complex geochemical backgrounds, such as the Great Plains and Gulf Coast, show varied responses to PCA, with significant performance degradation noted when critical variables are potentially omitted during dimensionality reduction.
    Basins like the Williston and Gulf Coast (without PCA) illustrate that a comprehensive approach, utilizing all available geochemical parameters, can substantially enhance model accuracy, especially in regions with complex interactions.

    The summarized performance of Gradient Boosting models indicates that while PCA can be useful for simplifying models and reducing computational burdens, its applicability should be carefully considered. In cases where geochemical data is complex and interactions are nuanced, retaining the full spectrum of data without PCA often yields better predictive outcomes.</p>

    <h3 style="color:gray">K-Nearest Neighbors (KNN) for Lithium Concentration Prediction</h3>

    <p>Using the KNN algorithm, we conducted an analysis to predict Lithium concentrations across various basins. This was selected due to its ability to handle non-linear relationships.</p>

    <h4><ins>Methodology</ins></h4>

    <p>KNN analysis was conducted using the scikit-learn library, considering both PCA-transformed and non-transformed datasets. The algorithm was configured with a range of k values to determine the optimal number of neighbors. The feature lists for each dataset included relevant geochemical parameters.</p>

    <h4><ins>Model Training and Evaluation</ins></h4>

    <p>For each basin, the KNN model was trained on known Lithium concentrations and evaluated using multiple metrics including Mean Squared Error (MSE), R-squared (R2), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Explained Variance. Additionally, the percentage of predictions within certain thresholds of the actual values was calculated to assess model accuracy.</p>

    <ul>
    <li><strong>MSE and RMSE</strong>: offer measures of the average error magnitude, with RMSE giving more weight to larger errors.</li>
    <li><strong>MAE</strong>: provides a straightforward average of error magnitudes.</li>
    <li><strong>R2</strong>: indicates the proportion of variance in the dependent variable predictable from the independent variables.</li>
    <li><strong>Explained Variance</strong>: measures how well our model accounts for the variation in the dataset.</li>
    <li><strong>Percentage Thresholds</strong>: indicate the percentage of the predicted values that fall within a certain ± threshold of the actual values.</li>
    </ul>

    <h4><ins>Predictive Performance</ins></h4>

    <p>The following plots display the performance metrics of the machine learning models in each basin for both scenarios:</p>

    <img src="${img_performance_knn}" alt="Performance Metrics">
    <br><br>

    <p>Summary of results:</p>
    <ul>
    <li>The KNN analysis was conducted using both PCA-transformed and non-PCA data inputs across various basins.</li>
    <li>For the PCA-transformed data, the best performing K value ranged from 1 to 3.</li>
    <li>The explained variance ranged from 10.36% to 99.55% for PCA-transformed data and from 10.36% to 99.55% for non-PCA data.</li>
    <li>The mean squared error (MSE) varied significantly across basins and data input types, with values ranging from 3.01 to 40,328.95.</li>
    <li>Percentage of predicted values within ±5%, ±15%, ±25%, and ±50% of actual values varied across basins, with differences observed between PCA and non-PCA data.</li>
    <li>Cross-validation MSE ranged from 4.10 to 8659.87 for PCA-transformed data and from 74.79 to 3864.63 for non-PCA data, indicating varying levels of model generalization performance.</li>
    <li>Performance metrics such as MSE, explained variance, and percentage of predicted values within certain ranges varied depending on the specific basin and the use of PCA in preprocessing the data.</li>
    </ul>

    <h3 style="color:gray">Support Vector Regression (SVR) for Lithium Concentration Prediction</h3>

    <p>Using the SVR algorithm, we conducted an analysis to predict Lithium concentrations across various basins. This was selected due to its ability to handle non-linear relationships.</p>

    <h4><ins>Methodology</ins></h4>

    <p>SVR analysis was conducted using the scikit-learn library, considering both PCA-transformed and non-transformed datasets. The algorithm was configured with a range of kernel options, C values, and epsilon values to determine the optimal hyperparameters for each model. The feature lists for each dataset included relevant geochemical parameters.</p>

    <h4><ins>Model Training and Evaluation</ins></h4>

    <p>For each basin, the SVR model was trained on known Lithium concentrations and evaluated using multiple metrics including Mean Squared Error (MSE), R-squared (R2), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Explained Variance. Additionally, the percentage of predictions within certain thresholds of the actual values was calculated to assess model accuracy.</p>

    <ul>
    <li><strong>MSE and RMSE</strong>: offer measures of the average error magnitude, with RMSE giving more weight to larger errors.</li>
    <li><strong>MAE</strong>: provides a straightforward average of error magnitudes.</li>
    <li><strong>R2</strong>: indicates the proportion of variance in the dependent variable predictable from the independent variables.</li>
    <li><strong>Explained Variance</strong>: measures how well our model accounts for the variation in the dataset.</li>
    <li><strong>Percentage Thresholds</strong>: indicate the percentage of the predicted values that fall within a certain ± threshold of the actual values.</li>
    </ul>

    <h4><ins>Predictive Performance</ins></h4>

    <p>The following plots display the performance metrics of the machine learning models in each basin for both scenarios:</p>

    <img src="${img_performance_svr}" alt="Performance Metrics">
    <br><br>

    <p>Summary of results:</p>
    <ul>
    <li>The SVR analysis was conducted using both PCA-transformed and non-PCA data inputs across multiple basins.</li>
    <li>PCA-transformed data inputs utilized principal components as features, while non-PCA data inputs consisted of specific chemical and geological features.</li>
    <li>Various kernel options including RBF, polynomial, and sigmoid were tested for SVR, with different basins favoring different kernels as the best performing.</li>
    <li>The hyperparameter C, representing the regularization parameter, varied across basins, with different optimal values selected for each.</li>
    <li>Performance metrics such as the percentage of predicted values within certain ranges of actual values, quantile losses, and cross-validation mean squared error (MSE) were evaluated.</li>
    <li>Results indicated varying levels of prediction accuracy and generalization performance across basins and data input types, highlighting the importance of selecting appropriate preprocessing techniques and hyperparameters for SVR regression.</li>
    </ul>

    <h3 style="color:gray">Random Forest for Lithium Concentration Prediction</h3>

    <h4><ins>Methodology</ins></h4>

    <p>The Random Forest models were implemented using the scikit-learn library, optimized for geochemical data analysis to predict lithium concentrations. Each model configuration was adapted to handle different data scenarios effectively:</p>
    <ul>
    <li><strong>Random Forest Setup</strong>: The model utilized a Random Forest Regressor, an ensemble learning method known for its robustness and accuracy in regression tasks. The forest comprised multiple decision trees, whose parameters were fine-tuned using hyperparameter optimization.</li>
    <li><strong>Hyperparameter Tuning</strong>: A grid search was conducted to find the optimal model settings, utilizing a parameter grid that included:</li>
    <li><strong>Number of Estimators</strong>: Options of 100, 200, 300 trees to explore the effect of forest size on performance.</li>
    <li><strong>Maximum Depth</strong>: Tree depths of None (fully grown trees), 10, and 20 to control overfitting.</li>
    <li><strong>Minimum Samples Split</strong>: Minimum number of samples required to split a node set at 2, 5, and 10, impacting the model's sensitivity to data variance.</li>
    <li><strong>Minimum Samples Leaf</strong>: Minimum number of samples required at a leaf node, tested with 1, 2, and 4, which helps in stabilizing the predictions across diverse data points.</li>
    </ul>

    <h4><ins>Predictive Performance</ins></h4>

    <p>Models that surpass the R^2 threshold of 0.75 demonstrate excellent predictive capabilities, suggesting that the models have captured the essential geochemical interactions required for accurate predictions.</p>

    <p>The following plots display the performance metrics of the machine learning models in each basin for both scenarios:</p>

    <img src="${img_performance_rf}" alt="Performance Metrics">
    <br><br>

    <p>The analysis suggests that Random Forest models without PCA generally provide better predictive performance when they have access to the complete geochemical dataset, as seen in the Gulf Coast and Williston Basins. This indicates that the richness of the full dataset provides a more accurate basis for predicting lithium concentrations, particularly in complex geochemical environments where interactions between numerous variables are critical. While PCA offers benefits in reducing dimensionality and computational demands, its appropriateness varies by the specific characteristics of each basin's geochemical profile. The effectiveness of models without PCA in certain basins advocates for a cautious approach to dimensionality reduction, suggesting that retaining more comprehensive data may sometimes be advantageous for maintaining predictive accuracy in Random Forest models.</p>

    <h3 style="color:gray">Neural Network for Lithium Concentration Prediction</h3>

    <h4><ins>Methodology</ins></h4>

    <p>The Neural Network models were constructed using Keras, with a configuration optimized for geochemical data analysis. Each model comprised an input layer, two hidden layers, and an output layer. The first hidden layer had 128 neurons, and the second had 64 neurons, both utilizing the ReLU activation function for non-linearity. The output layer consisted of a single neuron for predicting lithium concentrations. The Adam optimizer was employed with a learning rate of 0.001, and the loss function used was mean squared error to minimize the prediction error.</p>

    <h4><ins>Model Performance Across Geological Basins</ins></h4>

    <p>The performance analysis of Neural Network models on PCA-transformed and imputed data without PCA reveals distinct patterns that underscore the influence of data preprocessing techniques on predictive accuracy in geological settings. Focusing on the R^2 value, which assesses the proportion of variance in lithium concentrations that the model can predict, we establish a threshold of 0.75 to evaluate effective predictive performance. Here’s a detailed examination of how each scenario fares across various basins:</p>

    <h4><ins>Predictive Performance</ins></h4>

    <p>Models that exceed the R^2 threshold of 0.75 demonstrate a robust ability to predict lithium concentrations, suggesting that the models have successfully captured the key geochemical interactions required for accurate predictions:</p>

    <p>The following plots display the performance metrics of the machine learning models in each basin for both scenarios:</p>

    <img src="${img_performance_nn}" alt="Performance Metrics">
    <br><br>

    <p>The analysis underscores the importance of appropriate data preprocessing and feature extraction techniques in modeling complex geological datasets. PCA transformation generally enhances model performance by highlighting essential features and omitting redundant or irrelevant data. However, the variability in model success across different basins suggests that a one-size-fits-all approach may not be appropriate. Tailoring data preprocessing and modeling techniques to the specific characteristics of each basin could yield better predictive accuracy.</p>

    <p>In conclusion, while PCA offers substantial benefits in some geological contexts by simplifying the input data and potentially enhancing model accuracy, its application should be carefully considered against the backdrop of each basin's unique geochemical profile to optimize lithium concentration predictions effectively.</p>

    <h3 style="color:gray">Multi-Target Sequential Chaining</h3>

    <p>The Multi-Target Sequential Chaining (MTSC) approach leverages a sequential prediction strategy, where the prediction of one target variable aids in the modeling of the next. This technique was applied specifically to the Gulf Coast Basin data, focusing on a series of geochemical elements as targets, which are crucial in predicting lithium concentrations. The methodology harnesses the power of Gradient Boosting alongside Random Forest, Neural Network (MLP regressor), Support Vector Regression (SVR), and Extreme Gradient Boostin (XGB)  within a framework designed to optimize sequential predictions.</p>

    <h4><ins>Methodology Overview</ins></h4>

    <ul>
    <li><strong>Data Preparation</strong>: The dataset was initially processed to prioritize target variables based on the count of missing values. The approach was to predict the least problematic variables first and use their predictions as features for subsequent models.</li>
    <li><strong>Feature Set-Up</strong>: Starting with a set of key geochemical indicators (e.g., TDS, Cl), the model incrementally included newly predicted targets into the feature set, enhancing the predictor space as it moved through the target list.</li>
    <li><strong>Model Execution</strong>: For each target, a variety of models were evaluated through a rigorous GridSearchCV process to identify the optimal configuration. Models included Gradient Boosting, Random Forest, MLPRegressor, SVR, and XGBRegressor, each configured with a specific set of hyperparameters suitable for regression tasks.</li>
    </ul>

    <h4><ins>Performance Comparison</ins></h4>

    <p>The performance of Gradient Boosting was directly compared with other models based on standard regression metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE),R^2 score, and Explained Variance. Each model was tuned to minimize MSE and optimize the R^2 score, aiming to exceed an R^2 threshold of 0.75 to be deemed effective.</p>

    <ul>
    <li><strong>Best Model Performance to Predict Features</strong>: Gradient Boosting showed substantial efficacy in the sequential prediction model, particularly evident in its ability to handle complex nonlinear relationships within the geochemical data. However, while Gradient Boosting often provided strong predictive accuracy, it sometimes fell short in comparison to simpler or more flexible models, particularly in data scenarios characterized by high variability or less straightforward relationships.</li>
    <li><strong>Best Model Performance to Predict Li</strong>: Among the various models tested, the SVR and XGB emerged as the most effective, particularly for Lithium prediction, achieving the highest R^2 score of 0.798 and 0.792 espectively, underscoring their capability in handling the specific nuances of the geochemical dataset of the Gulf Coast Basin.</li>
    </ul>

    <h4><ins>Algorithmic Insights and Recommendations</ins></h4>

    <ul>
    <li><strong>Sequential Benefit</strong>: The sequential approach allowed for nuanced understanding and leveraging of interdependencies among geochemical variables. This strategy is particularly beneficial in complex environmental datasets where the interaction between variables can significantly influence the outcome.</li>
    <li><strong>Algorithm Selection</strong>: Although Gradient Boosting demonstrated solid performance, the standout results achieved by SVR and XGBRegressor, with R^2 scores of 0.798 and 0.792 respectively, highlight the efficacy of a diversified modeling approach. The strong performances of these models suggest that an ensemble strategy, leveraging the distinct strengths of various algorithms, could be advantageous for enhancing the robustness and accuracy of predictions in complex geochemical environments. This approach would capitalize on the complementary capabilities of different models to address various facets of the prediction task. The summary of parameters and performance metrics of the MTSC method is listed in the following table.</li>
    </ul>

    <img src="${img_mtsc_table}" alt="Table">
    <br><br>

    <p>The geoplot shown below displays both the known and the predicted lithium concentrations according to each model used in the MTSC method. This visualization enables users to visually evaluate the size of the circles, which correspond to the lithium concentrations, and to compare the spatial distribution of predicted versus actual concentrations. The example geomap below depicts only the predicted values from the gradient boosting model (green circles) and the actual lithium values (red circles).</p>

    <img src="${img_mtsc_geomap}" alt="GeoMap">
    <br><br>

    <h2 style="color:#996600">7. Conclusion</h2>

    <p>This study on geochemical data analysis and Lithium concentration prediction has leveraged a multifaceted machine learning approach to uncover and exploit the subtle nuances within produced water samples from various geological basins across the United States. By employing advanced statistical and machine learning techniques, including Principal Component Analysis (PCA) and sophisticated imputation methods, we have systematically enhanced our understanding and predictive capabilities concerning Lithium concentrations, a critical component in contemporary energy production and battery storage.</p>

    <p>The deployment of various models such as Gradient Boosting, Random Forest, Neural Networks, SVR, and KNN provided a robust framework to tackle the prediction tasks across different scenarios—one with imputed missing values and another using PCA-transformed data. These models were carefully tuned to address the unique characteristics of each basin, with an emphasis on capturing the intricate relationships within the geochemical parameters.</p>

    <p>Model results from the scenario 1 generally show better predictive performance when they can access the complete geochemical dataset, as observed in the Gulf Coast and Williston Basins. This suggests that the detailed information in the full dataset is crucial for accurate predictions, especially in complex environments. Although PCA can reduce computational demands by simplifying data, its effectiveness varies across different geochemical profiles. The findings support an approach to reducing data dimensionality, highlighting the potential benefits of retaining more comprehensive data for enhancing predictive accuracy in all models. For basins with underperforming in all models (Appalachian, Fort Worth, Great Plains, Illinois, Oklahoma Platform, Pacific, and Permian), the underperformance may be due to limited data quantity and a narrow range of low lithium concentrations.</p>

    <p>The use of multi-target sequential chaining in the Gulf Coast Basin exemplifies an innovative approach to modeling. This method not only streamlined the predictive process by utilizing outputs from one predictive model as inputs for another but also highlighted the effectiveness of integrating multiple machine learning strategies to enhance prediction accuracy. The MTSC approach, particularly when implemented with a robust algorithm like Extreme Gradient Boosting (XGB) and Support Vector Regression (SVR), offers a promising method for predicting lithium concentrations in produced water.</p>

    <p>The project has not only demonstrated the feasibility of using advanced machine learning techniques to predict Lithium concentrations in produced water but also highlighted the potential for these methodologies to revolutionize how industries related to energy production and environmental management operate. By turning intricate geochemical data into actionable insights, this research paves the way for more informed decision-making and strategic resource management in the energy sector.</p>

    <h2 style="color:#996600">8. Technologies</h2>

    <ul>
    <li>Languages 
        <ul>
            <li><a href="https://www.python.org/" target="_blank">Python 3.10 or higher</a></li>
            <li><a href="https://html.spec.whatwg.org/multipage/" target="_blank">HTML</a></li>
            <li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference" target="_blank">Javascript</a></li>
            <li><a href="https://developer.mozilla.org/en-US/docs/Web/CSS" target="_blank">CSS</a></li>
            <li><a href="https://www.postgresql.org/docs/current/sql.html" target="_blank">SQL(via PostgreSQL)</a></li>
        </ul>
    </li>
    <li>Libraries / Modules / Plugins
        <ul>
            <li><a href="https://matplotlib.org/" target="_blank">Matplotlib</a></li>
            <li><a href="https://www.scipy.org/scipylib" target="_blank">SciPy</a></li>
            <li><a href="https://scikit-learn.org/stable/index.html" target="_blank">Scikit-learn</a></li>
            <li><a href="https://github.com/jyangfsu/WQChartPy/tree/main?tab=readme-ov-file" target="_blank">WQChartPy</a></li> 
            <li><a href="https://seaborn.pydata.org/#" target="_blank">Seaborn</a></li>
            <li><a href="https://geopandas.org/en/stable/#" target="_blank">GeoPandas</a></li>
            <li><a href="https://pypi.org/project/folium/" target="_blank">Folium</a></li>
            <li><a href="https://pypi.org/project/branca/" target="_blank">Branca</a></li>
            <li><a href="https://leafletjs.com/" target="_blank">Leaflet</a></li>
            <li><a href="https://getbootstrap.com/" target="_blank">Bootstrap 4.5.2</a></li>
            <li><a href="https://jquery.com/" target="_blank">jQuery 3.5.1</a></li>
            <li><a href="https://popper.js.org/" target="_blank">Popper.js 1.16.0</a></li>
            <li><a href="https://d3js.org/" target="_blank">D3.js v7</a></li>
            <li><a href="https://www.papaparse.com/" target="_blank">PapaParse 5.3.0</a></li>
            <li><a href="https://pandas.pydata.org/" target="_blank">Pandas</a></li>
            <li><a href="https://www.numpy.org" target="_blank">NumPy</a></li>
            <li><a href="https://www.psycopg.org/docs/" target="_blank">Psycopg2</a></li>
            <li><a href="https://www.sqlalchemy.org/" target="_blank">SQLAlchemy</a></li>
            <li><a href="https://pypi.org/project/pyproj/" target="_blank">Pyproj 3.6.1</a></li>
            <li><a href="https://flask.palletsprojects.com/en/3.0.x/" target="_blank">Flask</a></li>
            <li><a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a></li>
        </ul>
    </li>
    <li>Other Tools
        <ul>
            <li><a href="https://www.postgresql.org/docs/" target="_blank">PostgreSQL</a></li>
            <li><a href="https://aws.amazon.com/rds/" target="_blank">Amazon Web Services RDS</a></li>
        </ul>
    </li>
    </ul>

    <h2 style="color:#996600">9. Data Sources</h2>

    <ul>
    <li>CMG Model Output</li>
    <li><a href="https://www.beg.utexas.edu/texnet-cisr/texnet" target="_blank">TexNet Seismic Data</a></li>
    <li><a href="https://injection.texnet.beg.utexas.edu/api/Export" target="_blank">Injection Data API</a></li>
    <li><a href="https://www.usgs.gov/" target="_blank">USGS Produced Water Data</a></li>
    </ul>

    <h2 style="color:#996600">10. Contributors</h2>

    <ul>
    <li><a href="https://github.com/roxanadrv" target="_blank">Roxana Darvari</a></li>
    <li><a href="https://github.com/bsvab" target="_blank">Brittany Svab</a></li>
    <li><a href="https://github.com/ajuarez2112" target="_blank">Alejandro Juarez</a></li>
    <li><a href="https://github.com/thesarahcain" target="_blank">Sarah Cain</a></li>
    </ul>

    `;

    // Insert the new HTML content into the page
    document.getElementById("report_li").innerHTML = newHTML;
}

// call the function
insertHTML();