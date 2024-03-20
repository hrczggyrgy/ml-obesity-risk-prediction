
# Exploratory Data Analysis for Obesity Risk

## Project Overview

This project aims to explore and analyze obesity levels based on various factors such as eating habits, physical condition, and lifestyle choices. The dataset involves individuals from countries like Mexico, Peru, and Colombia. The exploratory data analysis (EDA) seeks to uncover underlying patterns, correlations, and insights that may contribute to obesity.

### Data Description

The data consists of measurements and survey responses from participants of Mexico, Peru, and Colombia, focusing on their diet, physical activity, and other lifestyle factors. Key variables include gender, age, height, weight, family history of overweight, food consumption habits, physical activity frequency, technology use, transportation methods, and obesity levels (target variable).

### Objectives

- To understand the distribution of various factors contributing to obesity.
- To investigate the relationship between obesity levels and variables such as age, gender, physical activity, and eating habits.
- To identify significant predictors of obesity levels.
- To discover potential data-driven insights on mitigating obesity risk.

### Analysis Steps

1. Data loading and initial exploration: Load the dataset and get a preliminary understanding of its structure and key statistics.
2. Data cleaning and preprocessing: Handle missing values, outliers, and anomalies. Perform necessary transformations.
3. Explorative data analysis: Use statistics and visualizations to explore the data in depth.
4. Feature engineering: Create new features that might be helpful for modeling or provide additional insights.
5. Correlation and significance testing: Assess the relationship between variables and their impact on obesity levels.
6. Outlier and anomaly detection: Identify and analyze outliers or abnormal data points.
7. Conclusions and insights: Draw conclusions from the analysis and suggest actionable insights.

### Conclusion

This project provides a comprehensive analysis of factors contributing to obesity within the dataset. Key findings and insights can guide further research and interventions to manage and prevent obesity.

### Requirements

The required Python libraries for this project include pandas, numpy, matplotlib, seaborn, scipy, and sklearn. A `requirements.txt` file is available in the repository.

### Usage

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run Jupyter Notebook to explore the analysis: `jupyter notebook`.

### Acknowledgements

Dataset originally provided by a study involving participants from Mexico, Peru, and Colombia. Generated and transformed for educational purposes.
    
Training Information:

Data:
The model has been trained on a dataset with 18295 rows and 35 features.

The features include: Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS, log_BMI, age_group, physical_activity_score, caloric_intake_tendency, healthy_eating_score, BMR, meal_regularity_score, snacking_habit, stress_eating_indicator, sedentary_lifestyle_score, overall_lifestyle_score, log_Age, log_Height, log_Weight, sqrt_FCVC, sqrt_NCP, sqrt_CH2O, Height_Weight_interaction, Age_FCVC_interaction.

Model:
A RandomForestClassifier model has been used for training.

The model parameters are: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}.

Optimization:
The model has been optimized using optuna optimization with 100 trials.
The best trial score is: 0.9071330964744465



## Model Training
The model training process consisted of preprocessing on both numerical and categorical data.
After preprocessing, a RandomForestClassifier model was fit on the preprocessed data.
For categorical features preprocessing, the most frequent values were imputed for missing values and then one hot encoded.
For numerical features preprocessing, the mean was usually used for imputation follwed by StandardScaler for standardization.
The target (NObeyesdad) was label encoded before training.

For hyperparameter tuning, the methods used were the XGBoost classifier and Optuna for hyperparameter optimization.
This process was logged using the wandb library for visualization.
The best set of hyperparameters was selected based on the accuracy of the model.

Additional feature engineering was performed, including creating interactions between different features and normalizing some, to improve the model's performance.
