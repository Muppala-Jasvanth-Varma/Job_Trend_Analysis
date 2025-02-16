# Data Analyst Salary Analysis

## Overview
This project analyzes salary estimates for data analyst roles using machine learning techniques. It includes data preprocessing, exploratory data analysis, feature encoding, and model training with Random Forest Regression to predict salaries. Additionally, insights are extracted on salary trends across various industries and sectors.

## Dataset
The dataset is stored in `Data_Analyst.csv` and contains job listings with salary estimates and company details.

## Dependencies
Ensure you have the following Python libraries installed:

pip install pandas matplotlib seaborn scikit-learn

## Data Preprocessing
- Removed text within parentheses from the `Salary Estimate` column.
- Filtered out entries without salary ranges.
- Converted salary estimates into numerical values by averaging the range.
- Handled missing values by filling them with appropriate default values.
- Encoded categorical variables using `LabelEncoder`.

## Exploratory Data Analysis
- Analyzed salary trends by sector and industry.
- Identified top-paying sectors and industries.
- Visualized salary distributions using bar charts.

## Model Training
- Features used: `Location`, `Size`, `Founded`, `Industry`, `Sector`, `Company Name`.
- Target variable: `Salary Estimate`.
- Splitting dataset into training and testing sets (80%-20%).
- Training a `RandomForestRegressor` model with 100 estimators.
- Evaluating the model using Mean Absolute Error (MAE).

## Key Findings
- Displays top sectors and industries based on salary and job count.
- Identifies important features influencing salary predictions.
- Highlights salary trends across different industries.

## Results
- The trained model provides salary predictions with a calculated MAE.
- Industry-wise and sector-wise salary distribution visualizations.

## Visualizations
Two key bar plots:
1. **Top 10 Industries by Average Salary**
2. **Top 10 Industries by Job Count**

## Usage
Run the script using:
```
python salary_analysis.py
```

## Future Improvements
- Enhance feature engineering with additional factors like job title and experience level.
- Experiment with other regression models.
- Deploy as a web app for interactive salary predictions.

