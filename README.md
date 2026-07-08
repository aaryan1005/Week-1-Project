Project 1: Advanced Exploratory Data Analysis (EDA) & Feature Engineering
Goal: Transform raw, chaotic data into a mathematically clean, optimized dataset ready for ingestion by advanced machine learning algorithms.

📖 Project Overview
Real-world data is rarely ready for modeling out of the box. It is often plagued by missing values, extreme outliers, and non-predictive formats. This project focuses on the critical pre-processing pipeline: systematically cleaning the data using statistical methods and engineering new, high-value features that give machine learning models a stronger mathematical signal to learn from.

Key Objectives:

Perform deep exploratory data analysis to understand feature distributions and correlations.

Implement robust statistical imputation strategies to resolve missing data without introducing bias.

Detect and neutralize mathematical anomalies (outliers).

Synthesize new predictive features from existing columns to improve downstream model accuracy.

🧠 Methodology & Data Cleaning
1. Statistical Imputation (Handling Missing Data)
Dropping rows with missing values results in data loss. Instead, missing data was handled programmatically based on the distribution of the feature:

Normal Distributions: Imputed using the Mean.

Skewed Distributions: Imputed using the Median to resist the pull of extreme values.

Complex/Categorical Relationships: Imputed using K-Nearest Neighbors (KNN), predicting the missing value based on the similarity of surrounding data points.

2. Outlier Detection & Neutralization
Outliers can severely distort the decision boundaries of machine learning models. They were systematically identified and handled using two distinct mathematical approaches:

Z-Score Method: Applied to normally distributed features. Any data point with a Z-Score >3 or <−3 (representing 3 standard deviations from the mean) was flagged and capped.

Interquartile Range (IQR): Applied to skewed features. Calculated the bounds using Q1−1.5×IQR and Q3+1.5×IQR to cap extreme anomalous values safely.

⚙️ Feature Engineering
To provide the eventual machine learning models with a stronger predictive signal, 3 distinct new features were engineered from the raw columns:

Engineered Feature	Source Columns	Mathematical Transformation & Business Logic
Customer_Tenure_Months	Join_Date, Last_Purchase_Date	Calculated the time delta between initial sign-up and most recent activity to establish a continuous variable for customer loyalty.
Avg_Spend_Per_Item	Total_Revenue, Items_Purchased	Divided total revenue by the number of items. This normalizes spending behavior, separating bulk budget buyers from premium shoppers.
Activity_Frequency_Score	Logins_Past_30_Days, Email_Opens	Created a weighted composite index scaling from 1-10 to represent overall user engagement, reducing dimensionality while retaining variance.
(Note: If your dataset uses different features, simply swap the names in this table to match your specific variables).

💻 Technical Stack
Language: Python

Data Manipulation & Cleaning: pandas, numpy

Statistical Analysis: scipy.stats

Machine Learning Tools: scikit-learn (KNNImputer)

Visualizations: matplotlib, seaborn (for distribution and correlation heatmaps)

🚀 How to Run the Project
1. Clone the repository:

Bash
git clone https://github.com/yourusername/advanced-eda-feature-engineering.git
cd advanced-eda-feature-engineering
2. Install required dependencies:

Bash
pip install -r requirements.txt
3. Run the cleaning pipeline:

Bash
python clean_and_engineer.py
Note: Running this script will process the raw CSV, print the statistical summaries (before and after imputation) to the console, save the EDA distribution graphs in the /visualizations folder, and output the final, ML-ready dataset as cleaned_data_final.csv.
