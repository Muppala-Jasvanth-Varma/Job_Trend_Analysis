import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(r"W:\FOML\ml programs\.venv\codebasics.py\Data_Analyst\Data_Analyst.csv")

df['Salary Estimate'] = df['Salary Estimate'].str.replace(r"\(.*\)", "", regex=True).str.strip()
df = df[df['Salary Estimate'].str.contains('-')]

def parse_salary(salary):
    try:
        low, high = salary.split('-')
        low = int(low.replace('K', '').replace('$', '').strip())
        high = int(high.replace('K', '').replace('$', '').strip())
        return (low + high) / 2
    except ValueError:
        return None

df['Salary Estimate'] = df['Salary Estimate'].apply(parse_salary)
df.dropna(subset=['Salary Estimate'], inplace=True)

df.fillna({'Size': 'Unknown', 'Founded': -1, 'Industry': 'Unknown', 'Sector': 'Unknown'}, inplace=True)

label_enc = LabelEncoder()
categorical_columns = ['Location', 'Size', 'Industry', 'Sector', 'Company Name']
for col in categorical_columns:
    df[col] = label_enc.fit_transform(df[col])

sector_analysis = df.groupby('Sector').agg(
    avg_salary=('Salary Estimate', 'mean'),
    job_count=('Job Title', 'count')
).sort_values(by=['avg_salary', 'job_count'], ascending=False)

print("Top Sectors by Salary and Job Count:")
print(sector_analysis.head())

X = df[['Location', 'Size', 'Founded', 'Industry', 'Sector', 'Company Name']]
y = df['Salary Estimate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
print("Feature Importance:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))
import seaborn as sns

if df['Industry'].dtype != 'object':
    # Create reverse mapping from encoded values to original labels
    industry_mapping = {index: label for index, label in enumerate(label_enc.classes_)}
    df['Industry'] = df['Industry'].map(industry_mapping)

sector_analysis = df.groupby('Industry').agg(
    avg_salary=('Salary Estimate', 'mean'),
    job_count=('Job Title', 'count')
).sort_values(by='avg_salary', ascending=False)

top_sectors = sector_analysis.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_sectors.reset_index(),
    x='Industry',
    y='avg_salary',
    palette='coolwarm'
)
plt.title("Top 10 Industries by Average Salary", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("Average Salary (in K)", fontsize=12)
plt.xlabel("Industry", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_sectors.reset_index(),
    x='Industry',
    y='job_count',
    palette='viridis'
)
plt.title("Top 10 Industries by Job Count", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("Job Count", fontsize=12)
plt.xlabel("Industry", fontsize=12)
plt.tight_layout()
plt.show()
