import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import numpy as np
import plotly.express as px


# Load the dataset
file_path1 = "/Users/vascofelgueiras/Desktop/Universidade/AnaliseDados/Mental health Depression disorder Data Original.csv"
file_path2 = "/Users/vascofelgueiras/Desktop/Universidade/AnaliseDados/dummy_data.csv"
file_path3 = "/Users/vascofelgueiras/Desktop/Universidade/AnaliseDados/2019.csv"


#Passar para numérico onde tem string
df_mental_health = pd.read_csv(file_path1, low_memory=False ,dtype={
        'Schizophrenia (%)': str,
        'Eating disorders (%)': str
    })

# Converter coluna 5 para float
df_mental_health['Schizophrenia (%)'] = pd.to_numeric(df_mental_health['Schizophrenia (%)'], errors='coerce')
# Converter coluna 6 para float
df_mental_health['Eating disorders (%)'] = pd.to_numeric(df_mental_health['Eating disorders (%)'], errors='coerce')

df_time_on_social_media = pd.read_csv(file_path2)
df_world_happiness = pd.read_csv(file_path3)

# Print datasets individually
print(df_mental_health.head())
print(df_time_on_social_media.head())
print(df_world_happiness.head())

print(df_mental_health.info())
print(df_time_on_social_media.info())
print(df_world_happiness.info())

print(df_mental_health.describe())
print(df_time_on_social_media.describe())
print(df_world_happiness.describe())

# Merge datasets
df_mental_health = df_mental_health.rename(columns={"Entity": "Country"})
df_time_on_social_media = df_time_on_social_media.rename(columns={"location":  "Country"})
df_world_happiness = df_world_happiness.rename(columns={"Country or region":  "Country"})

merged_df = pd.merge(df_mental_health, df_time_on_social_media, on="Country", how='outer')
merged_df2 = pd.merge(merged_df, df_world_happiness, on="Country", how='outer')

# Print merge dataset
print(merged_df2.columns)

print(merged_df2.head())

print(merged_df2.info())

print(merged_df2.describe())



# Handling Missing Data
print("\nMissing Values per Column:")
print(merged_df2.isnull().sum())

print("\nMissing Values per Column(%):")
missing_percentage = (merged_df2.isnull().sum() / len(merged_df2)) * 100
print(missing_percentage)

# Fill with mean
for col in ["Schizophrenia (%)","Bipolar disorder (%)","Eating disorders (%)","Anxiety disorders (%)"]: merged_df2[col].fillna(merged_df2[col].mean(), inplace=True)

# Fill with median
for col in ["Depression (%)","Alcohol use disorders (%)"]: merged_df2[col].fillna(merged_df2[col].median(), inplace=True)

# Fill with mode
for col in ["platform","interests"]: merged_df2[col].fillna(merged_df2[col].mode()[0], inplace=True) #mode()[0] returns the most frequent value the 0 is there in case there are multiple modes

print("\nMissing Values After Handling:")
print(merged_df2.isnull().sum())

print("\nMissing Values per Column(%) After Handling:")
missing_percentage = (merged_df2.isnull().sum() / len(merged_df2)) * 100
print(missing_percentage)



#Boxplot Before Handling Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_df2[["Schizophrenia (%)","Eating disorders (%)","Anxiety disorders (%)"]])
plt.title("Boxplot Before Outlier Handling")
plt.show()

#Outliers specifically
Q1 = merged_df2["Eating disorders (%)"].quantile(0.25)
Q3 = merged_df2["Eating disorders (%)"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = merged_df2[(merged_df2["Eating disorders (%)"] < lower_bound) | (merged_df2["Eating disorders (%)"] > upper_bound)]
print(outliers)

#Handling Outliers
#Outliers to low or upper bound
for col in ["Schizophrenia (%)","Eating disorders (%)","Anxiety disorders (%)"]:
    Q1 = merged_df2[col].quantile(0.25)
    Q3 = merged_df2[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    merged_df2[col] = merged_df2[col].clip(lower=lower_bound, upper=upper_bound)

print("\nDataset After Handling Outliers:")
print(merged_df2.describe())

#Boxplot After Handling Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_df2[["Schizophrenia (%)","Eating disorders (%)","Anxiety disorders (%)"]])
plt.title("Boxplot After Outlier Handling")
plt.show()


# Absolute Frequency for Gender
gender_freq = merged_df2["gender"].value_counts()
print("\nAbsolute Frequency - Gender:")
print(gender_freq)

# Relative Frequency (%) for Gender
gender_relative = merged_df2["gender"].value_counts(normalize=True) * 100
print("\nRelative Frequency (%) - Gender:")
print(gender_relative)

# Cumulative Frequency - Satisfaction
time_spent_cumfreq = merged_df2["time_spent"].value_counts().sort_index().cumsum()
print("\nCumulative Frequency - Time spent on Social Media:")
print(time_spent_cumfreq)


# Bar chart for Age frequency
plt.figure(figsize=(20,20))
merged_df2["age"].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Age Frequency Distribution")
plt.xlabel("age")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()

# Pie chart for Gender
plt.figure(figsize=(5,5))
merged_df2["gender"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Gender Distribution")
plt.ylabel('')
plt.show()

## Prevalence of Anxiety Disorders by country (1990-2017)
fig = px.choropleth(
    merged_df2,
    locations='Code',
    color='Anxiety disorders (%)',
    hover_name='Country',
    animation_frame='Year',
    color_continuous_scale='Plasma',
    title='Prevalence of Anxiety Disorders by country (1990-2017)'
)
fig.show()

## Time spent in Social Media by Age
plt.figure(figsize=(10, 6))
sns.kdeplot(
    x=merged_df2["age"],
    y=merged_df2["time_spent"],
    cmap="Blues",
    fill=True,
    thresh=0.1
)
plt.title("Time spent by Age")
plt.xlabel("Age")
plt.ylabel("Time in Social Media")
plt.show()

#Depression Through The Years
df_year = merged_df2.groupby('Year')['Depression (%)'].mean()
plt.figure(figsize=(10, 6))
df_year.plot(kind='line', marker='o', color='skyblue')
plt.title('Depression Through The Years (%)')
plt.xlabel('Year')
plt.ylabel('Depression (%)')
plt.grid(True)
plt.show()

#Platform Distribution Among Users
platform_counts = merged_df2['platform'].value_counts().dropna()
platform_counts.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Social Media Platforms')
plt.ylabel('')
plt.show()

# Top 10 Countries by Happiness Score
if 'Score' in merged_df2.columns and 'Country' in merged_df2.columns:
    avg_score_country = merged_df2.groupby('Country')['Score'].mean().dropna().sort_values(ascending=False).head(10)
    avg_score_country.plot(kind='bar', color='goldenrod')
    plt.title('Top 10 Countries by Happiness Score')
    plt.ylabel('Happiness Score')
    plt.xlabel('Country')
    plt.show()

# Average Time Spent on Social Media by Profession
if 'profession' in merged_df2.columns and 'time_spent' in merged_df2.columns:
    avg_time_profession = merged_df2.groupby('profession')['time_spent'].mean().sort_values(ascending=False)
    avg_time_profession.plot(kind='bar', figsize=(12,6), color='slateblue')
    plt.title('Average Time Spent on Social Media by Profession')
    plt.xlabel('Profession')
    plt.ylabel('Average Time Spent')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Car Ownership by Profession
if 'profession' in merged_df2.columns and 'Owns_Car' in merged_df2.columns:
    plt.figure(figsize=(12,6))
    sns.countplot(x='profession', hue='Owns_Car', data=merged_df2)
    plt.title('Car Ownership by Profession')
    plt.xlabel('Profession')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Average Time Spent by Platform and Gender
if 'platform' in merged_df2.columns and 'gender' in merged_df2.columns and 'time_spent' in merged_df2.columns:
    pivot = merged_df2.pivot_table(index='platform', columns='gender', values='time_spent', aggfunc='mean')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Average Time Spent by Platform and Gender')
    plt.ylabel('Platform')
    plt.xlabel('Gender')
    plt.show()


# Average Income vs Happiness Index by Country
avg_income_score = (merged_df2.groupby('Country').agg({'income': 'mean','Score': 'mean'}).reset_index())

plt.figure(figsize=(10, 6))
sns.scatterplot(
        data=avg_income_score,
        x='income',
        y='Score',
    )
for _, row in avg_income_score.iterrows():
        plt.text(
            #posiciona o texto perto
            row['income'] + avg_income_score['income'].std()*0.01,
            row['Score'] + avg_income_score['Score'].std()*0.01,
            row['Country'],
            fontsize=8
        )

plt.title('Average Income vs Happiness Index by Country')
plt.xlabel('Average Income by Country')
plt.ylabel('Happiness Index (Score)')
plt.tight_layout()
plt.show()

#Anual Evolution: Time spent vs Anxiety disorders (%)
avg_time_depression = (merged_df2.groupby('Year').agg({'time_spent': 'mean', 'Anxiety disorders (%)': 'mean'}).reset_index().sort_values('Year'))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    avg_time_depression['Year'],
    avg_time_depression['time_spent'],
    marker='o',
    linestyle='-',
    label='Time spent mean',
    color='skyblue'
)
ax1.set_xlabel('Year')
ax1.set_ylabel('Time spent',color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

ax2 = ax1.twinx()
ax2.plot(
    avg_time_depression['Year'],
    avg_time_depression['Anxiety disorders (%)'],
    marker='s',
    linestyle='--',
    label='Anxiety disorders (%) Mean',
    color='red'
)
ax2.set_ylabel('Anxiety disorders (%) Mean', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Anual Evolution: Time spent vs Anxiety disorders (%)')
plt.tight_layout()
plt.show()