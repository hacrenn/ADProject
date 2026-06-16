# Data Analysis Toolkit (Python)

**Quick description**
This is a data analysis project that investigates the relationship between mental health indicators (e.g., depression, anxiety), time spent on social media, and measures of well-being (e.g., happiness index). The repository contains an interactive notebook, a script that automates the processing chain, CSV datasets, and a report with methodology and conclusions. Repository: GitHub — Author/maintainer: hacrenn — Main technologies used: Python Software Foundation, Project Jupyter

---

## Project objective

The objective is to demonstrate a complete exploratory analysis and visualization workflow to answer questions such as:
* Is there a correlation between average daily time spent on social media and depression rates by country/age?
* How have anxiety and depression indicators evolved over time (temporality by country)?
* Which demographic groups (age/gender) have the greatest exposure to identified risk factors?
The focus is exploratory—identifying patterns, generating enlightening visualizations, and proposing hypotheses for more robust statistical analyses.

---

## Repository contents (file summary)

* ProjectAD.ipynb — Jupyter notebook with the complete workflow: data reading, cleaning, merging, exploration, and interactive visualizations.
* ProjectAD.py — Python script that reproduces the notebook pipeline in a non-interactive way (useful for automatic execution and generation of figures/outputs).
* ReportAD.pdf — Written document with methodology, results, interpretations, and recommendations.
* data/ (or CSV files in the root) — Set of CSV files used (prevalence of mental illnesses, time spent on social networks, socioeconomic indicators, happiness index, etc.).
* docs/ (optional) — Notes, exported graphs, and auxiliary files used for the report.
Note: some files in the repository may contain absolute paths; it is recommended to use relative paths and organize all CSVs in a data/ folder before running the notebook or script.

---

## Description of the data and sources

The project integrates multiple sources to build a harmonized database by country/year:
* Prevalence data for mental disorders — percentages by country/year for depression, anxiety, etc.
* Social media usage data — metrics of average daily/weekly time spent by age group/platform (when available).
* Well-being indicators — happiness index, GDP per capita, and other contextual socioeconomic factors.
* Demographic data — distribution by age and gender for stratification.
Whenever possible, the original sources should be documented (source, year, link). The databases used and the transformations applied are referenced in the AD Report.pdf.

---

## Methodological flow (high level)

1. Ingestion — load CSVs, inspect columns and data types.
2. Cleaning — harmonize column names, convert types (numeric/date), and handle missing values ​​and outliers.
3. Harmonization — rename columns and map variables from different sources to a common schema (e.g., Entity → Country).
4. Merge/Join — combine datasets by keys (usually Country + Year), choosing join strategies that preserve relevant observations.
5. Transformation — create derived variables (normalizations, percentiles, age range categorizations).
6. Exploration — descriptive statistics, correlations, pivot tables, and visualizations (choropleth maps, time series, stratified scatter plots).
7. Interpretation — summarize observed patterns, limitations, and hypotheses for further investigation.

The treatment decisions (e.g., imputation of missing data, removal of outliers, choice of temporal aggregation) are described in the report.
