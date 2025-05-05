# Credit Card Default Prediction Project

## Overview

This project analyzes the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository to predict credit card defaults using various machine learning techniques. The analysis includes data preprocessing, exploratory data analysis, model building, and performance evaluation.

## Files in this Project

1. **Credit_Default_Report.md**
   - The main report document in Markdown format
   - Contains the complete analysis following the academic report template
   - Ready to be converted to PDF after adding figures

2. **Figures_to_Include.md**
   - Lists all figures that should be extracted from the notebook
   - Provides details on where each figure should be placed in the report
   - Includes descriptions of what each figure shows

3. **Executive_Summary.md**
   - Provides a concise summary of the key findings and insights
   - Highlights the most important results and their implications
   - Can be used as a standalone document or incorporated into the main report

4. **Finalization_Instructions.md**
   - Step-by-step guide for finalizing the report
   - Instructions for extracting figures, converting to PDF, and final review
   - Recommendations for ensuring template compliance

5. **default_report.ipynb**
   - Original Jupyter notebook containing the code and analysis
   - Source of all visualizations and results referenced in the report
   - Should be run to generate the figures needed for the report

6. **Final_Report_Template.pdf**
   - The template provided for the report structure
   - Used as a guide for organizing the content of the report

## How to Use These Files

1. Start by reviewing the **Credit_Default_Report.md** file to understand the complete analysis
2. Refer to **Figures_to_Include.md** to identify which visualizations need to be extracted
3. Run the **default_report.ipynb** notebook to generate all necessary figures
4. Follow the steps in **Finalization_Instructions.md** to complete the report
5. Use the **Executive_Summary.md** as needed for a high-level overview

## Dataset Information

The dataset contains information on default payments, demographic factors, credit data, payment history, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Size**: 30,000 observations with 24 features
- **Target Variable**: Default payment next month (1=Yes, 0=No)

## Key Findings

- Random Forest with PCA + K-Means sampling achieved the best overall performance
- Payment history variables were identified as the most important predictors
- Sampling techniques improved the detection of defaulters
- Dimensionality reduction through PCA helped maintain model performance while reducing complexity

## Requirements

To finalize and view the report:
- Jupyter Notebook environment (for running the notebook)
- Markdown editor or converter (for converting the report to PDF)
- Basic understanding of data science and machine learning concepts
