# Understanding Trust and Confidence in Law Enforcement: A Data Science Approach

## JBG050 Data Challenge 2 || Group 11

### Project Context
Law enforcement agencies, including the Metropolitan Police Service (MPS) in London, are increasingly focusing on building trust and confidence among communities. This project employs data science methods to understand the factors influencing trust and confidence in the MPS. By analyzing data from the Public Attitude Survey (PAS) and other crime statistics, the project aims to provide insights and actionable recommendations for key decision-makers and operational officers within an ~8-week timeframe.

### Objectives
Our project seeks to address the following subquestions:
- What are the causal factors driving variations in trust and confidence in the MPS across all boroughs in London?
- How do changes in crime metrics predict trust and confidence? How does this differ between different regions in London?

### The Repository
This repository contains the code used for analysis based on which our recommendations were defined. We utilize two primary datasets: the Public Attitude Survey and a comprehensive crime dataset sourced from [data.police.uk](https://data.police.uk/). 

#### Steps to Get the Code Running
1. **Clone the Repository**: Use Git to clone the repository to your local machine.
2. **Import Requirements**: Ensure all necessary packages and libraries are installed.
3. **Import Relevant Data**:
   - Download the PAS Excel file `PAS_T&Cdashboard_to Q3 23-24.xlsx` from [The London Datastore - MOPAC Surveys](https://data.london.gov.uk/dataset/mopac-surveys) and place it in a folder titled "data" within the repository.
   - For crime data, visit [DATA.POLICE.UK - Archive](https://data.police.uk/data/archive/) and download the following datasets (or any combination that covers all months from 2016-2023):
     - December 2018
     - December 2021
     - April 2024
   - Extract the contents from the downloaded zip files, place them in the local Downloads directory.
   - Run the Jupyter notebooks numbered 01 to 04 to preprocess the data files necessary for the subsequent analysis files. This will generate additional needed files in the "data" folder within the repository.
     - **Note**: Update the variable field with the local downloads directory path in file 01.
4. **Run Main Analysis Files**:
   - Running the analysis files numbered 05 and 06 will produce the results used to base the recommendations on.

#### Files Overview
- **Jupyter Notebooks**:
  - `00 PAS_eda.ipynb` (optional to run)
  - `01 retrieving_data.ipynb`
  - `02 neib_prior.ipynb`
  - `03 data_analysis.ipynb`
  - `04 street_data_exploration.ipynb`
- **R File**:
  - `05 causal_analysis.r`
- **Python Files**:
  - `06 prediction_analysis.py`
  - `07 Stationarity_test.py` (optional to run)
  - `08 dash_map.py` (optional to run)
- **Archive Folder**:
  - Contains old code files left for reference.
