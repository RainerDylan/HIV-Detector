import pandas as pd

# Load dataset
file_path = "/Users/daisy.c.gunawan/Documents/_coding-stuffs/python_apps/ai_project/Global_HIV_Test_Data.csv"  # Ensure the correct filename
df = pd.read_csv(file_path)

# Ensure numerical data is correctly formatted (remove commas and convert to float)
df["HIV Prevalence (in Adults)"] = pd.to_numeric(df["HIV Prevalence (in Adults)"], errors="coerce")
df["ART Coverage (%)"] = pd.to_numeric(df["ART Coverage (%)"], errors="coerce")
df["Healthcare Expenditure per Capita (USD)"] = pd.to_numeric(df["Healthcare Expenditure per Capita (USD)"], errors="coerce")
df["HIV Incidence Rate (per 1,000 People)"] = pd.to_numeric(df["HIV Incidence Rate (per 1,000 People)"], errors="coerce")
df["Physicians per 1,000 People"] = pd.to_numeric(df["Physicians per 1,000 People"], errors="coerce")

# Calculate global median for Healthcare Expenditure per Capita (USD)
global_median_health_expenditure = df["Healthcare Expenditure per Capita (USD)"].median()

# Function to classify risk level
def classify_risk_level(row):
    # Condition 1: HIV Prevalence Rate exceeding 1%
    high_hiv_prevalence = row["HIV Prevalence (in Adults)"] >= 1  

    # Condition 2: ART Coverage less than 95%
    low_art_coverage = row["ART Coverage (%)"] < 95  

    # Condition 3: Healthcare Expenditure per Capita below global median
    low_healthcare_expenditure = row["Healthcare Expenditure per Capita (USD)"] < global_median_health_expenditure  

    # Condition 4: HIV Incidence Rate greater than 0.5 per 1,000 people
    high_hiv_incidence = row["HIV Incidence Rate (per 1,000 People)"] > 0.5  

    # Condition 5: Physicians per 1,000 people below 1.0
    weak_healthcare_system = row["Physicians per 1,000 People"] < 1.0  

    # If at least 2 out of 5 conditions are met, classify as High Risk
    if sum([high_hiv_prevalence, low_art_coverage, low_healthcare_expenditure, high_hiv_incidence, weak_healthcare_system]) >= 2:
        return "High Risk"
    else:
        return "Low Risk"

# Apply classification function
df["Risk Level"] = df.apply(classify_risk_level, axis=1)

# Save the updated dataset
updated_file_path = "Test_Global_HIV_Risk_Data_Table_With_Risk_Level.csv"
df.to_csv(updated_file_path, index=False)

print("Risk level classification completed. The updated dataset has been saved as:", updated_file_path)
