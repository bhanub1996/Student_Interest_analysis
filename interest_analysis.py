import pandas as pd
import re
from pathlib import Path  
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


filepath = Path('C:/Users/bhanu/Downloads/DAProejct_Student_out.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  

# Define the path to your input CSV file
input_file = 'C:/Users/bhanu/Downloads/DAProejct_StudentSurveyRawData_(18march_3May).xlsx - DA_Project_SurveyDataset.csv'

# Read the input CSV file into a pandas DataFrame
df = pd.read_csv(input_file)
df = df.dropna(subset=['attended_classes_for'])


# Function to calculate the average from a pattern like "7 to 8 am"
def calculate_average(pattern):
    # Extract the hour values using regular expression
    hours = re.findall(r'\d+', pattern)
    if len(hours) == 2:
        # Convert the extracted hour values to integers and calculate the average
        average = (int(hours[0]) + int(hours[1])) / 2
        return average
    else:
        return None
def interest_check(x):
    # print(x)
    if 'Learning' in x:
        return 1 
    else: return 0


# Apply the function to the 'wakeup_time' column and assign the result to 'wakeup_time_avg' column
df['wakeup_time_avg'] = df['wakeup_time'].apply(calculate_average)
# print(df['attended_classes_for'][0])
df['interested'] = df['attended_classes_for'].apply(interest_check)
# Print the updated DataFrame
print(len(df))


df = df.groupby('Roll_Number').agg(
    avg_wakeup_time=('wakeup_time_avg', 'mean'),
    attended_classes_with_interest_avg=('interested',  'mean')
)


# Print the result
# print(result)
df.to_csv(filepath)  

bins = [0,2,4,6,8,10, np.inf]
labels = ['Way Too Early', 'Too Early', 'Early', 'Normal', 'Late', 'Very Late']
df['wakeup_time_category'] = pd.cut(df['avg_wakeup_time'], bins=bins, labels=labels)

bins = [0,0.2,0.4,0.6,0.8, np.inf]
labels = ['Not at all interested', 'Less interested','Slightly Interested', 'Interested', 'Very Interested']
df['attended_classes_with_interest_avg'] = pd.cut(df['attended_classes_with_interest_avg'], bins=bins, labels=labels)

# Create a contingency table
contingency_table = pd.crosstab(df['wakeup_time_category'], df['attended_classes_with_interest_avg'])

# Perform chi-square test
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Print the chi-square statistic and p-value
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

# Visualize the contingency table using a heatmap
plt.figure(figsize=(8, 6))
plt.title('Contingency Table Heatmap')
plt.xlabel('Attended Classes with Interest Average')
plt.ylabel('Wake-up Time Category')
plt.xticks(np.arange(contingency_table.shape[1]), contingency_table.columns)
plt.yticks(np.arange(contingency_table.shape[0]), contingency_table.index)
plt.imshow(contingency_table, cmap='Blues', aspect='auto')
plt.colorbar(label='Frequency')
plt.show()
