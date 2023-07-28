import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Placement_Data_Full_Class.csv') #import data
print(df.head())
# Assuming 'status' column has 'Not Placed' and 'Placed'
placement_counts = df['status'].value_counts()

#--placement histogram
plt.bar(placement_counts.index, placement_counts.values)
# Add text annotations with precise numbers above each bar
for i, count in enumerate(placement_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.xlabel('Placement')
plt.ylabel('Number of Students')
plt.title('Placement Outcome')
plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.show()

#--specialisation piechart
placed_students = df[df['status'] == 'Placed']
specialisation_counts = placed_students['specialisation'].value_counts() 
# Pie Chart
plt.pie(specialisation_counts.values, labels=specialisation_counts.index, autopct='%1.1f%%')
plt.title('Specialisation Distribution for Successfully Placed Students')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

#--workexpiechart
workex_students = df[df['workex'] == 'Yes']
# Calculate the number of students with work experience who got placed
placed_with_workex = workex_students[workex_students['status'] == 'Placed'].shape[0]
# Calculate the number of students with work experience who didn't get placed
not_placed_with_workex = workex_students[workex_students['status'] == 'Not Placed'].shape[0]
# Data to plot
labels = ['Placed with Work Experience', 'Not Placed with Work Experience']
sizes = [placed_with_workex, not_placed_with_workex]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # Explode the first slice
# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
# Add a title
plt.title("Placement Status based on Work Experience")
# Show the pie chart
plt.show()

#--degreet_bar
# Count the number of students with different degree_t values who got placed
degree_t_counts = placed_students['degree_t'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
degree_t_counts.plot(kind='bar', color='skyblue')

# Add labels and title
plt.xlabel('Degree Type')
plt.ylabel('Number of Students Placed')
plt.title('Number of Students with Different Degree Types Who Got Placed')

# Show the bar chart
plt.tight_layout()
plt.show()
