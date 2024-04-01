#@title Load the dataset (Global_YouTube_Statistics_2023_pruned.csv)
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Mount Google Drive - the file is too large; we opted for this method of uploading the file to Colab.
drive.mount('/content/drive')

#@title Read the file from Google Drive
file_path = '/content/drive/MyDrive/Colab Notebooks/Global_YouTube_Statistics_2023_pruned.csv' #Set file path as needed. The file will be selected from Google Drive.
youtube_df = pd.read_csv(file_path,header=1, encoding='ISO-8859-1')
print(youtube_df)

#@title Inspecting the Dataframe

# Remove all channels where 'video views' < 1 and readjust rankings'
youtube_df = youtube_df[youtube_df['video views'] > 1 ]
youtube_df.reset_index(drop=True, inplace=True)

youtube_df = youtube_df.copy()

youtube_df['rank'] = youtube_df['subscribers'].rank(ascending=False, method='min').astype(int)

# Print the shape of the dataframe
print("Shape of the dataframe:", youtube_df.shape)

# Displaying the DataFrame
display(youtube_df.head())

# Inspecting the DataFrame
youtube_df.info()

#@title Investigate the data

uploads = youtube_df['uploads'].dropna()

count = uploads.sum()
min_uploads = uploads.min()
max_uploads = uploads.max()
upload_range = max_uploads - min_uploads
mean_upload = uploads.mean()
median_upload = uploads.median()
mode_uploads = uploads.mode().values[0]
variance_uploads = uploads.var()
std_dev_uploads = uploads.std()

uploads_by_channel = youtube_df.groupby('Youtuber')['uploads'].sum().dropna()

top_uploaders = uploads_by_channel.sort_values(ascending=False)

subs_by_channel = youtube_df.groupby('Youtuber')['subscribers'].sum().dropna()

top_subs = subs_by_channel.sort_values(ascending=False)

views_by_channel = youtube_df.groupby('Youtuber')['video views'].sum().dropna()

top_views = views_by_channel.sort_values(ascending=False)

# Print the Descriptive Statistics
print(f"1. Count of uploads: {count}")
print(f"2. Minimum uploads: {min_uploads:.2f}")
print(f"3. Maximum uploads: {max_uploads:.2f}")
print(f"4. Upload range: {upload_range:.2f}")
print(f"5. Mean Upload: {mean_upload:.2f}")
print(f"6. Median Upload: {median_upload:.2f}")
print(f"7. Mode of Uploads: {mode_uploads:.2f}")
print(f"8. Variance of Uploads: {variance_uploads:.2f}")
print(f"9. Standard Deviation of Uploads: {std_dev_uploads:.2f}")

print("\nChannels with the Most Uploads: ")
print(top_uploaders)

print("\nChannels with the Most Subscribers: ")
print(top_subs)

print("\nChannels with the Most Views: ")
print(top_views)

# Visualizing Raw Data
print(youtube_df.head().transpose())

#@title Pie Chart Representation of Youtube Channels by Category

# Group the data by the 'category' column and count the number of channels in each category
category_counts = youtube_df['category'].value_counts()

# Create a custom color palette with unique colors for each category
num_categories = len(category_counts)
color_palette = plt.cm.tab20c(range(num_categories))  # You can choose a different colormap if needed

# Create a pie chart without labels
plt.figure(figsize=(8, 8))
plt.pie(category_counts, autopct='%1.1f%%', startangle=140, colors=color_palette)
plt.title('Distribution of YouTube Channels by Category (2023)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add a legend
plt.legend(category_counts.index, title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))

# Display the pie chart
plt.show()

#@title Pie Chart Representation of Youtube channels by country of origin
# Group the data by the 'country' column and count the number of channels in each country
country_counts = youtube_df['Country'].value_counts()

# Calculate the total number of channels
total_channels = len(youtube_df)

# Filter countries with less than 1% of the total channels
limit = 0.01 * total_channels
filtered_country_counts = country_counts[country_counts >= limit]
other_count = country_counts[country_counts < limit].sum()

# Add the "Less than 1%" category
filtered_country_counts['Less than 1%'] = other_count

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(filtered_country_counts, labels=filtered_country_counts.index, autopct='%1.1f%%', startangle=140, rotatelabels=True)
plt.title('Distribution of YouTube Channels by Country (2023)', y=1.15)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add a legend
plt.legend(filtered_country_counts.index, title='Country', loc='center left', bbox_to_anchor=(1.2, 0.5))

# Display the pie chart
plt.show()

#@title Cleaning the data
from sklearn.model_selection import train_test_split

# Select only numeric columns
numeric_columns = youtube_df.select_dtypes(include=['number'])

# Filter the DataFrame to keep only rows with uploads <= 6,000
youtube_df = youtube_df[youtube_df['uploads'] <= 6000]

#Filter the DataFrame to keep channels with video views <= 75,000,000,000 - Optional
youtube_df = youtube_df[youtube_df['video views'] <= 75000000000]

# Filter the DataFrame to keep only rows with 'country' = 'United States' to donwsize the data pool
youtube_df = youtube_df[youtube_df['Country'] == "United States" ]

# Remove rows with 'NaN' values in any of the numeric columns
youtube_df = youtube_df.dropna(subset=numeric_columns.columns)

# Drop non-numeric columns from the DataFrame
youtube_df = youtube_df[numeric_columns.columns]

# Reset the index and 'rank' attribute of the DataFrame
youtube_df = youtube_df.reset_index(drop=True)
youtube_df['rank'] = range(1, len(youtube_df) + 1)

# Displaying the DataFrame
display(youtube_df.head())

# Inspecting the DataFrame
youtube_df.info()

#@title Display Correlation Data
# Choose a specific variable for correlation
chosen_variable = 'uploads'

# Filter the correlations for the chosen variable
correlations = youtube_df.corr()["uploads"].drop("uploads")

# Create Bar plot using Seaborn
sns.set(style='white')
plt.figure(figsize=(24, 10))
ax = sns.barplot(x=correlations.index, y=correlations.values)
plt.title(f'Correlation between {chosen_variable} and Other Input Variables')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.show()

#@title Visualizing the correlation as a Heat Map
corr = youtube_df.corr()

plt.subplots(figsize=(24,10))
sns.heatmap(corr, cmap='RdYlGn', annot=True)
plt.title('Correlation Heat Map')
plt.show()

#@title Creating the Linear Regression Model
model = LogisticRegression(solver='liblinear', max_iter=1000)

#@title Preparing the data (Train vs. Test Split)
#Split the data into features and target
target = youtube_df['video views']
features = youtube_df[['uploads']]

#Split the dataset into features and target
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=7)

# Print the sizes of the training and testing sets
print('Training Sets: Count of Feature instances: ', len(feature_train))
print('Training Sets: Count of Target instances: ', len(target_train))

print('Testing Sets: Count of Feature instances: ', len(feature_test))
print('Testing Sets: Count of Target instances: ', len(target_test))

# Print the first few rows of the features and target dataframes
print('\nFeatures:')
print(features.head())

print('\nTarget:')
print(target.head())

#@title Training the LRM
# Train the model on the training data
model.fit(feature_train, target_train.values.ravel())

#@title Predict the targets for the given feature_test
target_predicted = model.predict(feature_test)

#@title Linear Progression Model (Uploads vs. Video Views)
# Extract the 'subscribers' and 'uploads' columns
y_train = youtube_df['video views'].values.reshape(-1, 1)
x_train = youtube_df['uploads'].values

# Create and fit a Linear Regression model
model = LinearRegression()
model.fit(x_train.reshape(-1, 1), y_train)

# Get the slope (m) and intercept (c) of the regression line
m = model.coef_[0][0]
c = model.intercept_[0]

# Predict subscribers using the model
predicted_subscribers = model.predict(x_train.reshape(-1, 1))

# Visualize the data and regression line
plt.scatter( x_train,y_train, marker='D', color='blue', label='Actual Values')
plt.plot(x_train, predicted_subscribers, color='red', label='Regression Line')
plt.title("Uploads vs Video Views")
plt.xlabel('Uploads')
plt.ylabel('Video Views')
plt.legend()
plt.show()

# Print the slope (m) and intercept (c)
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

#@title Performing K-fold Cross-Validation and Creating a new Linear Regression Model
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np

# Create a linear regression object
lin_reg = LinearRegression()

# Define the number of folds for cross-validation
k = 5

# Define the cross-validation method (K-fold)
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize a list to store the model coefficients
coefficients = []

# Perform K-fold cross-validation
for train_index, test_index in kfold.split(features, target):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    lin_reg.fit(X_train, y_train)

    # Store the model coefficients (slope and intercept)
    coefficients.append((lin_reg.coef_[0], lin_reg.intercept_))

# Compute the mean of the coefficients
mean_coefficients = np.mean(coefficients, axis=0)

# Extract the mean slope (m) and intercept (c)
mean_m = mean_coefficients[0]
mean_c = mean_coefficients[1]

#@title Create a Linear Regression Model based on K-fold Cross-Validation results

# Extract the Fe'subscribers' and 'uploads' columns
y_train = youtube_df['video views'].values.reshape(-1, 1)
x_train = youtube_df['uploads'].values

# Create and fit the Linear Regression model with the mean coefficients
mean_model = LinearRegression()
mean_model.coef_ = mean_m
mean_model.intercept_ = mean_c
mean_model.fit(x_train.reshape(-1, 1), y_train)

# Predict subscribers using the model
predicted_uploads = mean_model.predict(x_train.reshape(-1, 1))

# Visualize the data and regression line
plt.scatter(x_train, y_train, marker='D', color='blue', label='Actual Values')
plt.plot(x_train, predicted_uploads, color='red', label='Regression Line')
plt.title("Uploads vs Video Views")
plt.xlabel('Uploads')
plt.ylabel('Video Views')
plt.legend()
plt.show()

# Print the mean slope (m) and intercept (c)
print(f"Mean Slope (m): {mean_m}")
print(f"Mean Intercept (c): {mean_c}")

#@title Create a function that predicts the views
def predict_the_output_subscribers(uploads, m, c):
    return [mean_m * upload + mean_c for upload in uploads]

#@title What if a Youtuber has 1k uploads?
uploads_1k = 1000   # Number of uploads
predicted_views_1k = predict_the_output_subscribers([uploads_1k], m, c)[0]
print(f"Views when uploads are {uploads_1k}: {predicted_views_1k:.2f}")

#@title What if a Youtuber has 100 uploads?
uploads_100 = 100   # Number of uploads
predicted_views_100 = predict_the_output_subscribers([uploads_100], m, c)[0]
print(f"Views when uploads are {uploads_100}: {predicted_views_100:.2f}")

#@title What if a Youtuber has 10 uploads?
uploads_10 = 10   # Number of uploads
predicted_views_10 = predict_the_output_subscribers([uploads_10], m, c)[0]
print(f"Views when uploads are {uploads_10}: {predicted_views_10:.2f}")

#@title What if a Youtuber has 1 upload?
uploads_1 = 1   # Number of uploads
predicted_views_1 = predict_the_output_subscribers([uploads_1], m, c)[0]
print(f"Views when uploads are {uploads_1}: {predicted_views_1:.2f}")

#@title What if a Youtuber has 0 uploads?
uploads_0 = 0   # Number of uploads
predicted_views_0 = predict_the_output_subscribers([uploads_0], m, c)[0]
print(f"Views when uploads are {uploads_0}: {predicted_views_0:.2f}")

#@title Linear Regression Based on Video Views and Uploads
# Imports
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

drive.mount('/content/drive')

# Load the dataset (Global_YouTube_Statistics_2023_pruned.csv)
path = "/content/drive/MyDrive/Colab/Global_YouTube_Statistics_2023_pruned.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')

#@title Filter out trivial data
df = df[df['video views'] > 1 ]
df.reset_index(drop=True, inplace=True)
df = df[df['uploads'] >= 1]

# Divide views column by a 10,000 to make it easier to read
df["video views"]=df["video views"].div(10000)
df

#@title Get Rid of Trvial Data (E.g. urban_population)
df = df[["rank","Youtuber","subscribers","video views","uploads"]]
df

#@title See what in our graph correlates to subscribers
df.corr()["subscribers"]

import seaborn as sns
# Plot the graph between video views and subscribers
sns.lmplot(x="subscribers",y="video views", data=df, fit_reg=True,ci=None)

#@title Split Training and test data
from sklearn.model_selection import train_test_split
features = ['subscribers','uploads']
X = df.loc[:, features]
y = df.loc[:, ['video views']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, train_size = .80)

#@title Import linear regression model
from sklearn.linear_model import LinearRegression
reg= LinearRegression()

#@title Train Linear Regression model
predictors=['subscribers','uploads']
target="video views"
reg.fit(X_train[predictors], y_train["video views"])
#get the slope and y-intercept
m=reg.coef_
c=reg.intercept_
print(m,c)
y_test

#@title Use test data
#use our test data to get a prediction
prediction=reg.predict(X_test)
prediction = pd.DataFrame(prediction, columns =["predictions"])
#reset the index on our y_train data
y_test.reset_index(drop=True, inplace=True)
#add our predictions as a column next to target data
y_test["predictions"]=prediction
y_test

from matplotlib import pyplot as plt
y_test.plot(kind='scatter', x='video views', y='predictions', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

#@title Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mean_abs_error=mean_absolute_error(y_test['video views'],prediction['predictions'])
print(mean_abs_error)

df.describe()["video views"]

#@title Method for predictions
def getVideoViews(x1,x2,m,c):
   return [(m[0] * x1 )+ (m[1]*x2)-c]

#@title How many video views will a person have if he has 20 million subscribers and 200 uploads?
twent_mil_200=getVideoViews(20000000,200,m,c)
print("a person will have",twent_mil_200[0].round(2)*1000,"if they have 20 million subscribers and 200 uploads")

#@title How many video views will a person have if he has 13 million subscribers and 30 uploads?
thirteenmil=getVideoViews(13000000,30,m,c)
print("a person will have",thirteenmil[0].round(2)*1000,"if they have 13 million subscribers and 30 uploads")

#@title Classification Model
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive - the file is too large,
# so we opted for this method of uploading the file to Colab.
drive.mount('/content/drive')

#Read the file from Google Drive
file_path = '/content/drive/MyDrive/Colab/Global_YouTube_Statistics_2023_pruned.csv' #Set file path as needed. The file will be selected from Google Drive.
youtube_df = pd.read_csv(file_path, encoding='ISO-8859-1') #Gabe removed 'header=1'

# @title Simplifying the Data

# Remove rows with 'NaN' values
youtube_df = youtube_df.dropna()

# Select relevant features and target variable
features = ['subscribers', 'video views', 'uploads', 'created_year', 'highest_monthly_earnings', 'Population']
target = 'category'

# Remove non-relevant columns and drop rows with missing values (if any)
youtube_df = youtube_df[features + [target]].dropna()

youtube_df.head()

#@title Displaying correlations
# Calculate the correlation between channel category and all features
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
youtube_df['category_encoded'] = label_encoder.fit_transform(youtube_df["category"])

correlation = youtube_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=1)
plt.title('Correlation')
plt.show()

#@title Creating the Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = youtube_df[features]
y = youtube_df['category_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

# Create and train a logistic regression model
logreg_model = LogisticRegression(max_iter=40000, random_state=42)
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred = logreg_model.predict(X_test)

unique_categories = sorted(y.unique())

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, labels=unique_categories, zero_division=1)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

#@title Visualizing Results with a Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#@title Results Analysis

print("""Upon exploring possibilites for potential classification problems, we believe our dataset is better suited to
linear regression problems. The sample classification problem above showcases some of the issues the dataset holds for
classification problems. These issues include relatively weak correlations between many of the fields, a wide range of unique
values in categorical fields, a relatively small sample which by the nature of the dataset is skewed towards highly
succesful YouTube channels.""")