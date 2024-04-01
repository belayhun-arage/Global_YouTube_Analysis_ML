# Project documentation
# Global Youtube Analysis

**1. Problem Definition and Understanding**
1. We are encountering a **Learning Problem**. We are going to use global Youtube statistics for 2023 to analyze a Youtuber's subscriber count and their upload count, as well as other relationships between attributes.

2. The task is to determine how a Youtuber's **upload count** will affect their **subscriber count**.

3. This is a great oppportunity to analyze how much a Youtuber needs to post on average to grow their audience. Possible constraints are varying upload schedule of the Youtuber over the years as well as channels that provide other services, such as Youtube's own Gaming channel, which focuses on connecting users with up and coming channels.


**2. Data Collection and Processing**

1. To solve the problem, we are using a dataset with global Youtube Statistics as of 2023.

2. To clean up the data, we edited the names of some of the Youtubers. Some characters in the names were from their respective country so names were edited to be represented using English language letters. For example, "ช่อง8 : Thai Ch8" was edited to be shown as simply "Thai Ch8".
  - Although not specific to our current scope, each channel had their country of origin added where they were not shown before. Furthermore, attributes that were not a numeric value were removed to help with the processing of the data.
  - Seeing that there is a lot of data with varying subscriber counts and upload counts, the data pool was reduced to United States channels to reduce the amount of noise in the model.

3. To explore the data, we found important information about the data such as the top youtuber's views, upload counts, top subscriber counts, an analysis of the shape of the dataset, the types of attributes, and descriptive statistics of the dataset.

**3. Model Selection and Implementation**

1. We selected the Linear Regression Model to determine how the views of a channel can predict the subscriber count of a Youtube channel. This is classified as Supervised Learning.

2. To achieve the model, several libraries were used:
  - pandas
  - matplotlib
  - google.colab
  - seaborn
  - numpy
  - sklearn
3. To optimize and fine tune the model, some data was removed:
  - Any rows or attributes that were not numbers were removed.
  - Any channels with more than 6,000 uploads were removed.
  - All countries that were not the United States were removed.
  - All channels with 1 view or less were removed.
We found the removal of this data to result in more accurate predictions.

4. While this is a good model, the amount of solid correlations between the different attributes in the data is scarce. Most of the correlation was between the amount of views and revenue made in the past 30 days, as well as a strong correlation between the views and subscriber count. This makes it difficult for the model to find a robust prediction. Perhaps more tuning is required.

**4.  Model Optimization and Tuning**

1. At first the training for the model was the data in the order that it came in: by subscriber count, descending. We first tried running the model with the features as the upload count of each Youtube channel and the subscriber count as the target. As the model was fine tuned, we focused more on the relation between video views and subscriber count because the correlation was more solid.

2.  Training the model by descending subscriber count worked, but K-fold cross-validation was used to experiment wiht the model training. Cross-validation provided a better trained model, shown by  a more accurate slope and intercept.

**5. Data Visualization**
1. Although we started off with subscriber count and upload count as our respective target and feature, it became clear that there is a stronger correlation between video views and subscriber count. To analyze the correlation between all attributes, we created a bar graph that showed that one of the best relations was subscriber count and video views.

2.  To further visualize the correlation of the data, we created a heat map that more clearly illustrated the relations between each attribute and the rest. The results were clear that one of the best relations is between subscriber count and video views, which reinforced our decision to focus on video views and subscriber count.

**6. Results and Conclusion**

1. The results indicate there is a strong relation between video views and subscriber count. However, it must be taken into account that the Youtube algorithm that determines recommendations to users are quite complex. A Youtuber's subscriber count or upload count may mean they get more views, but at Youtube there are more factors that are taken into account such as video length, age rating, past strikes a channel has taken, and overall engagement with a video such as like/dislike ratio and discussion in the comments.
