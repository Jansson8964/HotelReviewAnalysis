# APPLIED BIG DATA AND VISUALIZATION REPORT (Group 17)

## Hotel Review Analysis Using Spark

![hotel-image](https://github.com/user-attachments/assets/cd24df00-9613-4662-9c61-dac66aaaedb4)

- Tammay Srivastava - 23118067
- Leiliane da Silva Claudio - 23256516
- Yeshwanth Nataraj - 23228954
- Song Yang - 23042427
- Suryanarayanan Subhashchandrabose - 23009225

## INTRODUCTION

Our project aims to utilize Spark to analyze hotel reviews. By implementing machine learning models for sentiment classification, this predictive model will determine whether reviews are negative or positive. Focusing our analysis on a specific hotel, we will compare its performance with other hotels in the same area. This comparison and insight will assist hotels in accurately classifying their reviews and improving their services based on customer feedback.

## DATA EXPLORATION

### How The Data Was Collected

The dataset used in this project originates from a research study conducted by Diego Campos, Rodrigo Rocha Silva, Jorge Bernadino, and their team at the University of Coimbra, focusing on "Text Mining in Hotel Reviews: Impact of Words Restriction in Text Classification." Originally obtained from Kaggle, this dataset comprises a substantial volume of data, featuring over 515,738 entries spread across 17 columns. Notably, the data was collected from Booking.com, a prominent online platform for hotel bookings. Its significance lies in the fact that it contains genuine reviews from real customers, making it a valuable resource for training our model.

![image](https://github.com/user-attachments/assets/a84f3c76-e931-4deb-9178-9262a5bfbe50)

### Challenges

While the dataset maintained a well-structured format, it posed certain challenges. One notable issue was the segregation of reviews into distinct positive and negative categories. While this categorization serves specific analytical needs, it may not accurately reflect the reality of most review datasets, which often contain unclassified feedback. As a result, relying solely on models that focus solely on identifying positive and negative sentiments could limit their usefulness, particularly in scenarios where reviews are not pre-categorized. This solution was implemented by merging the positive and negative reviews, sampling a portion of the data to manage its size efficiently, creating binary classification for reviewer scores, addressing class imbalance, and performing necessary data preprocessing steps using PySpark.

### Hadoop Analysis

In this project, Hadoop has been instrumental in managing and processing our large-scale hotel reviews dataset. Utilizing Hadoop’s distributed file system (HDFS), we efficiently stored and organized our data across a cluster, ensuring resilience and scalability. The HDFS directory listings reflect the successful execution of data pre-processing steps, including cleaning and spell-checking, preparing the dataset for advanced analytics.

<img width="452" alt="Picture2" src="https://github.com/user-attachments/assets/fdb69a47-4442-4aa7-a5d8-166e2dc4657d">

<img width="452" alt="Picture3" src="https://github.com/user-attachments/assets/f1edec13-4333-4528-9918-fb9d02981d2e">

### Understanding the Dataset

A thorough understanding of the dataset is essential to avoid errors during data analysis and modeling. We comprehensively inspected the dataset, which involved examining all column names such as `Hotel_Name`, `Negative_Review`, `Positive_Review`, `Reviewer_Score`, `Reviews_Clean`, `Score`, and `Spell_Checked`. This examination helped us comprehend what each column represents, facilitating our subsequent data handling and analysis methods.

Statistical summaries of numeric columns revealed a `Reviewer_Score` ranging from 2.5 to 10, with a mean close to 6.83, indicating moderate average satisfaction among reviewers. A review of the first and last few entries provided insights into the varying sentiments expressed in the reviews, from highly positive to distinctly negative. Furthermore, the dataset contains a wide variety of unique entries across text fields like reviews, with 1,425 unique hotel names, highlighting substantial variability and a broad scope for detailed analysis.

Since we faced a classification problem, we first created a target feature based on the score. We observed that scores ranged from a low of 2.5 to a high of 10. Given that some reviews indicated minimal satisfaction, we assumed that the lowest score of 2.5 might have been assigned by the platform rather than by the users themselves. We thus transformed this into a binary classification problem, categorizing scores below 6 as negative and scores of 6 and above as positive, with 6 serving as the threshold for neutrality.

### Class Distribution

![image (13)](https://github.com/user-attachments/assets/af66cb90-2121-4a79-beed-eb16608530fd)

An analysis of the class distribution revealed a significant imbalance. To address this issue, PySpark's capabilities were leveraged to balance the classes by undersampling the positive reviews. Specifically, only 12% of the positive reviews were sampled, aligning the distribution more closely with that of the negative reviews.

![Picture5](https://github.com/user-attachments/assets/3190805c-0df2-47f5-978c-aff045a0f868)

### Loading the Dataset

After acquiring the dataset, the initial step in our pre-processing was to load it into our working environment using PySpark. We initiated a Spark session to facilitate distributed data processing. The dataset, saved in CSV format, was loaded into a Spark DataFrame.

For text processing and analysis, we utilized additional Python libraries. This included using the NLTK library for natural language tasks and TextBlob for deriving insights from text data. Visualization of specific analysis results was accomplished using Matplotlib and Seaborn libraries within Python. These tools were employed to generate plots and graphics to better understand data trends and distributions.

![image (12)](https://github.com/user-attachments/assets/4224b320-0c06-4fd4-9c5f-bd8e49d4939f)

![image (10)](https://github.com/user-attachments/assets/e1fc007c-3a1b-40a4-8cce-5201058af64b)

### Data Cleaning

Data cleaning is a crucial step in our preprocessing workflow, essential for making sense of the information and creating meaningful features for our models. We utilized the capabilities of Spark for all data cleaning tasks, ensuring scalability and efficiency in our operations. Here are the specific steps we took using Spark:

- **Handling Null Values:** We utilized Spark functions to detect and handle null values in each column of our DataFrame. Depending on the scenario, we filled null values with appropriate substitutes or omitted the rows/columns containing them.
- **Data Formatting:** Ensuring that all data entries are consistently formatted, for example, dates in a uniform format, text in the same case (lower or upper), etc.
- **Error Correction:** Identifying and correcting errors in the dataset, such as typographical errors or inconsistent entries using techniques like regex (regular expressions).

![image (6)](https://github.com/user-attachments/assets/bc66d925-4dfa-4d54-8b1e-7340d54237ba)

## EXPLORATORY DATA ANALYSIS

### Visualization

![image (16)](https://github.com/user-attachments/assets/6db19ee5-d214-461b-92f2-90146a40d8bd)

![image (20)](https://github.com/user-attachments/assets/81f4cea7-11be-42da-bcb7-7eae703b9f82)

![image (20)](https://github.com/user-attachments/assets/6c3f771b-b5f1-4d23-ab90-e2f8afec8574)

![image (17)](https://github.com/user-attachments/assets/c6eaf074-0aa2-41bd-a274-13f7ee08cb72)

### Word Cloud

#### Why is the Word Cloud Important?

While choosing the appropriate dataset, we noticed that the review scores were not matching the guest sentiment about the hotel. The mismatch becomes clear in the scores between 6 and 7. Please see an example below. Keep in mind that the punctuations were removed in the data cleaning and misspellings are common in the reviews.

![Snipaste_2024-09-14_10-34-03](https://github.com/user-attachments/assets/0e9d703b-db39-4eea-b8bc-495eabd375b4)

As we can see above, the review doesn't match the overall score. If there is nothing positive about the hotel, how can they still get a 6.5 score? This is misleading to the hotel looking for areas to improve and to the users looking for a trustworthy score.

#### For the Results Section

I had two questions in mind:

1. What words appear the most in positive and negative reviews?
2. Can we get any insights from it?

Since we've already done the data cleaning, we can now explore the word cloud to solve the above problem.

![image (3)](https://github.com/user-attachments/assets/0d369daa-53e4-4d67-aa4f-d2fa1b4c7ef1)

![image (2)](https://github.com/user-attachments/assets/3f739d27-0568-4499-8044-9f41ef88ed8a)

In our analysis, we employed advanced text processing techniques and visualization tools to comprehensively understand the sentiments expressed in hotel reviews. We began by flattening nested lists of words from both positive and negative reviews into singular, manageable lists. Utilizing the Natural Language Toolkit (NLTK)'s `FreqDist` function, we quantified the frequency of each word, allowing us to pinpoint the most prevalent terms associated with each sentiment.

For the visualization aspect, we chose Matplotlib and Seaborn libraries to create clear and informative bar graphs that illustrate the top 30 most frequent words in both positive and negative reviews. We enhanced these visualizations with color gradients from Matplotlib’s color maps, facilitating an intuitive understanding of the data. This graphical representation not only highlights the predominant words but also serves as an effective communication tool in our report, providing immediate insights into customer feedback trends.

![1](https://github.com/user-attachments/assets/b34fe5b6-d14f-48f2-9aad-659779df1e5f)

It appears that "room," "location," "staff," "clean," and "friendly" are the most popular words in positive reviews. The management can use these words to investigate where the hotel is doing a good job.

![2](https://github.com/user-attachments/assets/3f8fd9f5-9e79-4690-bfc7-8f1340293e2c)

In contrast, the most common words in negative reviews include "room," "staff," "breakfast," and the fact that the hotel might be old. This suggests that issues related to these aspects are frequent sources of complaints. The word "room" appears more frequently in negative reviews than in positive ones.

![3](https://github.com/user-attachments/assets/33a5b0b1-d818-4899-bf45-ecb427dd36b2)

We can see that all the words make sense and are related to hotels. This indicates that the stop words were effectively removed. The hotel management can use these insights to understand what guests who leave negative or positive reviews are discussing.

![4](https://github.com/user-attachments/assets/17e4d165-1d50-4ab4-90b9-8e8773f63eff)

**Visualizations Generated Using Power BI**

We also generated visualizations with the help of Power BI to compare the analysis with the one obtained from pandas-profiling reports.

![WhatsApp Image 2024-04-22 at 02 26 42_9f8211fb](https://github.com/user-attachments/assets/87d529f2-135e-4ead-b0e4-78ae6d5f21de)

The processed data was imported into Power BI to create an interactive dashboard. This dashboard includes built-in filters and hierarchies, allowing hoteliers to view recent negative reviews. Using the search function for negative and positive reviews, the hotelier can choose weighted words obtained from the word cloud and search for them throughout the reviews.

![Snipaste_2024-09-14_10-42-54](https://github.com/user-attachments/assets/0cfe6b70-cf0a-407b-a849-afd3579e6429)

The hotelier can select specific reviews and obtain details about the review date, nationality, and score. This functionality helps hoteliers analyze their shortcomings, gain insights from significant negative words in reviews, and take a streamlined approach to increase customer satisfaction and improve the hotel score.

![Snipaste_2024-09-14_10-40-41](https://github.com/user-attachments/assets/c55ada89-c98d-4bf4-9ec1-56cceee47aed)

This dashboard also allows hoteliers to check details about other hotels in the same city, gaining insights such as popular tags for the most successful quarter.

Upon selecting the required filters like city and hotel name, summary data is presented to the user, including the total number of reviews, hotel address, average score, average score per quarter, and popular tags for the hotel per quarter.

![Snipaste_2024-09-14_10-44-18](https://github.com/user-attachments/assets/764bba91-56ff-4de6-a291-0790fb3097b0)

## Next Steps ##

### Confirm System Configuration ###

We will verify that our Hadoop and Spark systems are optimally configured to handle our data processing and storage needs effectively. This includes ensuring that HDFS is set up correctly for secure and efficient data storage.

### Model Development and Deployment ###

- **Sentiment Analysis Model with Spark NLP**: We will utilize Spark NLP, which offers robust capabilities for natural language processing, to implement our sentiment analysis model. This will involve selecting an appropriate pre-trained model from Spark NLP's extensive library or training a new model if a more customized solution is necessary.
- **Training the Model**: If a new model is required, we will train this model using a labeled subset of our data, where reviews have been categorized as negative or positive. This training process will involve feature engineering techniques such as tokenization, stemming, and perhaps the use of embeddings like Word2Vec or GloVe to enhance the model's understanding of textual nuances.

### Model Optimization and Evaluation ###

We will optimize the model by tuning hyperparameters and employing techniques such as cross-validation to ensure robustness and high accuracy. The model’s performance will be evaluated using standard metrics like accuracy, precision, recall, and F1-score.

### Analysis and Correlation Studies ###

- **Applying the Model**: The trained model will be applied to the entire dataset to categorize each review. This will generate a comprehensive set of data with sentiment labels that can be further analyzed.
- **Correlation Analysis**: With sentiment labels in place, we will use statistical methods to explore correlations between review sentiments and various factors such as the geographic location of hotels, reviewer demographics, and specific feedback content. This analysis will help identify key drivers of positive and negative sentiments.
- **Advanced Analytics**: We will also implement more complex analytical techniques such as clustering to identify patterns and trends and regression analysis to predict the impact of specific variables on review sentiments.

### Configuration of Apache Spark Environment ###

As part of our data analysis setup, we successfully installed and configured Apache Spark. This involved a systematic approach to establish a robust computational environment tailored for large-scale data processing.

- **Java Development Kit Installation**: We installed the Java Development Kit, a prerequisite for running Spark.
- **Apache Spark Installation**: We downloaded and unpacked Apache Spark version 3.3.0 and ensured the integrity of the downloaded files.
- **PySpark Installation**: Using the `pip` command, we installed PySpark to bridge Spark with our Python environment.
- **Environment Variables**: We set environment variables to link our installation paths for Java and Spark, integrating these components seamlessly into our system.
- **Library Paths**: We added necessary library paths to Python’s system path, facilitating direct imports and usage of Spark functionalities.
- **Spark Session**: We initiated a Spark session and verified its functionality through a test DataFrame, confirming the operational status of Spark within our environment.
![image](https://github.com/user-attachments/assets/d9fc9161-3949-43ed-acba-b2cb9ae000d9)

### Sentiment Analysis of Hotel Reviews: Machine Learning Model Report ###

![Snipaste_2024-09-14_10-49-23](https://github.com/user-attachments/assets/104f6206-6fba-446f-9e3d-48e8e1ac42e1)

![Snipaste_2024-09-14_10-49-31](https://github.com/user-attachments/assets/90a37c53-13cf-4f03-abcf-febe6ccd3fe4)

As a result, we obtained an accuracy of the trained model on the test data of approximately 82.1%.

![Snipaste_2024-09-14_10-49-36](https://github.com/user-attachments/assets/bf0b9a10-3a15-444d-86a5-612aec86c447)
