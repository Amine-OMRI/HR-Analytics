


---
<p><img alt="https://www.ceratizit.com/" height="120px" src="https://consent.trustarc.com/v2/asset/09:13:49.986pc83s3_CERATIZIT_G_WEB_color_transparent.png" align="left" hspace="10px" vspace="0px"></p>  <h1>HR-Analytics</h1>
Data science open position challenge in order to lay the foundation for a technical discussion during the interview with the condidate 



---

## Table of content

* [0. Modules and versions](#modules)
* [1. About the context of this study](#content)
* [2. Getting started: basic steps](#Steps)
  * [a. Preprocessing](#preprocessing)
  * [b. Modeling](#modeling)
  * [c. Evaluation Metrics](#scoring)
* [3. Models benchmarking](#benchmarking)
  * [a. Unbalanced Data](#unbalanced)
  * [b. SMOTE Data](#smote)
* [4. Best Mode selection](#bestmodel)
  * [a. Fine tuning](#tuning)
  * [b. Scores](#scores)
* [5. Submission](#submission)
	
## 0. Modules and versions <a name="modules">
	
* pandas==1.1.5
* numpy==1.19.5
* matplotlib==3.2.2
* seaborn==0.11.1
* category-encoders==2.2.2
* regex==2019.12.20
* xgboost==1.4.2
* scikit-learn==0.22.2.post1
* imbalanced-learn==0.4.3

## 1. About the context of this study <a name="content">

### a. **Business or activity:**</br>
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and `categorization` of candidates. `Information related to demographics, education, experience are in hands from candidates signup and enrollment`.

The given dataset was designed to understand the factors that lead a person to leave their current job for HR research as well. Through a model(s) that uses current data on `qualifications`, `demographics`, and `experience`, we will be able to `predict the likelihood of a candidate seeking a new job or working for the company`, as well as `interpreting the factors that influence the employee's decision.`

The whole data divided to train and test . Target isn't included in test but the test target values data file is in hands for related tasks. A sample submission correspond to enrollee_id of test set provided too with columns : `enrollee _id , target`(submission format).

**Note:**
`The dataset is imbalanced. Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality. Missing imputation can be a part of your pipeline as well.`

---
* Since we have labeled data, we are typically in the context of supervised learning.

* From the brief description of the problem, we can notice that we are facing a classification task and that the data are already collected. Therefore, there is no need to collect data unless we want to enrich the data to get more relevant information. 

* We are therefore looking for a classifier that produces probabilities of belonging to one of two classes that predict the probability that an applicant will seek a new job or work for the company, a classifier that predicts not the classes to which the examples belong, but the probability that the examples fit a particular class as well as the interpretation of the factors influencing the employee's decision.

* We can either use any binary classifier to learn a fixed set of classification probabilities (e.g., p in [0.0, 1.0] ) or conditional probability models which can be generative models such as naive Bayes classifiers or discriminative models such as logistic regression (LR), neural networks, if we use cross-entropy as the cost function with sigmoidal output units. This will provide us with the estimates we are looking for.




## 2. Getting started<a name="Steps">
 The basic steps that I will follow to address this problem and ensure to achieve business value and minimize the risk of error are as follows: 

1.   Understanding the Business
2.   Loading the data
3.   Preprocessing and exploratory analysis
4.   Predictive modeling
5.   Interpretation of results

Ps: we could iterate between the steps depending on our objectives and the result of each step..

### a. Preprocessing  <a name="preprocessing">
	#### Concatenation of aug_train and aug_test

* Since we have the `aug_train` dataset that contains the `target`, which is `not the case` with the `aug_test` dataset, which, as mentioned, will be used for submission and the dataset is `unbalanced`. Most features are categorical (nominal, ordinal, binary), some with high cardinality. Imputation of missing data may be part of our pipeline. **`We will need to concatenate the aug_train and aug_test data`**. Otherwise, we may run into problems if there is a categorical attribute whose values are not all present in the `aug_train` and `aug_test` data.

* We will `isolate the target column` and perform the necessary processing (Imputation, encoding, ...) on the `concatenated aug_train and aug_test` after that we will separate them again into `train` and `test` just like they were at the beginning.

* After processing the `concatenated sets` and separating them to recover the original data set with the same dimensions, we will bring the `target column back to the preprocessed train set` and use it to create three different sets for TRINING, TEST and VALIDATION. for the modeling part
	
  #### Cleaning the data
  The strategy for cleaning the data is as follows:
  * Delete columns that are not relevant to the problem. 
  * Find the missing values for each column.
  * Drop columns that have more than 20% missing data. Write down the columns.
  * Convert the columns to their correct data type.
  * Encode the categorical variables using the appropriate encoding methodologie.
  #### List of preprocessing approaches
  We used different pre-processing to deal with missing values, categorical values with high cardinalities and below the list of approaches
  		
	* **App 1:** Binary encoded city + Combined Ordinal encoded experience + imputing missing values with the most frequent value (the mode)
	* **App 2:** Binary encoded city + Ordinal encoded experience + imputing missing values with the most frequent value (the mode)
	* **App 3:** Hashing encoded city + Combined Ordinal encoding experience + imputing missing values with the most frequent value (the mode)
	* **App 4:** Hashing encoded city + Ordinal encoding experience + imputing missing values with the most frequent value (the mode)
	* **App 5:** One-hot encoded city + Combined Ordinal encoded experience + imputing missing values with the most frequent value (the mode)
	* **App 6:** One-hot encoded city + Ordinal encoded experience + Replacing missing values with new category = "missing" in Gender, Company_size, Company_type
	* **App 7:** One-hot encoded city + Ordinal encoded experience + imputing missing values with the most frequent value (the mode)
	* **App 8:** One-hot Combined (cat with freq <1%) encoded city + Ordinal encoded experience + Replacing missing values with new category = "missing" in Gender, Company_size, Company_type
	* **App 9:** One-hot Combined (cat with freq <1%) encoded city + Combined Ordinal encoded experience + Replacing missing values with new category = "missing" in Gender, Company_size, Company_type
	* **App 10:** One-hot Combined (cat with freq <1%) encoded city + Combined Ordinal encoded experience + imputing missing values with the most frequent value (the mode)
	

### b. Modeling  <a name="modeling">
   We have used three different models:
   * XGBoost (**OUR BEST MODEL**): Boosting is an ensembling technique where new models are added to correct errors made by existing models. Models are added sequentially until no further improvements can be made. A popular example is the AdaBoost algorithm that weights data points that are difficult to predict.
	Gradient boosting is an approach that creates new models that predict the residuals or errors of the previous models, then adds them together to get the final prediction. It is called "gradient boosting" because it uses a gradient descent algorithm to minimize the loss when adding new models.
	XGBoost (eXtreme Gradient Boosting) is a gradient boosted decision tree implementation designed for speed and performance. It is a very popular algorithm that has recently dominated applied machine learning for structured or tabular data
   * Linear SVM
   * CatBoostRegressor
   **Note** we tried them all with unbalanced data and with balanced SMOTE data.
	
### c. Evaluation Metrics  <a name="scoring">

We have two classes we need to identify — appicants that gonna change the company and those who will not — with one category representing the overwhelming majority of the data observations.
We might focus on identifying the **positive cases**. The metric our intuition tells us we should maximize is known in statistics as **recall**, or the ability of a model to find all the relevant cases within a dataset. The precise definition of recall is `(TP) / (TP + FN)`. 
* **TP:** are data point classified as positive by the model that actually are positive (meaning they are correct)
* **FN:** are data points the model identifies as negative that actually are positive (incorrect). 
	
In our case, TP are correctly identified 1 – Looking for a job change, and FN would be individuals the model labels as  0 – Not looking for job change that actually were actually looking for a job. Recall can be thought as of a model’s ability to find all the data points of interest in a dataset.

**When we increase the recall, we decrease the precision** and our model would probably not be useful, as we would then have many **FPs**, i.e., many applicants labeled 1 - Looking for a job change, who were in fact not looking for a job. this model would suffer from low precision, or the ability to identify only relevant data points.
* **FP:** are data points the model identifies as positive that actually are negative (incorrect). 
	
The precise definition of **precision** is `(TP) / (TP + FP)` and it expresses the proportion of the data points our model says was relevant actually were truely relevant.

In some situations, we might know that we want to maximize either recall or precision at the expense of the other metric. However, in our case where we want to find an optimal blend of precision and recall we can combine the two metrics using what is called the **F1 score**.

The F1 score is the harmonic mean of precision and recall taking both metrics into account in the following equation:<p align="center"> ![the F1 Score](https://miro.medium.com/max/376/1*UJxVqLnbSj42eRhasKeLOA.png)</p>

	
Since we are dealing with unbalanced data, we use the **F1 score** as the evaluation metric most of the time, but we also check the **Accuracy** and the **roc_auc_score** on the test data.
	
#### Observations of Precision and Recall: 
	
* **ROC Curves** summarize the trade-off between the true positive rate (**TPR**) and false positive rate (**FPR**) for a predictive model using different probability thresholds. The threshold represents the value above which a data point is considered in the positive class. Altering this threshold, we can try to achieve the right balance between the false positives and false negatives, we can quantify a model’s ROC curve by calculating the total Area Under the Curve (**AUC**), a metric which falls between 0 and 1 with a higher number indicating better classification performance.
	<p align="center"> ![the TPR and FPR Score](https://miro.medium.com/max/2000/1*HzWxvbikCtiB-QtFb48WoQ.png)</p>
	
* **Precision-Recall curves** summarize the trade-off tradeoff between precision and recall for different threshold. A high area under the curve (**AUC**) represents both high recall and high precision, where high precision relates to a low false positive rate (**FPR**), and high recall relates to a low false negative rate (**FNR**). High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).

ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.The focus on the minority class makes it an effective diagnostic for imbalanced binary classification models.
	
	
## 3. Models benchmarking<a name="benchmarking">
	
### a. Unbalanced Data  <a name="unbalanced">

	XGBoost: UNBALANCED DATA             
| Approach | Accuracy | F1-score | AUC | Precision 0.0 | Precision 1.0 | Recall 0.0 | Recall 1.0 | F1-score 0.0 | F1-score 1.0 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |  :-: | :-: | :-: |
| app 1 | 78.81% | 0.486076 | 0.66 | 0.83 | 0.58 | 0.90 | 0.42 | 0.87 | 0.49 |
| app 2| 78.91% | 0.493734 | 0.66 | 0.83 | 0.58 | 0.90 | 0.43 |  0.87 | 0.49 | 
| app 3| 78.81% | 0.486076 | 0.66 | 0.83 | 0.58 | 0.90 |  0.42 | 0.87 | 0.49 |
| app 4| 79.33% | 0.507463 | 0.67 | 0.84 | 0.59 | 0.90 |  0.45 | 0.87 | 0.51 |
| app 5| 79.18% | 0.494297 | 0.66 | 0.83 | 0.59 | 0.91 |  0.43 | 0.87 | 0.49 |
| app 6| 79.49% | 0.547756 | 0.70 | 0.85 | 0.58 | 0.88 |  0.52 | 0.87 | 0.55 |
| app 7| 79.07% | 0.494325 | 0.66 | 0.83 | 0.59 | 0.90 |  0.43 | 0.87 | 0.49 |
| app 8| 79.54% | 0.544186 | 0.69 | 0.85 | 0.58 | 0.88 |  0.51 | 0.87 | 0.54 |
| app 9| 79.38% | 0.505632 | 0.67 | 0.84 | 0.59 | 0.90 |  0.44 | 0.87 | 0.51 |
| app 10| 79.28% | 0.535673 | 0.69 | 0.85 | 0.58 | 0.88 |  0.50 | 0.87 | 0.54 |


### b. SMOTE Data  <a name="smote">
  
	XGBoost: SMOTE DATA         
| Approach | Accuracy | F1 score |
| --- | --- | --- |
| app 1 | 85.28 | 0.85 |
| app 2| 85.45 | 0.84 |
| app 3 | 85.30 | 0.84 |
| app 4| **86.86** | **0.86** |
| app 5| 85.78 | 0.85 |
| app 6| 85.23 | 0.84 |
| app 7| 85.23 | 0.84 |
	
## 4.  Best Mode selection <a name="bestmodel">
	
the best model that performed well was XGBoost with the below scores:
* On SMOTE Balanced data : Accuracy = 86.86 F1 score = 0.867
* On the initial unbalanced data : Accuracy = 80 F1 score = 0.55

### a. Fine tuning  <a name="tuning">
	
The best parameters of our model after Fine Tuning are the following:
* eta : 0.1
* subsample = 1.0
* colsample_bytree = 0.4
* max_depth = 8
* minimum_child_weight = 6
	
### b. Scores  <a name="scores">
Accuracy: 86.86%
F1 score: 0.867740
| --- | precision | recall | f1-score | support|
| --- | --- | --- | --- | --- |
| 0.0 | 0.86 | 0.87 | 0.87 | 1440 |
| 1.0 | 0.87 | 0.86  | 0.87  | 1437 |
| accuracy |  |  | 0.87 | 2877 |
| macro avg | 0.87 | 0.87 | 0.87 | 2877 |
| weighted avg | 0.87 | 0.87 | 0.87 | 2877 |
            


## 5. Submission  <a name="submission">
Thank you for taking the time to learn more about the solution we have implemented, we appreciate your feedback. 
Here is the link to the CSV [file](https://github.com/Amine-OMRI/HR-Analytics/blob/main/Submission/submission.csv) of the submission.

**Note** Please use Googel Colab to open the notebooks and re-run the code to see the result of each cell as there are many code cells and colab will not show them if you open them in Github.

