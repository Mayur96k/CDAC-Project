
# Why do we need Resume Screening?

For each recruitment, companies take out the resume, referrals and go through them manually.
Companies often received thousands of resumes for every job posting.
When companies collect resumes then they categorize those resumes according to their requirements 
and then they send the collected resumes to the Hiring Team's.
It becomes very difficult for the hiring teams to read the resume and select the resume according 
to the requirement, there is no problem if there are one or 
two resumes but it is very difficult to go through 1000’s resumes and select the best one.
To solve this problem, we will screen the resume using machine learning and Nlp using Python so that
we can complete days of work in few minutes.


# Introduction :-

Resume screening is the process of determining whether a candidate is qualified for a role based on his 
or her education, experience, and other information captured on their resume.
It’s a form of pattern matching between a job’s requirements and the qualifications of a candidate based 
on their resume.
The goal of screening resumes is to decide whether to move a candidate forward – usually onto an interview – 
or to reject them.




    



## Prerequisites

- Python 



# Modules & Libraries Description

Modules :-
    
KNN
 
 It's supervised technique, used for classification. "K" in the KNN repersent the number of nearest 
        neighbours used to classify or predict in case of continuous variable.

NLP
 
 NLP is a field in machine learning with the ability of a computer to understand, analyze, manipulate, 
        and potentially generate human language.

Libraries :-
    

NumPy

NumPy is one of the fundamental packages for Python providing support for large multidimensional arrays 
        and matrices

Pandas

It is an open-source, Python library. Pandas enable the provision of easy data structure and quicker data 
        analysis for Python. For operations like data analysis and modelling, 

Matplotlib

This open-source library in Python is widely used for publication of quality figures in a variety of hard copy formats and interactive environments across platforms.
We can design charts, graphs, pie charts, scatterplots, histograms, error charts, etc. with just a few lines of code.

Seaborn

When it comes to visualisation of statistical models like heat maps, Seaborn is among the reliable sources. This Python library is derived from Matplotlib and closely integrated with Pandas data structures.

Scipy

This is yet another open-source software used for scientific computing in Python. Apart from that, Scipy is also used for Data Computation, productivity, and high- performance computing and quality assurance.

Scikit-learn

It is a free software machine learning library for the Python programming language and can be effectively used for a variety of applications which include classification, regression, clustering, model selection, naive Bayes’, grade boosting,K-means, and preprocessing.

Nltk

Natural Language toolkit or NLTK is said to be one among the popular Python NLP Libraries. It contains a set of processing libraries that provide processing solutions for numerical and symbolic language processing in English only.


# Functionality of Application

Screening resumes usually involves a three-step process based on the role’s minimum and preferred qualifications. Both types of qualifications should be related to on-the- job performance and are ideally captured in the job description.

These qualifications can include:

Work experience
Education
Skills and knowledge
Personality traits
Competencies

# Tools & Technologies used

Machine Learning  along with text mining and natural language processing algorithms, can be applied for the development of programs 
(i.e. Applicant Tracking Systems) capable of screening objectively thousands of resumes in few minutes without bias to identify the 
best fit for a job opening based on 
thresholds, specific criteria or scores.


# Tech innovations in resume screening

Designed to meet the needs of recruiters that current technology can’t solve, a new class of recruiting technology called AI for recruitment has arrived.
AI for recruiting is an emerging category of HR technology designed to reduce — or even remove — time-consuming, administrative activities like manually screening resumes.The best AI software is designed to integrate seamlessly with your current recruiting stack so it doesn’t disrupt your workflow nor
the candidate workflow.Industry experts predict this type of automation technology will transform the recruiting function.

# The project typically involves the following steps:

1.Data collection: 
Gather a large dataset of resumes, ideally with diverse backgrounds and job roles. 
This dataset will be used to train and evaluate the machine learning model.

2.Preprocessing: 
Clean and preprocess the resume data to remove irrelevant information, such as formatting, 
special characters,Punctuations and stopwords. This step may also include techniques like tokenization to normalize the text.

3.Feature extraction: 
Extract relevant features from the preprocessed resumes. This could involve techniques such as bag-of-words, TF-IDF (Term Frequency-Inverse Document Frequency), or word embeddings like Word2Vec or 
GloVe. These features represent the resumes in a numerical format that machine learning algorithms can process.

4. Labeling or classification: 
Assign labels or categories to the resumes based on specific criteria. For example,if the job requires a certain skill set or experience level, resumes can be classified into "relevant" or "not 
relevant" categories accordingly. This step often requires manual labeling of a subset of the data by human experts to serve as 
training examples for the machine learning model.

5.Model training: 
Train a machine learning model using the labeled dataset. Common algorithms used for resume screening include logistic regression, support vector machines (SVM), random forests, or deep learning models such as convolutional 
neural networks (CNN) or recurrent neural networks (RNN).

6.Model evaluation: Assess the performance of the trained model using evaluation metrics such as accuracy, precision, recall,
or F1-score. This step helps determine the effectiveness of the model and identify areas for improvement.

7.Deployment: Deploy the trained model to a production environment where it can be used to automatically screen and rank new resumes based on their relevance to the job position. The model takes in a resume as input, applies the learned patterns and features, and outputs a prediction or score indicating the suitability of the candidate.

## Deployment

Pickle Module:

The pickle module in Python is used for serializing and deserializing Python objects. Serialization refers to the process of converting an object into a byte stream, which can be stored in a file or transmitted over a network. 
The pickle module provides functions for both serialization (pickle.dump(), pickle.dumps()) and deserialization (pickle.load(), pickle.loads()). These functions work with any Python object, including built-in types 

## Host Locally

Run the project on Localhost

To Start Server

```bash
  resume.py
```

Open a web browser
Go to the project links

To access the home page
```bash 
http://localhost:8502/
``` 




## Features

- Connected to Mysql Database




- Contact Us data save in Mysql Database 



- Fullscreen mode Desktop/Mobile Compatibility




- Cross platform


# Screenshots




![Screenshot from 2023-05-24 18-09-50](https://github.com/Mayur96k/CDAC-Project--Resume-screening-using-NLP-and-Machine-Learning-Algorithm/assets/114133429/8dc8922e-eb71-4547-bbe4-d580f961e226)


![Screenshot from 2023-05-24 19-14-44](https://github.com/Mayur96k/CDAC-Project--Resume-screening-using-NLP-and-Machine-Learning-Algorithm/assets/114133429/77c26d65-66b9-4e2c-8a81-2b9594cc87e1)


## About The Code

The code  provided is for Streamlit application for resume screening. It contains several functions for text preprocessing and classification using a trained model. Let's go through the code step by step:

1. Importing libraries: The code starts by importing necessary libraries such as numpy, pandas, re, nltk, contractions, pickle, and Streamlit.

2. Loading model and vectorizer: The code opens two pickle files, `project_model.pk` and `vectorizer_new.pk`, using the `open()` function and assigns them to the variables `model_file` and `cv_file`, respectively. Then, it loads the pickled objects into the variables `model` and `cv` using the `pk.load()` function. Finally, the file handles are closed.

3. Function definitions:
   - `decontract(resume)`: This function takes a resume as input, converts all characters to lowercase, splits the resume into words, applies contraction fixing using the `contractions.fix()` function from the `contractions` module, and returns the modified resume.
   - `preprocessing(resume)`: This function performs various preprocessing steps on the resume. It filters out links, user IDs (tags), and blank words. It also splits the resume into words, removes non-alphabetical characters from each word using the `rem_non_alpha_char()` function, filters out unnecessary spaces, and joins the words back into a single string. The resulting modified resume is returned.
   - `get_wordnet_pos(tag)`: This function maps part-of-speech (POS) tags to WordNet POS tags. It takes a tag as input and returns the corresponding WordNet POS tag.
   - `lemmatizer_fn(resume)`: This function lemmatizes the words in the resume. It uses POS tagging with `nltk.pos_tag()` to obtain POS tags for each word. Then, it applies lemmatization using `lm.lemmatize()` from `WordNetLemmatizer` and the `get_wordnet_pos(tag)` function to get the root words. The lemmatized words are joined back into a single string and returned.
   - `remove_stopwords(resume)`: This function removes stopwords from the resume. It takes a resume as input and filters out words that are in the `stp_words` set (containing stopwords). The filtered words are joined back into a single string and returned.
   - `cleaning(resume)`: This function applies a series of cleaning steps to the resume. It removes emojis using the `remove_emoji()` function (which is not provided in the code), applies decontraction, preprocessing, lemmatization, and stopwords removal. The cleaned resume is returned.

4. Setting up Streamlit application: The code uses Streamlit to create a simple user interface for the resume screening. It sets the title and creates an input box for the user to enter the resume data. It also creates a submit button. When the user submits the input, the code executes the following steps:
   - Calls the `cleaning()` function to clean the user input.
   - Splits the cleaned resume into lines.
   - Creates a pandas DataFrame (`input_df`) from the cleaned lines.
   - Uses the loaded vectorizer (`cv`) to transform the input DataFrame into a feature matrix (`x_inp`).
   - Defines a dictionary mapping the predicted class indices to their corresponding labels.
   - Uses the loaded model (`model`) to predict the class label for the transformed feature matrix.
   - Displays the predicted label in a text area as the model interpretation.

5. Streamlit interface: The code sets up the Streamlit interface by displaying the input box and submit button. When the user enters the resume data and clicks the submit button


## Authors

- Mayur Jagtap
- mayurjagtap96k@gmail.com

