import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions
import pickle as pk
import streamlit as st
from nltk.stem.snowball import SnowballStemmer

model_file = open('/home/hp/Desktop/Job Appln/NLP/project_model.pk','rb')
cv_file = open('/home/hp/Desktop/Job Appln/NLP/vectorizer_new.pk','rb')
model = pk.load(model_file)
cv = pk.load(cv_file)
model_file.close()
cv_file.close()


def decontract(resume):
    # Making all characters lower case for uniformity
    resume = resume.lower()

    words = resume.split()
    final_resume = ""
    for word in words:
        final_resume = final_resume + contractions.fix(word) + " "

    return final_resume
    
def preprocessing(resume):
    # Crearing a list of each word in the resume
	words = resume.split()

    # filtering out links
	words = list(
        map(lambda text: text if text.find('http' or 'www') == -1 else text[:(text.find('http' or 'www'))], words))
	words = list(filter(lambda x: len(x) > 0, words))  
    # to remove blank words

    #     print(words)

    # filtering out userid's (tags)
	words = list(filter(lambda x: x[0] != '@', words))

    # Creating a string using all the words in the list
	words = " ".join(words)

    # Splitting again using '.' (This and above 1 step is to get words separated by '.' separately and not one, hi..am..not)
	words = words.split('.')

    #   extra efforts for below rem_non_alpha_char function
	resume = " ".join(words)
	words = resume.split()

	words = rem_non_alpha_char(words)

    #     # Filtering out un-necessary spaces in the words
	words = list(filter(lambda x: x != '', words))
    #
    # sb = SnowballStemmer(language='english')
    # words = list(map(lambda x: sb.stem(x),words))

    # Creating a string using all the words in the list
	mod_resume = " ".join(words)

	return mod_resume

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ  # 'a'
    elif tag.startswith('V'):
        return wordnet.VERB  # 'v'
    elif tag.startswith('N'):
        return wordnet.NOUN  # 'n'
    elif tag.startswith('R'):
        return wordnet.ADV  # 'r'
    else:
        return wordnet.NOUN

lm = WordNetLemmatizer()
def lemmatizer_fn(resume):
    # takes list of words and return list of tuples where [('word',pos_tag)]
    word_pos_tags = nltk.pos_tag(resume.split())

    # lemmaatizing the each word: lematizer takes word and wordnet_tag and returns root word
    words = [lm.lemmatize(idx, get_wordnet_pos(tag)) for idx, tag in word_pos_tags]
    mod_resume = " ".join(words)
    return mod_resume


stp_words = set(stopwords.words('english'))
stp_words.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm',
                  'im', 'll', 'y', 've', 'u', 'ur', 'don',
                  'p', 't', 's', 'aren', 'kp', 'o', 'kat',
                  'de', 're', 'amp', 'will', 'wa', 'e', 'like', 'ok', 'lot', 'go', 'oh', 'no', 'ti', 'back', 'not', 'a',
                  'i', 'the'])
stp_words.remove('nor')
# stp_words.remove('not')
stp_words.remove('no')


def remove_stopwords(resume):
    words = [word for word in tweet.split() if word not in stp_words]
    mod_resume = " ".join(words)
    return mod_resume


def cleaning(resume):
    # tweet = remove_emoji(tweet)
    resume = decontract(resume)
    resume = preprocessing(resume)
    # tweet = lemmatizer_fn(resume)
    mod_resume = remove_stopwords(resume)
    return mod_resume


st.title(':face_with_symbols_on_mouth: Resume Screening :face_with_symbols_on_mouth:')

# Create an input box for the user to enter text
st.write(" ")
user_input = st.text_input("Enter your data here")

# Create a submit button for the user to submit their input
submit_button = st.button("Submit")

# If the user clicks the submit button or press enter, display their input
if submit_button and user_input:
#    st.write("You entered:", user_input)

    clean_cmnt = cleaning(user_input)
    inp = clean_cmnt.split('\n')
    input_df = pd.DataFrame(inp)

#    st.write("You entered:", inp)
#    st.write("You entered:", cv)
#    st.write("You entered:", input_df)
    x_inp = cv.transform(input_df.iloc[:,0])

    dict = { 0: 'Java Developer',1: 'Testing', 2: 'DevOps Engineer', 3:'Python Developer', 4: 'Web Designing',5:'HR',6:'Hadoop',7:'Blockchain',8:'ETL Developer',9:'Operations Manager',10:'Data Science',11:'Sales',12:'Mechanical,Engineer',13:'Arts',14:'Database',15:'Electrical Engineering',16:'Health and fitness',17:'PMO',18:'Business Analyst',19:'DotNet Developer',20:'Automation Testing',
21:'Network Security Engineer',22:'SAP Developer',23:'Civil Engineer',24:'Advocate'}
    prdiction = model.predict(x_inp)
    st.text_area("Model Interpretation", value=f"Given CV match for {dict.get(prediction[0])} profile")

    st.write(f"{'-'*120} Thank You {'-'*120}")
