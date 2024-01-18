import streamlit as st
import pandas as pd
import PyPDF2
from pyresparser import ResumeParser
from sklearn.neighbors import NearestNeighbors
import skills_extraction as skills_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))
from pyresparser import ResumeParser
import os
from docx import Document
import skills_extraction as skills_extraction
import csv
import math

#https://huggingface.co/datasets/InferencePrince555/Resume-Dataset/tree/main
df_resume=pd.read_csv(r"C:\Users\alan7\Documents\A5\Development Methods of Applied Intelligent Software Systems\Job Recommendation\updated_data_final_cleaned.csv",sep=";", on_bad_lines="skip")
jd_df=pd.read_csv(r'C:\Users\alan7\Documents\A5\Development Methods of Applied Intelligent Software Systems\Job Recommendation\jd_structured_data2.csv')


def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def test_cv(resume_text):
    resume_skills=list(skills_extraction.extract_skills(resume_text))

    skills=[]
    skills.append(' '.join(word for word in resume_skills))
    if skills==['']:
        return
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)


    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = (jd_df['Key Skills'].values.astype('U'))


    def getNearestN(query):
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances, indices

    distances, indices = getNearestN(jd_test)
    test = list(jd_test) 
    matches = []

    for i,j in enumerate(indices):
        dist=round(distances[i][0],2)
        temp = [dist]
        matches.append(temp)

    matches = pd.DataFrame(matches, columns=['Match confidence'])

    # Following recommends Top 5 Jobs based on candidate resume:
    jd_df['match']=matches['Match confidence']

    return jd_df.sort_values('match',ascending=True).head(5)

class Found(Exception): 
    pass

def compute_positive(df_resume):
    count_positive=0
    total_count=0
    
    list_category=list(df_resume["instruction"])
    list_resume=list(df_resume["Resume_test"].astype(str))

    for i in range(1800):
        try:
            resume_text=list_resume[i]
            category_text=list_category[i]
            result_df=test_cv(resume_text)
            job_list_result=list(result_df["Job Title"])

            concat_job_list_str=' '.join(job_list_result)
            for word in concat_job_list_str.split(" "):
                for actual_job_str in category_text.split(" "):
                    if word in actual_job_str:
                        raise Found
                        
            total_count=total_count+1
            #input()
        except Found:
            count_positive=count_positive+1
            total_count=total_count+1
        except:
            pass
    return count_positive,total_count
count_positive,total_count=compute_positive(df_resume)
print("count_positive: ",count_positive)
print("total_count: ",total_count)
print("percentage_positive: ",count_positive/total_count)