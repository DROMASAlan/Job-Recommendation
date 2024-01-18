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

# Function to process the resume and recommend jobs
def process_resume(file_path):
    # Extract text from PDF resume
    resume_skills=skills_extraction.skills_extractor(file_path)

    # Perform job recommendation based on parsed resume data
    skills=[]
    skills.append(' '.join(word for word in resume_skills))
    
    st.write(skills)
    
    
    # Feature Engineering:
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

# Streamlit app
def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

    if uploaded_file is not None:
        # Process resume and recommend jobs
        file_path=uploaded_file.name

        df_jobs = process_resume(file_path)

        # Display recommended jobs as DataFrame
        st.write("Recommended Jobs:")
        st.dataframe(df_jobs[["Job Title","Job Experience Required","Role Category","Functional Area","Industry"]])

# Run the Streamlit app
if __name__ == '__main__':
    main()