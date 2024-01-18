## Alan DROMAS, Antoine FERREIRA, Emanuel EDMOND


# How to use:
Install the requirements using "pip install -r requirements.txt"

Run the command "streamlit run __init__.py" in a command window to run the GUI

Upload a pdf file of your resume and it will provide the most relevant jobs

![Percentage Precision](https://raw.githubusercontent.com/DROMASAlan/Job-Recommendation/main/Capture%20d'%C3%A9cran%202024-01-18%20164717.png)

# Methodology algorithm and result:

We created a list of key skills extracted from the list of jobs dataset which is from: https://statso.io/jobs-dataset/.

Then we search of the keyword skills in the uploaded resume.

Then we vectorize the key skill in the resume and the key skills from all the job list and we find the job that match the most with the resume.

The 5 jobs that have the best match are displayed in the GUI.

We could improve the dataset with more skills or more specifc resume.

# Difficulty:

We had difficulty to evaluate the system.

We gathered 38k resume with job title from the following website: https://huggingface.co/datasets/InferencePrince555/Resume-Dataset/tree/main

We only used 1800 resume to backtest the algorithm because it takes 2 sec to find a job corresponding to a resume

We only check if the returned job title have corresponding word with the resume dataset job title, we obtain a success rate of 13% but it is not the most optimal way to check if a recommended job corresponds to a resume.

There might be some resume in some field that aren't in the job list database so it can't match them properly.

![Percentage Precision](https://raw.githubusercontent.com/DROMASAlan/Job-Recommendation/main/Capture%20d'%C3%A9cran%202024-01-18%20153041.png)


check_cv_percentage.py: check % of matching resume with job offer

Skill_extraction.py: get the skills from the resume

extract list skills.py: get list of skills from list of jobs

__init__.py: GUI App (streamlit)

job_recommender.py: Backend to match resume with job offer

jd_structured_data2: list of job offer from https://statso.io/jobs-dataset/
skills2.csv: list of skills from the jobs offer

