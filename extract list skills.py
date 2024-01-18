import pandas as pd

df=pd.read_csv(r"C:\Users\alan7\Downloads\jobs.csv")

list_skills=list(df["Key Skills"])

final_list_skills=[]

for skills in list_skills:
    array_skill=skills.split("|")
    for sub_skill in array_skill:
        if " " in sub_skill:
            sub_skill=sub_skill.lstrip()
        final_list_skills.append(sub_skill)

final_list_skills=set(final_list_skills)
final_list_skills=list(final_list_skills)
pd.DataFrame(final_list_skills).to_csv(r"C:\Users\alan7\Downloads\jobs_new_skills.csv",index=False)
#print(final_list_skills)