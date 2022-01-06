import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file = pd.read_csv('anime.csv')
file = file.reset_index()

features = ['Rating Score','Number Votes','Studios','Synopsis', 'Tags', 'Episodes']
def combined_features(row):
    return str(row["Rating Score"])+" "+ str(row["Number Votes"])+" "+ str(row["Studios"])+" "+ str(row["Synopsis"])+" "+ str(row["Tags"])+" "+ str(row["Episodes"])+" "
def get_title_from_index(index):
    return file[file["index"] == index]["Name"].values[0]
def get_index_from_title(title):
    return file[file["Name"] == title]["index"].values[0]

file["combined_feature"]=file.apply(combined_features,axis=1)

cv = CountVectorizer()
count_matrix=cv.fit_transform(file["combined_feature"])

st.title('Anime Recommender')
selected_anime = st.selectbox(
'Which anime did you like?',
(file['Name'].values))

if st.button('Recommend'):
    with st.spinner(text='In progress'):
        cosine_sim = cosine_similarity(count_matrix)
        liked_movie_index = cosine_sim[get_index_from_title(selected_anime)]
        similar_anime = list(enumerate(liked_movie_index))
# similar_anime_sorted = sorted(similar_anime)
        similar_anime.sort(key = lambda row: row[1],reverse=True)
# print(similar_anime)
    st.success('Done')
    for i in range(10):
        st.write(get_title_from_index(similar_anime[i][0]))
