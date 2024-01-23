import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from concurrent.futures import ThreadPoolExecutor
from streamlit_lottie import st_lottie
import json

global_image = "image/movie-poster.jpg"
api_key = "8265bd1679663a7ea12ac168da84d2e8"

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


st.set_page_config(layout='wide')
def fetch_poster_and_name(type, title):
    if type == "Series":
        type = "tv"
    else:
        type = "movie"
    url = f'https://api.themoviedb.org/3/search/{type}?api_key={api_key}&language=en-US&query={title}'
    data = requests.get(url).json()
    try:
        poster_path = data['results'][0]['poster_path']
        # url = f'https://api.themoviedb.org/3/{type}/{id}?api_key={api_key}&language=en-US'
        # data = requests.get(url).json()
        # poster_path = data['poster_path']
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}", title
        else:
            return global_image, title
    except IndexError:
        return global_image, title

def recommend(media, type):
    with ThreadPoolExecutor() as executor:
        movies = data[data['title'] == media].index[0]
        distance = sorted(list(enumerate(similarity[movies])), reverse=True, key=lambda x: x[1])
        futures = [executor.submit(fetch_poster_and_name, type, data.iloc[i[0]].title) for i in distance[1:26]]
        movie_posters, movie_names = zip(*[future.result() for future in futures])
        return movie_names, movie_posters

@st.cache_data
def load_data(fpath):
    data = pd.read_csv(fpath)
    data = data.replace(np.nan, '')
    df = data[['show_id', 'type', 'title', 'director', 'cast', 'country', 'rating', 'description']]
    return df

data = load_data('netflix_titles.csv')

tfidf = TfidfVectorizer(strip_accents='ascii', analyzer='word', stop_words='english', max_features=15000)
vectorizer = tfidf.fit_transform(data['description'])
similarity = cosine_similarity(vectorizer)

lottie_coding = load_lottiefile("netflix-logo.json")
st.markdown(
    f'<div style="position: fixed; top: 10px; right: 10px;">'
    f'{st_lottie(lottie_coding, speed=1, reverse=False, loop=False, quality="high", height=220)}'
    f'</div>',
    unsafe_allow_html=True
)

st.title('Netflix Recommender System')
type = st.radio(
    'What you want to watch?',
    ('Movie', 'Series')
)

if type == 'Movie':
    options = st.selectbox(
        'Type movie name...',
        data[data['type'] == 'Movie']['title'].tolist()
    )
else:
    options = st.selectbox(
        'Type series name...',
        data[data['type'] == 'TV Show']['title'].tolist()
    )

recommended_movie_names, recommended_movie_posters = recommend(options, type)
st.write(f"If you're watching {options} then...")
st.subheader('You should also watch')

num_rows = 5
num_cols = 5

with st.container():
    for row in range(num_rows):
        col1, col2, col3, col4, col5 = st.columns(num_cols)
        for col, (name, poster) in zip([col1, col2, col3, col4, col5], zip(recommended_movie_names[row * num_cols:(row + 1) * num_cols], recommended_movie_posters[row * num_cols:(row + 1) * num_cols])):
            if poster != global_image:
                col.image(poster, caption=name, width=220)
            else:
                col.image(global_image, caption=name, width=220)
