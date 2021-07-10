from flask import Flask, render_template,request
#from static.recommenders import recommend_movie

import pandas as pd
import sklearn
import numpy as np
data=pd.read_csv(r'F:\vscode\GOEDUHUB_\Task_17_Movi_Rec_sys\data_content.csv')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vect=CountVectorizer()
vectors=vect.fit_transform(data['director_genre_actors'])
similarity=cosine_similarity(vectors)
def recommend_movie(movie):
    if movie not in data['movie_title'].unique():
        return []
    else:
        i=data.loc[data['movie_title']==movie].index[0]
        lst=list(enumerate(similarity[i]))
        lst=sorted(lst,key=lambda x:x[1],reverse=True)
        lst=lst[1:11]
        l=[]
    
        for i in range(len(lst)):
            a=lst[i][0]
            l.append(data['movie_title'][a])
        return l


app= Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('webcode.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    movie_name=request.form.get('Movie Name')
    pred=recommend_movie(movie=movie_name)
    return render_template('webcode.html',dat=pred,len=len(pred))

if __name__=='__main__':
    app.run(debug=True)