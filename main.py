from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

rating_data = pd.read_csv('/content/drive/MyDrive/Recommendation_Models/data/Movie_Small/ratings.csv')
movie_data = pd.read_csv('/content/drive/MyDrive/Recommendation_Models/data/Movie_Small/movies.csv')

rating_data.head() #Check the data
movie_data.head() #Check the data
print(movie_data.shape)#Check the data
print(rating_data.shape)#Check the data

rating_data.drop('timestamp',axis=1,inplace=True) #Processing the rating_data
rating_data.head()

movie_data.drop('genres', axis = 1, inplace = True)#Processing the movie data
movie_data.head()

user_movie_data=pd.merge(rating_data,movie_data,on='movieId') # Merge the two data
user_movie_data.head()

user_movie_rating = user_movie_data.pivot_table('rating', index = 'userId', columns='title').fillna(0) # Edit data to use the SVD

user_movie_rating.shape #Check the data
user_movie_rating.head() #Check the data

movie_user_rating = user_movie_rating.values.T # Processing the data => T matrix
movie_user_rating.shape

type(movie_user_rating) #Check the data type

SVD = TruncatedSVD(n_components=12) #S Use SVD(Singular Value Decomposion), 특이값 분해 사용
matrix = SVD.fit_transform(movie_user_rating)
matrix.shape

matrix[0] # Reduce the dimension to 12 components and calculate the Pearson correlation coefficient using the data, -> 12개의 component로 차원을 축소, 데이터를 활용해서 피어슨 상관계수를 구한다.

corr = np.corrcoef(matrix) # Find the correlation coefficient -> 상관계수 구하기
corr.shape

corr2 = corr[:200, :200]
corr2.shape

plt.figure(figsize=(16, 10)) # Show the parts about correlation coefficient 상관계수 부분 보여주기.
sns.heatmap(corr2)

movie_title = user_movie_rating.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("300: Rise of an Empire (2014)") # Give the input to this module(the name of movie)

corr_coffey_hands  = corr[coffey_hands]
list(movie_title[(corr_coffey_hands >= 0.9)])[:50]
