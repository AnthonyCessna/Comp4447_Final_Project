import pandas as pd
import seaborn as sns
import numpy as np
import nltk
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
import pytz
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def main():
    sid = SentimentIntensityAnalyzer()

    posts_df = pd.read_csv("RedditDataCleaned.csv")

    posts_df['scores'] = posts_df['cleaned_text'].apply(lambda x: sid.polarity_scores(x))

    posts_df['compound']  = posts_df['scores'].apply(lambda score_dict: score_dict['compound'])

    posts_df['neg']  = posts_df['scores'].apply(lambda score_dict: score_dict['neg'])

    posts_df['neu']  = posts_df['scores'].apply(lambda score_dict: score_dict['neu'])

    posts_df['pos']  = posts_df['scores'].apply(lambda score_dict: score_dict['pos'])

    def sentiment(row):
        if row >=.2 and row < .8:
            return 'pos'
        elif row >= .8:
            return 'strong_pos'
        elif row <= -.2 and row > -.8:
            return 'neg'
        elif row <= -.8:
            return 'strong_neg'
        else:
            return 'neu'

    def section_day(row):
        if row >= 6 and row <= 11:
            return "Morning"
        elif row >= 12 and row <= 17:
            return "Afternoon"
        elif row >= 18 and row <= 23:
            return "Evening"
        else:
            return "Night"

    posts_df['comp_score'] = posts_df['compound'].apply(lambda x: sentiment(x))
    posts_df['Section_of_Day'] = posts_df['time'].apply(lambda x: section_day(x))
    posts_df.to_csv("RedditDataCleaned.csv")

    plt.figure(figsize = (20,8))

    ax = sns.violinplot(x="time", y="compound", data=posts_df,
                        inner=None, color=".8")

    ax = sns.pointplot(x="time", y="compound", data=posts_df, ci = None, estimator=np.median)

    ax = sns.stripplot(x="time", y="compound", data=posts_df, jitter = True)

    ax.set( xlabel = "Hour", ylabel = "Sentiment Score", title='Post sentiment over the course of a day')


if __name__ == '__main__':
    main()