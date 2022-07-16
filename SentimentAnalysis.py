import nltk
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

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

    posts_df['comp_score'] = posts_df['compound'].apply(lambda x: sentiment(x))

    posts_df.to_csv("RedditDataCleaned.csv")

    plt.figure(figsize = (20,8))

    ax = sns.violinplot(x="time", y="compound", data=posts_df,
                        inner=None, color=".8")

    ax = sns.pointplot(x="time", y="compound", data=posts_df, ci = None)

    ax = sns.stripplot(x="time", y="compound", data=posts_df, order = 
                    ["06", "07", "08", "09","10","11", "12", "13", "14", "15", "16", "17", "18", "19", 
                        "20", "21", "22", "23", "00", "01", "02", "03", "04", "05"], jitter = True)



if __name__ == '__main__':
    main()