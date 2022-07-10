import nltk
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def main():
    sid = SentimentIntensityAnalyzer()

    posts_df = pd.read_csv("RedditDataCleaned.csv")

    posts_df['scores'] = posts_df['cleaned_text'].apply(lambda x: sid.polarity_scores(x))

    posts_df.to_csv("RedditDataCleaned.csv")

if __name__ == '__main__':
    main()