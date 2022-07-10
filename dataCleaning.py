import pandas as pd
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

def main():
    posts_df = pd.read_csv("RedditData.csv")

    # Keeps only ascii characters
    def clean_txt(row):
        return row.encode('ascii', 'ignore').decode('ascii')

    posts_df["cleaned_text"] = posts_df["selftext"].apply(lambda x: clean_txt(x))

    # Cleans text to be used in word cloud
    def wordcloud_clean(row):
        row = contractions.fix(row)
        
        tokens = word_tokenize(row)
        tokens = [w.lower() for w in tokens]
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        
        mystring = ' '.join(map(str, words))
        
        return mystring
    
    posts_df["wordcloud_txt"] = posts_df["cleaned_text"].apply(lambda x: wordcloud_clean(x))

    text = ""

    for index, row in posts_df.iterrows():
        text += row['wordcloud_txt']
        
    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("mygraph.png")

    # next I want to convert the time since UTC to us central time for analysis

    def convert_time(row):
        time = datetime.datetime.fromtimestamp(row)
        time = time.astimezone(pytz.timezone('US/Central')).strftime('%Y-%m-%d %H:%M:%S %Z%z')
        return time

    posts_df["time"] = posts_df["created_utc"].apply(lambda x: convert_time(x))

    posts_df.to_csv("RedditDataCleaned.csv")


if __name__ == '__main__':
    main()