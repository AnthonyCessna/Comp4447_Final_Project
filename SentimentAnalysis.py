import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def main():
    sid = SentimentIntensityAnalyzer()


if __name__ == '__main__':
    main()