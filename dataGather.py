import RedditAuthentication
import requests
import pandas as pd

"""
Turns the reponse from the Reddit API into a pandas dataframe

Parameters
----------
res : reponse from a GET request to the reddit API
"""
def redditResponse(res):
    data = pd.DataFrame()
     # adds all 100 posts to a dataframe
    for post in res.json()['data']['children']:
        data = data.append(
            {
                'title':post['data']['title'], 
                'selftext': post['data']['selftext'],
                'created_utc' : post['data']['created_utc'],
                'num_comments': post['data']['num_comments'],
                'upvote_ratio': post['data']['upvote_ratio'],
                'ups': post['data']['ups'],
                'downs': post['data']['downs'],
                'id': post['data']['id'],
                'kind': post['kind']
            },
            ignore_index=True
        )

    return data

def main():

    data = pd.DataFrame()
    params = {'limit': 100}
    redditConnection = RedditAuthentication.RedditAuthentication()

    # requests 100 posts many times
    for i in range(10):
        res = requests.get("https://oauth.reddit.com/r/depression/new",
                   headers=redditConnection.headers,
                   params=params)

        new_df = redditResponse(res)
        row = new_df.iloc[len(new_df)-1]
        fullname = row['kind'] + '_' + row['id']
        params['after'] = fullname
        data = data.append(new_df, ignore_index=True)

    # Saves data frame to csv file
    data.to_csv("RedditData.csv")


if __name__ == '__main__':
    main()