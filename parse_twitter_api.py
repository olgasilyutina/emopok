import tweepy
import tqdm
import emoji
import re
import pandas as pd

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

emojis = '\U0001F601,\U0001F602,\U0001F603,\U0001F604,\U0001F605,\U0001F606,\U0001F609,\U0001F60A,\U0001F60B,\U0001F60C,\U0001F60D,\U0001F60F,\U0001F612,\U0001F613,\U0001F614,\U0001F616,\U0001F618,\U0001F61A,\U0001F61C,\U0001F61D,\U0001F61E,\U0001F620,\U0001F621,\U0001F622,\U0001F623,\U0001F624,\U0001F625,\U0001F628,\U0001F629,\U0001F62A,\U0001F62B,\U0001F62D,\U0001F630,\U0001F631,\U0001F632,\U0001F633,\U0001F635,\U0001F637,\U0001F638,\U0001F639,\U0001F63A,\U0001F63B,\U0001F63C,\U0001F63D,\U0001F63E,\U0001F63F,\U0001F640,\U0001F645,\U0001F646,\U0001F647,\U0001F648,\U0001F649,\U0001F64A,\U0001F64B,\U0001F64C,\U0001F64D,\U0001F64E,\U0001F64F'.split(",")

texts = []
names = []
time = []

for emoji in emojis:
    print(emoji)
    for status in tqdm.tqdm(tweepy.Cursor(api.search, q=emoji, lang='ru', tweet_mode='extended', count=10000, wait_on_rate_limit=True ,wait_on_rate_limit_notify= True).items(1000)):
        texts.append(status.full_text)
        names.append(status.author.name)
        time.append(status.created_at)
        
df = pd.DataFrame({'texts': texts, 'names': names, 'time': time})
df.to_csv('emoji_full_text_twitter.csv', index=False)

emoji_col = []

df['texts'] = df['texts'].astype(str)
df['names'] = df['names'].astype(str)

emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())

r = re.compile('|'.join(re.escape(p) for p in emojis_list))
for i in range(len(df)):
    emoji_col.append(','.join(r.findall(str(df['texts'][i]))).split(","))

df['emoji'] = emoji_col

s = df.apply(lambda x: pd.Series(x['emoji']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'emoji'

df2 = df.drop('emoji', axis=1).join(s)
df2['emoji'] = pd.Series(df2['emoji'], dtype=object)
df2 = df2[df2['emoji'] != ""]

df2.to_csv('emoji_full_text_twitter.csv', index=False)

