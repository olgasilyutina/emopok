
<h1><center>emopok xgboost 🌳 </center></h1>
<center>authors: [Aina Nurmagombetova](https://github.com/anurma) 🤙 [Alina Cherepanova](https://github.com/alinacherepanova) 🙋 [Anya Bataeva](https://github.com/fyzbt) 🤯 [Olya Silyutina](https://github.com/olgasilyutina) 🤩</center>


```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
```


```python
# prepare data
unique_df = pd.read_csv('./data/unique_emopok.csv')
textfeatures_df = pd.read_csv('./data/textfeatures_emopok.csv')
sent_df = pd.read_csv('./data/sentiments_emopok.csv')
topics_df = pd.read_csv('./data/dum_topics_emopok.csv')
emoji_df = pd.read_csv('./data/emoji_texts_df.csv')
emo_clusters = pd.read_csv('./data/emopok_clusters.csv')
d2v_vectors_df = pd.read_csv('./data/d2v_vectors_emopok.csv')
```


```python
sent_df.columns = ['index', 'sent', 'texts']
xgb_df = unique_df.merge(textfeatures_df.drop(['text'], axis=1), on = 'index')
xgb_df = xgb_df.merge(sent_df.drop('texts', axis = 1), on = 'index')
xgb_df = pd.concat([xgb_df.reset_index(drop=True), d2v_vectors_df], axis=1)
xgb_df = pd.concat([xgb_df.reset_index(drop=True), topics_df], axis=1)
emoji_df = emoji_df[['emoji', 'index']].drop_duplicates()
xgb_df = xgb_df.merge(emoji_df, on = 'index')
emo_clusters.columns = ['emoji', 'cluster_group']
xgb_df = xgb_df.merge(emo_clusters, on = 'emoji')
xgb_df = xgb_df.drop(['emoji', 'texts'], axis = 1)
```


```python
len(xgb_df)
```




    547642




```python
xgb_df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>n_chars</th>
      <th>n_commas</th>
      <th>n_digits</th>
      <th>n_exclaims</th>
      <th>n_hashtags</th>
      <th>n_lowers</th>
      <th>n_mentions</th>
      <th>n_urls</th>
      <th>n_words</th>
      <th>...</th>
      <th>topic_11</th>
      <th>topic_12</th>
      <th>topic_13</th>
      <th>topic_14</th>
      <th>topic_15</th>
      <th>topic_16</th>
      <th>topic_17</th>
      <th>topic_18</th>
      <th>topic_19</th>
      <th>cluster_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>755</td>
      <td>3</td>
      <td>123</td>
      <td>0</td>
      <td>1</td>
      <td>301</td>
      <td>0</td>
      <td>3</td>
      <td>666</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 134 columns</p>
</div>




```python
xgb_df.to_csv('./data/xgb_emopok.csv', index = False)
```


```python
xgb_df = pd.read_csv('./data/xgb_emopok.csv')
```


```python
# cols = [c for c in xgb_df.columns if 'topic' in c.lower()]
# xgb_df = xgb_df.drop(cols, axis = 1)
```


```python
for i in tqdm(['16_3', '6', '3', '1', '23']):
    xgb_df_sample = xgb_df[xgb_df['cluster_group'] == i].sample(40000)
    xgb_df = xgb_df[xgb_df['cluster_group'] != i]
    xgb_df = pd.concat([xgb_df, xgb_df_sample])
```


    HBox(children=(IntProgress(value=0, max=5), HTML(value='')))


    



```python
xgb_df = xgb_df.reset_index(drop=True)
```


```python
# top clusters
xgb_df.groupby('cluster_group').index.count().sort_values('index', ascending = False).reset_index().head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_group</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16_3</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>21</td>
      <td>29775</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>28627</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13</td>
      <td>16723</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16_7</td>
      <td>16192</td>
    </tr>
    <tr>
      <th>9</th>
      <td>18</td>
      <td>15311</td>
    </tr>
  </tbody>
</table>
</div>




```python
Y = xgb_df[['cluster_group']]
X = xgb_df.drop(['cluster_group', 'index'], axis = 1)
```


```python
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```


```python
model = XGBClassifier(n_jobs=4, silent=False, objective='multi:softprob', n_estimators=1)
```


```python
eval_set = [(X_test, y_test.values.ravel())]
model.fit(X_train, y_train.values.ravel(), eval_set=eval_set, verbose=True)
```

    [0]	validation_0-merror:0.866463





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=1,
           n_jobs=4, nthread=None, objective='multi:softprob', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=False, subsample=1)




```python
sorted_idx = np.argsort(model.feature_importances_)[::-1]
cols = []
imp = []
for index in sorted_idx:
    cols.append(X_train.columns[index])
    imp.append(model.feature_importances_[index])
```


```python
feature_importances = pd.DataFrame({'features': cols, 'importances': imp})
```


```python
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
feature_importances = feature_importances[feature_importances['importances'] > 0.01]
feature_importances = feature_importances.sort_values('importances')
```


```python
trace = go.Bar(
    y = list(feature_importances['features']),
    x = list(feature_importances['importances']),
    marker=dict(color='#3AA2FB'),
    orientation = 'h')

layout = dict(title = '',
              width=400,
              height=400,
              xaxis = dict(title = ''),
              yaxis = dict(title = ''))

fig = dict(data=[trace])
iplot(fig)
```


<div id="de47f273-6636-4cae-ad70-a7628e8959df" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("de47f273-6636-4cae-ad70-a7628e8959df", [{"marker": {"color": "#3AA2FB"}, "orientation": "h", "x": [0.011170841753482819, 0.012584937736392021, 0.013353443704545498, 0.016062840819358826, 0.016130337491631508, 0.016511406749486923, 0.01818370260298252, 0.019037039950489998, 0.019051596522331238, 0.019591130316257477, 0.02237125113606453, 0.023974813520908356, 0.0321112722158432, 0.03524245321750641, 0.04295428469777107, 0.04296538233757019, 0.09344596415758133, 0.40817898511886597], "y": ["n_hashtags", "n_uppers", "d2v_52", "n_mentions", "n_exclaims", "d2v_37", "d2v_70", "d2v_59", "d2v_9", "d2v_81", "n_nonasciis", "n_commas", "n_digits", "sent", "d2v_87", "n_words", "topic_12", "topic_11"], "type": "bar", "uid": "949c27cc-70a0-11e9-a137-a45e60d44769"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
preds = model.predict(X_test)
```

    /Users/o.silutina/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:



```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds)
```


```python
X_test['predictions'] = preds
test_indexes = xgb_df.iloc[X_test.index]
test_indexes = test_indexes[['index']].reset_index()
test_predictions = X_test.reset_index()[['index', 'predictions']]
test_predictions.columns = ['level_0', 'predictions']
test_indexes = test_indexes.merge(test_predictions, on = 'level_0')
fin_test = unique_df.merge(test_indexes, on = 'index').sort_values('level_0')\
    .merge(emoji_df, on = 'index', how = 'inner')\
    .merge(emo_clusters, on = 'emoji', how = 'inner')
```


```python
fin_test = fin_test.drop(['emoji', 'level_0'], axis = 1).drop_duplicates()
```


```python
fin_test = fin_test.assign(NE=fin_test.predictions.astype(str) == fin_test.cluster_group)
```


```python
fin_test.to_csv('./data/success_pred.csv', index = False)
```


```python
good_ones = fin_test[(fin_test['NE'] == True)]
```


```python
# fin_test[(fin_test['NE'] == True) & (~fin_test['predictions'].isin(['16_3', '14', '3', '1', '12', '16_1', '15_7',\
#                                                                    '15_0', '2', '21', '13', '6', '23', '24', '5',\
#                                                                    '20']))]
```


```python
for i in [249744, 70842, 323158, 323158, 296003, 182053, 104306, 216050, 347738, 39268, 208593, 249754, 153240,\
         337200, 73613, 136165, 272755, 10123, 327009, 155539, 297794, 281368, 150979, 43400, 210014, 20756, 158854,\
         153096, 332217]:
    prediction_ind = str(fin_test[fin_test['index'] == i]['predictions'].tolist()[0])
    prediction = ' '.join(set(emo_clusters[(emo_clusters['cluster_group'] == prediction_ind) & \
                                      (~emo_clusters['emoji'].isin(['🐀', '🐸', '⚜']))]\
                          .sample(5, replace=True)['emoji'].tolist()))
    print('Сообщение: ' + str(fin_test[fin_test['index'] == i]['texts'].tolist()[0]) + \
          'Предсказание: ' + prediction
           + '\n')
    
```

    Сообщение: Ну наконец-то эти пары закончились, как же я отвык за несколько месяцев 😓Предсказание: 👎 🔞 😓 🎆 👏
    
    Сообщение: @lovvely_1d люди столько не живут😓Предсказание: 🖕 😤 👊 🔞 👏
    
    Сообщение: Что делать если у тебя есть краш, но ты очень сильно боишься признаться ему🙁😓😓😓Предсказание: 🇷🇺 😥 ♥ 🔞 😓
    
    Сообщение: Что делать если у тебя есть краш, но ты очень сильно боишься признаться ему🙁😓😓😓Предсказание: 💔 👊 🌊 😫
    
    Сообщение: Судьба уберегла 😫😫😫🤐🤐🤐Предсказание: 😐 😟 💤 🤥 🙁
    
    Сообщение: Да не ниче🤐Предсказание: 🤪 🧐 💤 🤥 🙁
    
    Сообщение: @SirPhilipp Да блин не поймите неправильно😨Я вот человек взрослый мне 15(через 2 года)а вот моя мать это не понимает!!Я как истинная бабочка🦋покидаю свой кокон,-800 гр за месяц на голоде(параметры 120/80)и мама говорит что у меня анорексия,надо сделать вид при ней что кушаю чтобы отстала🙏Предсказание: 🎶 🌹 🎼 🎧 💐
    
    Сообщение: Конец рабочего дня 💫💫💫💜💜💜 Поскорее бы закончился этот день 🙌🙌🙌 Жду завтрашний день  ммммм бабочки 🦋 в животе #curiousaboutARMY https://t.co/M4wCyyqEOlПредсказание: 🌹 🥀 🌲 🎧 💐
    
    Сообщение: RT @Booissaa: Заботливый бу, подсылает своих агентов даже, когда играет с семьей😘🐾 https://t.co/3rEKfB3LK5Предсказание: 😘 😍
    
    Сообщение: @elleys24 Почему это так мило? 😍Какие бейбики 🐣💞Предсказание: 🐼 🙉 🙈 🐕 🐥
    
    Сообщение: Как же давно я не болела! Простыла. 🤒 Бля как плохо-то. Отвыкла 😫Предсказание: 😦 🤮 😧 😠 😡
    
    Сообщение: Ну например, глянь на девушку с ником &quot;ня🐣&quot;Предсказание: 🐇 🐶 🐥 😸
    
    Сообщение: ААААААААА!!!!!! БРАТА В ПЯТЫЙ КЛАСС ПРИНЯЛИИИИ😆😆😆😆😍🤢😂
    
    УЧИТЕЛЕМ БУДЕТ ДМИТРЙИ ЕГОРВТЧИК ДАДА ТОТ САМЫЙ УРОД🤢🤢🤢
    
    хотя зато мы вместе с братом теперь всю школу разнесем, ииндииндааааа😂😂😂😂😎Предсказание: 😡 😵 😷 😧
    
    Сообщение: Я ОЧЕНЬ ЗАКОМПЛЕКСОВАННЫЙ  ЧЕЛОВЕК!Однако,все люди для меня по-своему красивы,но потом я иду к зеркалу и...О БОЖЕ МОЙ 😱 Передо мной стоит самый настоящий УРОД 😫😩😭🤢🤮Я выгляжу,как маленький,жирный,усатый испанец😭💥🔫Предсказание: 🤢 🤕 🤧 🤬
    
    Сообщение: @Maria__Way Гордая фея без одежды 🙈🙊🧚‍♀️Предсказание: 👮 💆 🙇
    
    Сообщение: 📢🆘Наступает лихорадочная пора🔥и речь не о наступающих праздниках, а о сессии!✍🏻🙇‍👩‍🎓👨‍🎓🆘
    ⏳📚📐Часики тикают. Самое время уточнить график сессии.
    🎓По ссылке график учебного процесса и ☝🏻сроки сессий по группам
    ✍🏻✍🏻✍🏻👇🏻👇🏻👇🏻#tltTGu #flagshipuniversity
    https://t.co/jwm7020uaR https://t.co/T98AGINudCПредсказание: 👆 👇
    
    Сообщение: Появилась новая любимая песня(альбом)🎧  Все больше и больше влюбляют в себя 🙈Предсказание: 🦋 🙏 🎼 💐
    
    Сообщение: @a_life_is_game Ну Тоха...уверена, у Чимы всё будет хорошо❤💜
    (автор ты супер😘🌸)Предсказание: 😘 😍
    
    Сообщение: Эмоджи - это воистину потрясающее изобретение человечества. С их помощью можно обозначить и выразить буквально ВСЁ! 
    Сегодня ты приуныл - Используй 😿🙁! 
    Чувствуешь себя пёселем - 🐕🐶! 
    Радуешься приходу весны - Так много вариантов 🦋🌿🌺☀! 
    Задумал убить кого-то - No prob 🙂.Предсказание: 🤩 😎 😊 🙂 😙
    
    Сообщение: Арабский бот Денис, это странно🤷‍♂Предсказание: ♀ ♂
    
    Сообщение: так ещё и ночью 🧙‍♂️Предсказание: ♀ ♂
    
    Сообщение: решил похмельно обсудить с коллегами Ханну Монтану, но в очередной раз столкнулся со стеной недопонимания💅🏻 время рассказать всем, что во вторник последний мой рабочий день на архитектурном Титанике🖤😉⭐️😂🌠🔥😌👍🏻💸💃😳💦🦉😡👌🏻😭💪🏻💥🍟📹🏢👸🏼💙 vsem pisПредсказание: 😄 🤣 😂 😁
    
    Сообщение: А хулек 🤷‍♂Предсказание: 🤦 🤷 🙅 🙍
    
    Сообщение: @free_fox_ Бля я по-русски путаюсь то 🙈🤦‍♀️Предсказание: ♀ ♂
    
    Сообщение: Как обычно🤷‍♀️ Ничего не меняетсяПредсказание: 🙎 🤷 🙅 🙍
    
    Сообщение: @bantanyana на пять, адекватный фанат🙋‍♀️Предсказание: 👮 🙇 👯
    
    Сообщение: Блен, Саня, ты всё испортил🤦‍♀️Предсказание: ♀ ♂
    
    Сообщение: Аааааа!!!ИГРА ПРЕСТОЛОВ ваще 🔥🔥🔥 😱😭😱😭😱😭Предсказание: 😱 😰 😨
    
    Сообщение: я в тюрьме всем ауе нашим ауе👋 здесь очень плохо ребята😪 и просто невыносимо лучше сюда непопадать здесь ломается моя жизнь поговорите со мной мне так плохо😪 https://t.co/3vJYIjHEUmПредсказание: 👊 😤 🤦‍♂️
    

