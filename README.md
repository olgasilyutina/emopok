## Predicting emojiis here, folks

![](https://i.kym-cdn.com/photos/images/original/001/274/468/20b.gif)

So, how it all works:

* folder /data: the anonymized dataset used for analysis
* folder /getting_data_EDA: data loading, preparation and EDA 
* folder /word_embeddings: CBOW and Skipgram models, clustering, TSNE 
* folder /rf_classifier: word2vec, LDA, random forest
* sentiments were predicted here https://github.com/olgasilyutina/socialsent/tree/master

<<<<<<< HEAD
### Prediction example

Original text: Я в тюрьме всем нашим **ауе** здесь так плохо и одиноко лучше сюда непопадать здесь ломается моя жизнь 
Recommended emojis: 😭😔😒😪😢
Original emoji: 😪


Produced by 
Aina Nurmagombetova 🤙
Alina Cherepanova 🙋
Anya Bataeva 🤯
Olya Silyutina 🤔

=======
*e.g.*

**message**: Я в тюрьме всем нашим ауе здесь так плохо и одиноко лучше сюда непопадать здесь ломается моя жизнь https://t.co/3SETIXDVjq
>>>>>>> 72d4eeac10da1c7fdfbd7b0dd2c9140eba9ed191

**predictions**: 😭😔😒😪😢

**actual emoji**: 😪
