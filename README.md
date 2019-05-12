## Predicting emojis here, folks

![](https://i.kym-cdn.com/photos/images/original/001/274/468/20b.gif)

So, how it all works:

But first, pip install

```
!pip install -e git+https://github.com/olgasilyutina/emopok.git#egg=emopok
from emopok import emopok
emopok.textfeatures()
```

* data is [here](https://yadi.sk/d/FLR32upzfkfj6Q)
* models are [here](https://yadi.sk/d/QgbkYBHiwkB6-A)
* you can repeat data preparation process [here](https://github.com/olgasilyutina/emopok/blob/master/emopok_data_pipeline.ipynb)
* run xgboost [here](https://github.com/olgasilyutina/emopok/blob/master/emopok_xgboost.ipynb)
* sentiments were predicted [here](https://github.com/olgasilyutina/socialsent3/blob/master/example.ipynb)

### Project presentation

[DataFest 6](https://datafest.ru/) presentation video is [right here](https://youtu.be/tpuKgWVrbMU) ~ 4:22:15

PDF version of our presentation is [here](https://docviewer.yandex.ru/view/117475574/?*=oRKHkvpDrceUZJ9joodlkRcQ1gR7InVybCI6InlhLWRpc2stcHVibGljOi8vNzl6WUNQS1BlcFJrN09oZThGWEZtVWd0cEFBYWp6R2hzbk5yZHZwRXRHMVE5cjBBWUd6SFBuWEpRd1ZTTmNNSnEvSjZicG1SeU9Kb25UM1ZvWG5EYWc9PSIsInRpdGxlIjoiemF2dHJhX2RhdGFmZXN0X2Vtb3BvayAoMykgKDEpLnBkZiIsInVpZCI6IjExNzQ3NTU3NCIsInl1IjoiMjU1NDQwOTE1MzM1NzQ3MTciLCJub2lmcmFtZSI6ZmFsc2UsInRzIjoxNTU3NjQ5NTA1MTQxfQ%3D%3D)

### Prediction example

**message**: Я в тюрьме всем нашим **ауе** здесь так плохо и одиноко лучше сюда непопадать здесь ломается моя жизнь 

**recommended emojis**: 😭😔😒😪😢

**original emoji**: 😪

more examples [here](http://htmlpreview.github.io/?https://github.com/olgasilyutina/emopok/blob/master/example_predictions.html)

### References

Authored by [Aina Nurmagombetova](https://github.com/anurma) 🤙 [Alina Cherepanova](https://github.com/alinacherepanova) 🙋 [Anya Bataeva](https://github.com/fyzbt/) 🤯 [Olya Silyutina](https://github.com/olgasilyutina) 🤔

