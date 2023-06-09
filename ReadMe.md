# Titanic Survivors Data Research

**Основное содержание проекта (статья) находится в файле [`Paper.md`](https://github.com/pas-zhukov/Titanic-Survivors-Data-Research/blob/master/Paper.md).**

---

## Предисловие

Данный проект представляет собой научную работу по анализу данных и построению предиктивной модели. Главная и основная форма этой работы - статья, которая приложена в виде .md файла. Статья также [опубликована на Хабре](https://habr.com/ru/post/726454/).

В ходе этой работы был проведен анализ данных о пассажирах, с использованием таких библиотек, как pandas, numpy, matplotlib, seaborn. Была проведена обширная работа с источниками, касающихся исследования катастрофы Титаника, а также исследующих те же самые данные.

Исходные данные были взяты из соревнования на сайте [Kaggle.com](https://www.kaggle.com/competitions/titanic).

Для создания предиктивной модели была создана нейронная сеть на базе фреймворка pyTorch, её структура также описана в статье.

В соревновании Kaggle на тестовых данных удалось получить точность **78.7%** ([ссылка на профиль с результатом](https://www.kaggle.com/paszhukov/competitions?tab=active)).

## Summary

15 апреля 1912 года произошло крушение парохода «Титаник», став одной из самых значимых катастроф в истории человечества. В данной статье исследованы данные пассажиров Титаника (предоставленные в рамках ML-соревнования на [kaggle.com](https://www.kaggle.com/competitions/titanic/data)), сделаны и проверены предположения о влиянии определённых факторов на вероятность человека выжить в той катастрофе. Анализ данных сопровождается примерами кода на Python, с использованием пакета Pandas. Построена и обучена модель нейронной сети, предсказывающая вероятность человека выжить в катастрофе с точностью 0.78 на тестовых данных. Модель построена на базе фреймворка pyTorch.

## Содержание

* **[Статья полностью](https://github.com/pas-zhukov/Titanic-Survivors-Data-Research/blob/master/Paper.md)**

Код (ссылки на файлы проекта):
* [Анализ данных, feature engineering - `Data_Investigation.ipynb`](https://github.com/pas-zhukov/Titanic-Survivors-Data-Research/blob/master/Data_Investigation.ipynb)
* [Подготовка и нормализация данных - `data_functions.py`](https://github.com/pas-zhukov/Titanic-Survivors-Data-Research/blob/master/data_functions.py)
* [Структура нейронной сети - `networks.py`](https://github.com/pas-zhukov/Titanic-Survivors-Data-Research/blob/master/networks.py)
* [Обучение модели и вывод решения для тестовых данных `learning.py`](https://github.com/pas-zhukov/Titanic-Survivors-Data-Research/blob/master/learning.py)

## Личные впечатления

Для меня это стало первой серьезной работой по анализу данных. Пришлось поработать с большим колчиством источников, много вечеров просидеть, составляя разумный текст и картинки.
В ходе работы по необходимости и случайно пришлось узнать много нового. Причем, не все в итоге удалось применить в статье. Где-то многострочный код можно было заменить парой строк из pandas, но переписывать его уже не было смысла, т.к. всё необходимое от этого кода я уже получил.
Однако, в дальнейшем этот опыт мне точно пригодится для получения нового. Это соревнование на kaggle не зря предлагается как первое для вхождения в их среду, здесь есть над чем подумать, о чем почитать. Всегда можно поэкспериментировать с разными моделями из sci-kit learn и получить интересные результаты. 
