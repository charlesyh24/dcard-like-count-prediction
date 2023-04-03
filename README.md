# Dcard Like Count Prediction

## Description
It cleans and explores posts from [Dcard](https://www.dcard.tw/f), and trains tree-based models to extrapolate 24-hour like count of new posts.

## Purpose
It comes from technical interview of [Dcard's 2023 Machine Learning Intern](https://medium.com/dcardlab/medium-2023-machine-learning-intern-take-home-test-%E8%A7%A3%E6%9E%90-955aaa431ea2).

## Usage
First, you have to replace [training](data/dcard-post-train.csv)  and [test](data/dcard-post-test.csv) datasets. They are only for reference purpose due to NDA with Dcard. Then, you can run [main script](code/main.py). Tree-based models will be trained and evaluated by MAPE, which will show up at the end.

_(P.S. For training dataset, it is highly recommended that date range should be set up on the basis of complete weeks.)_

## Procedure
1. Preprocessing
    - Change data types
    - Encode `title`
    - Extract weekday and hour from `created_at` and encode them
    - Log-transform all `like_count`s
    - Encode `forum`
    - Discard all `comment_count`s, `author_id`, and `forum_stats`
1. Modeling
    - Random Forest
    - Gradient Boosting
    - Decision Tree
    - Hyperparameter Tuning
    - Ensemble Learning

## Content
- data (reference datasets for Dcard posts)
    - [dcard-post-train.csv](data/dcard-post-train.csv)
    - [dcard-post-test.csv](data/dcard-post-test.csv)
- code
    - [eda.ipynb](code/eda.ipynb): EDA report
    - [main.py](code/main.py): main script for running code
    - [preprocessing.py](code/preprocessing.py): sub script for preprocessing
    - [modeling.py](code/modeling.py): sub script for modeling