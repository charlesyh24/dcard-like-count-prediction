import pandas as pd
import numpy as np

def chg_dtype(df, col_dt, col_cat):
    """Change data types to datetime and category"""
    df[col_dt]  = pd.to_datetime(df[col_dt]).dt.tz_convert(tz='ROC')
    
    df[col_cat] = df[col_cat].astype('category')
    
    return df

def encode_title(df, col):
    """Encode title into 1(meaningful) or 0(meaningless)"""
    title = df[col].str.replace(' ', '')
    
    ind_alpha = title.str.contains('[A-Za-z]'*3, regex=True)
    ind_chn = title.str.contains('[\u4e00-\u9FFF]'*2, regex=True)
    df[col + '_mean'] = np.where((ind_alpha | ind_chn), 1, 0)
    
    df.drop(col, axis=1, inplace=True)
    
    return df

def extr_create(df, col, col_wkday, col_hr):
    """Extract weekday and hour from create date"""
    df[col_wkday] = df[col].dt.weekday
    df[col_hr] = df[col].dt.hour
    
    df.drop(col, axis=1, inplace=True)
    
    return df

def encode_create(df_train, df_test, cols):
    """Frequency-encode create weekday and hour"""
    for col in cols:
        n = df_train.groupby([col])[col].count()
        df_train = df_train.merge(n, how='left', left_on=col, right_index=True,
                                  suffixes=('_drop', ''))
        df_test = df_test.merge(n, how='left', left_on=col, right_index=True,
                                suffixes=('_drop', ''))

        df_train.drop(col + '_drop', axis=1, inplace=True)
        df_test.drop(col + '_drop', axis=1, inplace=True)
    
    return df_train, df_test

def log_trans_like(df, cols):
    """Log transform like count"""
    for col in cols:
        df[col + '_log'] = np.log(df[col])
        
        df.drop(col, axis=1, inplace=True)
    
    return df

def encode_forum(df_train, df_test, col, q):
    """Encode forum into 1(popular) or 0(unpopular)"""
    # calculate count for each forum
    n = df_train.groupby([col])[col].count()
    df_train = df_train.merge(n, how='left', left_on=col, right_index=True,
                              suffixes=('_drop', ''))
    df_test = df_test.merge(n, how='left', left_on=col, right_index=True,
                            suffixes=('_drop', ''))

    df_train.drop(col + '_drop', axis=1, inplace=True)
    df_test.drop(col + '_drop', axis=1, inplace=True)
    
    # use one-hot-encoding for count
    bar = n.quantile(q)
    df_train[col + '_pop'] = np.where(df_train[col] >= bar, 1, 0)
    df_test[col + '_pop'] = np.where(df_test[col] >= bar, 1, 0)
    
    df_train.drop(col, axis=1, inplace=True)
    df_test.drop(col, axis=1, inplace=True)
    
    return df_train, df_test

def drop_cols(df, cols):
    """Drop redundant columns"""
    df.drop(cols, axis=1, inplace=True)
    
    return df

def remove_nan_inf(df):
    """Remove nan/inf/-inf values"""
    df = df.replace(np.nan, 0).replace(np.inf, 0).replace(-np.inf, 0)
    
    return df

def split_train_test(df_train, df_test, col):
    """Split data into training and test sets"""
    X_train = df_train.drop(col, axis=1)
    X_test = df_test.drop(col, axis=1)
    
    y_train = df_train[col]
    y_test = df_test[col]
    
    return X_train, X_test, y_train, y_test

def preproc_data(df_train, df_test):
    """
    Steps for preprocess data:
    1. Change data types to datetime and category
    2. Encode title into 1(meaningful) or 0(meaningless)
    3. Extract weekday and hour from create date
    4. Frequency-encode create weekday and hour
    5. Log transform like count
    6. Encode forum into 1(popular) or 0(unpopular)
    7. Drop redundant columns
    8. Remove nan/inf/-inf values
    9. Split data into training and test sets
    """
    # name column variables
    col_title = 'title'
    col_create = 'created_at'
    col_create_wkday = col_create.replace('_at', '_wkday')
    col_create_hr = col_create.replace('_at', '_hr')
    col_like = ['like_count_1h',
                'like_count_2h',
                'like_count_3h',
                'like_count_4h',
                'like_count_5h',
                'like_count_6h',
                'like_count_24h']
    col_forum = 'forum_id'
    col_author = 'author_id'
    col_drop = ['comment_count_1h',
                'comment_count_2h',
                'comment_count_3h',
                'comment_count_4h',
                'comment_count_5h',
                'comment_count_6h',
                'author_id',
                'forum_stats']
    col_tgt = 'like_count_24h_log'
    
    # preprocess data
    df_train = chg_dtype(df_train, col_create, [col_forum, col_author])
    df_test = chg_dtype(df_test, col_create, [col_forum, col_author])
    
    df_train = encode_title(df_train, col_title)
    df_test = encode_title(df_test, col_title)
    
    df_train = extr_create(df_train, col_create, col_create_wkday, col_create_hr)
    df_test = extr_create(df_test, col_create, col_create_wkday, col_create_hr)
    
    df_train, df_test = encode_create(df_train, df_test,
                                      [col_create_wkday, col_create_hr])
    
    df_train = log_trans_like(df_train, col_like)
    df_test = log_trans_like(df_test, col_like)

    df_train, df_test = encode_forum(df_train, df_test, col_forum, 0.9)

    df_train = drop_cols(df_train, col_drop)
    df_test = drop_cols(df_test, col_drop)
    
    df_train = remove_nan_inf(df_train)
    df_test = remove_nan_inf(df_test)
        
    X_train, X_test, y_train, y_test = split_train_test(df_train, df_test, col_tgt)
    
    return X_train, X_test, y_train, y_test