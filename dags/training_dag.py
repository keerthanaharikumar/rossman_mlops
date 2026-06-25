from airflow import DAG 
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from datetime import datetime
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import calendar

default_args={
    'owner':'keerthana',
    'retries':1,
}


def engineer_features():
    hook=PostgresHook(postgres_conn_id='rossmann_postgres')
    sql="""
                SELECT 
            s.store,
            s.sales_date,
            s.sales,
            s.customers,
            s.open,
            s.promo,
            s.stateholiday,
            s.schoolholiday,
            st.storetype,
            st.assortment,
            st.competitiondistance,
            st.competitionopensincemonth,
            st.competitionopensinceyear,
            st.promo2,
            st.promo2sinceweek,
            st.promo2sinceyear,
            st.promointerval
        FROM sales_raw s
        JOIN store st ON s.store = st.store
        """
    df=hook.get_pandas_df(sql)
    df['sales_date'] = pd.to_datetime(df['sales_date'])
    df['day_of_week'] = df['sales_date'].dt.dayofweek
    df['month']=df['sales_date'].dt.month
    df['year']=df['sales_date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'].isin([5, 6]), 1, 0)
    df['promo_x_schoolholiday'] = np.where((df['schoolholiday'] == 1) & (df['promo'] == 1), 1, 0)
    months_since = (df['year'] - df['competitionopensinceyear']) * 12 + (df['month'] - df['competitionopensincemonth'])

    df['competition_open_months'] = np.where(
        df['competitionopensincemonth'].isna() | df['competitionopensinceyear'].isna() | (months_since < 0),
        0,
        months_since
    )
    df['month_abbr'] = df['month'].apply(lambda m: calendar.month_abbr[m])
    condition3 = df.apply(lambda row: row['month_abbr'] in str(row['promointerval']).split(','), axis=1)
    def get_promo2_date(row):
        if pd.isna(row['promo2sinceyear']) or pd.isna(row['promo2sinceweek']):

            return pd.NaT
        
        year = int(row['promo2sinceyear'])
        week = int(row['promo2sinceweek'])
        return datetime.strptime(f"{year}-{week}-1", '%Y-%W-%w')

    df['promo2_start_date'] = df.apply(get_promo2_date, axis=1)
    condition2 = (df['sales_date'] >= df['promo2_start_date'])

    df['is_promo2_active'] = np.where(
        (df['promo2'] == 1) & condition2 & condition3,
        1, 0)

    df=df.sort_values(['store','sales_date'])
    df['sales_lag_7'] = df.groupby('store')['sales'].shift(7)
    df['sales_lag_28'] = df.groupby('store')['sales'].shift(28)
    df.dropna(subset=['sales_lag_7', 'sales_lag_28'], inplace=True)
    df['storetype'] = df['storetype'].map({'a':0,'b':1,'c':2,'d':3})
    df['assortment'] = df['assortment'].map({'a':0,'b':1,'c':2})
    df['stateholiday'] = df['stateholiday'].map({'0':0,'a':1,'b':2,'c':3})



    df = df.drop(columns=['month_abbr', 'promo2_start_date','customers','competitionopensincemonth','competitionopensinceyear','promo2sinceweek','promo2sinceyear','promointerval','promo2','open'])
    engine=hook.get_sqlalchemy_engine()
    df.to_sql('features',engine,if_exists='replace',index=False)


def split_data(**context):
     hook=PostgresHook(postgres_conn_id='rossmann_postgres')
     df = hook.get_pandas_df("SELECT * FROM features")
     df['sales_date'] = pd.to_datetime(df['sales_date'])
     cutoff = df['sales_date'].max() - pd.Timedelta(weeks=6)
     train = df[df['sales_date'] <= cutoff]
     test  = df[df['sales_date'] >  cutoff]
    #  train = train.drop(columns=['sales_date'])
    #  test  = test.drop(columns=['sales_date'])
     engine=hook.get_sqlalchemy_engine()
     train.to_sql('train_data',engine,if_exists='replace',index=False)
     test.to_sql('test_data',engine,if_exists='replace',index=False)

     print(f"Train size: {len(train)}, Test size: {len(test)}, Cutoff: {cutoff.date()}")



def train_linear():
    hook=PostgresHook(postgres_conn_id='rossmann_postgres')
    train=hook.get_pandas_df('select * from train_data')
    test=hook.get_pandas_df('select * from test_data')
    feature_cols = [
        'store', 'promo', 'stateholiday', 'schoolholiday',
        'storetype', 'assortment', 'competitiondistance',
        'day_of_week', 'month', 'year', 'is_weekend',
        'promo_x_schoolholiday', 'competition_open_months',
        'is_promo2_active', 'sales_lag_7', 'sales_lag_28'
    ]

    X_train = train[feature_cols]
    y_train = train['sales']
    X_test  = test[feature_cols]
    y_test  = test['sales']

    X_train = train[feature_cols].fillna(0)
    X_test  = test[feature_cols].fillna(0)

    model=LinearRegression()
    model.fit(X_train,y_train)

    preds=model.predict(X_test)

 
    pred_df = pd.DataFrame({
        'model_name': 'linear_regression',
        'store': test['store'].values,
        'sales_date': test['sales_date'].values,
        'actual_sales': y_test.values,
        'predicted_sales': preds,
        'trained_at': str(datetime.now())
    })

    engine = hook.get_sqlalchemy_engine()
    pred_df.to_sql('predictions', engine, if_exists='append', index=False)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    os.makedirs('/opt/airflow/models', exist_ok=True)
    with open('/opt/airflow/models/linear_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Linear → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    return {
        'model': 'linear_regression',
        'rmse': round(rmse, 2),
        'mae':  round(mae, 2),
        'r2':   round(r2, 4),
        'trained_at': str(datetime.now()),
        'train_size': len(X_train)
    }



def train_xgboost():
    hook=PostgresHook(postgres_conn_id='rossmann_postgres')
    train=hook.get_pandas_df('select * from train_data')
    test=hook.get_pandas_df('select * from test_data')
    feature_cols = [
        'store', 'promo', 'stateholiday', 'schoolholiday',
        'storetype', 'assortment', 'competitiondistance',
        'day_of_week', 'month', 'year', 'is_weekend',
        'promo_x_schoolholiday', 'competition_open_months',
        'is_promo2_active', 'sales_lag_7', 'sales_lag_28'
    ]

    X_train = train[feature_cols]
    y_train = train['sales']
    X_test  = test[feature_cols]
    y_test  = test['sales']


    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    pred_df = pd.DataFrame({
        'model_name': 'XGBoost',
        'store': test['store'].values,
        'sales_date': test['sales_date'].values,
        'actual_sales': y_test.values,
        'predicted_sales': preds,
        'trained_at': str(datetime.now())
    })
    
    engine = hook.get_sqlalchemy_engine()
    pred_df.to_sql('predictions', engine, if_exists='append', index=False)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    os.makedirs('/opt/airflow/models', exist_ok=True)
    with open('/opt/airflow/models/xgboost.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"XGBoost → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    return {
        'model': 'XGBoost',
        'rmse': round(rmse, 2),
        'mae':  round(mae, 2),
        'r2':   round(r2, 4),
        'trained_at': str(datetime.now()),
        'train_size': len(X_train)
    }



def train_lightgbm():
    hook=PostgresHook(postgres_conn_id='rossmann_postgres')
    train=hook.get_pandas_df('select * from train_data')
    test=hook.get_pandas_df('select * from test_data')
    feature_cols = [
        'store', 'promo', 'stateholiday', 'schoolholiday',
        'storetype', 'assortment', 'competitiondistance',
        'day_of_week', 'month', 'year', 'is_weekend',
        'promo_x_schoolholiday', 'competition_open_months',
        'is_promo2_active', 'sales_lag_7', 'sales_lag_28'
    ]

    X_train = train[feature_cols]
    y_train = train['sales']
    X_test  = test[feature_cols]
    y_test  = test['sales']

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train,y_train)

    preds=model.predict(X_test)



    pred_df = pd.DataFrame({
        'model_name': 'LightGBM',
        'store': test['store'].values,
        'sales_date': test['sales_date'].values,
        'actual_sales': y_test.values,
        'predicted_sales': preds,
        'trained_at': str(datetime.now())
    })
    
    engine = hook.get_sqlalchemy_engine()
    pred_df.to_sql('predictions', engine, if_exists='append', index=False)


    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    os.makedirs('/opt/airflow/models', exist_ok=True)
    with open('/opt/airflow/models/lightgbm.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"LightGBM → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    return {
        'model': 'LightGBM',
        'rmse': round(rmse, 2),
        'mae':  round(mae, 2),
        'r2':   round(r2, 4),
        'trained_at': str(datetime.now()),
        'train_size': len(X_train)
    }


def save_metrics(**context):
    hook=PostgresHook(postgres_conn_id='rossmann_postgres')
    linear_matrix= context['ti'].xcom_pull(task_ids='train_linear')
    xgboost_matrix= context['ti'].xcom_pull(task_ids='train_xgboost')
    lightgbm_matrix= context['ti'].xcom_pull(task_ids='train_lightgbm')
    result = hook.get_first("SELECT MAX(sales_date) FROM features")
    data_up_to = str(result[0])
    
    for metrics in [linear_matrix, xgboost_matrix, lightgbm_matrix]:
        hook.run("""
            INSERT INTO model_metrics 
                (model_name, rmse, mae, r2, train_size, trained_at, data_up_to)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, parameters=(
            metrics['model'],
            metrics['rmse'],
            metrics['mae'],
            metrics['r2'],
            metrics['train_size'],
            metrics['trained_at'],
            data_up_to
        ))
    
    print(f"Saved metrics for 3 models, data_up_to={data_up_to}")
    




with DAG (
    dag_id='rossmann_training',
    default_args=default_args,
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
) as dag:
    
    engineer_feature_task=PythonOperator(
          task_id='engineer_features',
          python_callable=engineer_features
    )

    split_data_task=PythonOperator(
          task_id='split_data',
          python_callable=split_data
    )

    train_linear_task=PythonOperator(
          task_id='train_linear',
          python_callable=train_linear
    )


    train_xgboost_task=PythonOperator(
          task_id='train_xgboost',
          python_callable=train_xgboost
    )


    train_lightgbm_task=PythonOperator(
          task_id='train_lightgbm',
          python_callable=train_lightgbm
    )

    save_metrics_task = PythonOperator(
        task_id='save_metrics',
        python_callable=save_metrics,
        
    )

    engineer_feature_task >> split_data_task >> [train_linear_task, train_xgboost_task, train_lightgbm_task] >> save_metrics_task
    