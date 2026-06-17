from airflow import DAG 
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
import pandas as pd
from datetime import datetime
import numpy as np
import calendar

default_args={
    'owner':'keerthana',
    'retries':1,
}


def engineer_features():
    hook=PostgresHook(postgres_conn_id='rossmann_postgres')
    sql="""
        select * from sales_raw join store on sales_raw.store=store.store
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
     train = train.drop(columns=['sales_date'])
     test  = test.drop(columns=['sales_date'])
     engine=hook.get_sqlalchemy_engine()
     train.to_sql('train_data',engine,if_exists='replace',index=False)
     test.to_sql('test_data',engine,if_exists='replace',index=False)

     print(f"Train size: {len(train)}, Test size: {len(test)}, Cutoff: {cutoff.date()}")






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

    engineer_feature_task >> split_data_task
    