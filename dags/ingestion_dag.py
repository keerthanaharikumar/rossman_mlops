from airflow import DAG 
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
import pandas as pd
from datetime import datetime


default_args={
    'owner':'keerthana',
    'retries':1,
}

def load_store():
        df=pd.read_csv("/opt/airflow/data/store.csv")

        nullable_int_cols = [ 'CompetitionDistance',
            'CompetitionOpenSinceMonth', 
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek',
            'Promo2SinceYear']
        for col in nullable_int_cols:
               df[col] = df[col].astype('Int64')
               
        df['Promo2'] = df['Promo2'].map({1: True, 0: False})
      

        hook=PostgresHook(postgres_conn_id='rossmann_postgres')

        for _,row in df.iterrows():
             hook.run(
                   """
                insert into store (Store,StoreType,Assortment,CompetitionDistance,CompetitionOpenSinceMonth, CompetitionOpenSinceYear,Promo2,Promo2SinceWeek,Promo2SinceYear,PromoInterval)
                values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict(Store) do nothing
                """,
                parameters=tuple(None if pd.isna(v) else v 
                                 for v in (
                                        row['Store'], row['StoreType'], row['Assortment'],
                                        row['CompetitionDistance'], row['CompetitionOpenSinceMonth'],
                                        row['CompetitionOpenSinceYear'], row['Promo2'],
                                        row['Promo2SinceWeek'], row['Promo2SinceYear'],
                                        row['PromoInterval']
                                 ))
             )


def get_watermark():
      hook=PostgresHook(postgres_conn_id='rossmann_postgres')
      sql="""
        select last_load_date from etl_metadata where table_name='sales_raw'
        """
      result=hook.get_first(sql)
      
      if result is None or result[0] is None:
            return '2013-01-01'
      return str(result[0])


def load_sales_batch(**context):
      last_date = context['ti'].xcom_pull(task_ids='get_watermark')
      df=pd.read_csv("/opt/airflow/data/train.csv")

      df['Date'] = pd.to_datetime(df['Date'])
      df = df.drop(columns=['DayOfWeek'])

      last_date = pd.to_datetime(last_date)
      batch_end = last_date + pd.Timedelta(weeks=6)
    
      batch = df[(df['Date'] > last_date) & (df['Date'] <= batch_end)]


      nullable_int_cols = ["Sales","Customers","Open","Promo","SchoolHoliday"]

      for col in nullable_int_cols:
        batch[col] = batch[col].astype('Int64')
               
      batch['Promo'] = batch['Promo'].map({1: True, 0: False})
      batch['Open'] = batch['Open'].map({1: True, 0: False})
      batch['SchoolHoliday'] = batch['SchoolHoliday'].map({1: True, 0: False})
      batch['Date'] = batch['Date'].dt.date
      

      hook=PostgresHook(postgres_conn_id='rossmann_postgres')

      for _,row in batch.iterrows():
            hook.run(
                   """
                insert into sales_raw (store,sales_date,sales,customers,open,promo,stateHoliday,schoolHoliday)
                values(%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (store, sales_date) DO NOTHING
                """,
                parameters=tuple(None if pd.isna(v) else v 
                                 for v in (
                                        row['Store'], row['Date'], row['Sales'],
                                        row['Customers'], row['Open'],
                                        row['Promo'], row['StateHoliday'],
                                        row['SchoolHoliday']
                                 ))
             )
      if batch.empty:
        return str(last_date.date())  
      return str(batch['Date'].max())
 

def update_watermark(**context):
      latest = context['ti'].xcom_pull(task_ids='load_sales_batch')
      hook=PostgresHook(postgres_conn_id='rossmann_postgres')
      hook.run(
             """
                INSERT INTO etl_metadata (table_name, last_load_date)
                VALUES ('sales_raw', %s)
                ON CONFLICT (table_name) DO UPDATE SET last_load_date = EXCLUDED.last_load_date
                """,
                parameters=(latest,)
    
      )
      


      

      
with DAG (
    dag_id='rossmann_ingestion',
    default_args=default_args,
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
) as dag:
    create_tables = PostgresOperator(
         
         task_id="create_rossmann_table",
            postgres_conn_id="rossmann_postgres",  
            sql="""
                create table if not exists store(
                    Store int primary key,
                    StoreType char(1),
                    Assortment char(1),
                    CompetitionDistance int, 
                    CompetitionOpenSinceMonth int, 
                    CompetitionOpenSinceYear int, 
                    Promo2 boolean, 
                    Promo2SinceWeek int, 
                    Promo2SinceYear int, 
                    PromoInterval text
                );

                create table if not exists sales_raw(
                    Store int references store(Store),
                    sales_date date,
                    Sales int,
                    Customers int,
                    Open boolean,
                    Promo boolean,
                    StateHoliday char(1),
                    SchoolHoliday boolean,
                    primary key(Store,sales_date)
                );

                create table if not exists etl_metadata(
                    table_name varchar primary key,
                    last_load_date date
                );

                CREATE TABLE IF NOT EXISTS model_metrics (
                 id SERIAL PRIMARY KEY,
                model_name VARCHAR,
                rmse FLOAT,
                mae FLOAT,
                r2 FLOAT,
                train_size INT,
                trained_at TIMESTAMP,
                data_up_to DATE
                );

                CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR,
                store INT,
                sales_date DATE,
                actual_sales FLOAT,
                predicted_sales FLOAT,
                trained_at TIMESTAMP
                );
                
            """
    )
    


    load_store_task=PythonOperator(
          task_id='load_store',
          python_callable=load_store,
    )

    load_sales_task=PythonOperator(
         task_id='load_sales_batch',
         python_callable=load_sales_batch,
         
   )

    get_watermark_task=PythonOperator(
         task_id='get_watermark',
         python_callable=get_watermark,
         
   )
  
    update_watermark_task=PythonOperator(
         task_id='update_watermark',
         python_callable=update_watermark,
         
   )
   
    create_tables >> load_store_task >> get_watermark_task >> load_sales_task >> update_watermark_task
    