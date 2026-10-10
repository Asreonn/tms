import pandas as pd
from pymongo import MongoClient
import schedule
import time

client = MongoClient('mongodb://localhost:27017/')
db = client['trainDatas']
collection = db['trainDatas']

csv_file_path = 'tms_acceleration.csv'

df = pd.read_csv(csv_file_path, delimiter=';')

df = df.apply(lambda x: x.astype(str).str.replace(',', '.').astype(float) if x.dtype == "object" else x)

initial_data = df.head(50000).to_dict(orient='records')
collection.insert_many(initial_data)

print("İlk 50,000 satır MongoDB'ye yüklendi.")

remaining_data = df.iloc[50000:]

start_index = 0


def upload_next_1000():
    global start_index
    end_index = start_index + 1000
    
    if start_index >= len(remaining_data):
        print("Tüm veriler yüklendi.")
        return schedule.CancelJob
    
    next_batch = remaining_data.iloc[start_index:end_index].to_dict(orient='records')
    collection.insert_many(next_batch)
    start_index = end_index
    
    print(f"{start_index} satır MongoDB'ye yüklendi.")

schedule.every(30).seconds.do(upload_next_1000)

while True:
    schedule.run_pending()
    time.sleep(1)
