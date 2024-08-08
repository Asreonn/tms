import pandas as pd
import json

csv_file_path = 'AnomalyDedection/tms_acceleration.csv'
json_file_path = 'AnomalyDedection/output.json'

df = pd.read_csv(csv_file_path, delimiter=';')

df_first_100 = df.head(100000)
float_columns = df_first_100.select_dtypes(include=['float64', 'object']).columns
for col in float_columns:
    df_first_100[col] = df_first_100[col].astype(str).str.replace(',', '.').astype(float)

json_data = df_first_100.to_json(orient='records', lines=False, indent=4)

with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)

print(f'İlk 100 satır {json_file_path} dosyasına başarıyla kaydedildi.')
