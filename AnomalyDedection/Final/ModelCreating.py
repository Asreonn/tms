import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import joblib
import os
import folium
from folium.plugins import HeatMap

client = MongoClient('mongodb://localhost:27017/')
db = client['trainDatas']
collection = db['trainDatas']

cursor = collection.find()
data = pd.DataFrame(list(cursor))

X = data[['max_acc_z']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_hyperparameters = {
    "num_layers": 4,
    "units_0": 240,
    "learning_rate": 0.001,
    "units_1": 256,
    "units_2": 144,
    "units_3": 160
}

def build_best_autoencoder(hyperparameters):
    input_dim = X_scaled.shape[1]
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = input_layer
    for i in range(hyperparameters['num_layers']):
        encoded = tf.keras.layers.Dense(units=hyperparameters[f'units_{i}'], activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = tf.keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
                        loss='mse')
    return autoencoder

best_autoencoder = build_best_autoencoder(best_hyperparameters)

train_size = int(len(X_scaled) * 0.7)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]

history = best_autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=False, validation_data=(X_test, X_test))

model_path = os.path.join(os.getcwd(), 'best_autoencoder.h5')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

best_autoencoder.save(model_path)
joblib.dump(scaler, scaler_path)

print("Başlangıç modeli eğitildi ve kaydedildi.")

reconstructed_all = best_autoencoder.predict(X_scaled)
reconstruction_errors_all = np.mean(np.power(X_scaled - reconstructed_all, 2), axis=1)

def create_heatmap(data, lat_col='latitude', lon_col='longitude', value_col='reconstruction_error'):
    heat_data = [[row[lat_col], row[lon_col], row[value_col]] for index, row in data.iterrows()]

    m = folium.Map(location=[data[lat_col].mean(), data[lon_col].mean()], zoom_start=10)

    HeatMap(heat_data).add_to(m)

    heatmap_path = os.path.join(os.getcwd(), 'heatmap.html')
    m.save(heatmap_path)
    print(f"Heatmap '{heatmap_path}' dosyasına kaydedildi.")

data['reconstruction_error'] = reconstruction_errors_all

if 'latitude' in data.columns and 'longitude' in data.columns:
    create_heatmap(data)
else:
    print("Latitude ve Longitude sütunları bulunamadı, ısı haritası oluşturulamıyor.")
