import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import joblib
import os
import folium
from folium.plugins import HeatMap
import schedule
import time

@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.mse(y_true, y_pred)

def build_best_autoencoder(hyperparameters, input_dim):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = input_layer
    for i in range(hyperparameters['num_layers']):
        encoded = tf.keras.layers.Dense(units=hyperparameters[f'units_{i}'], activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = tf.keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
                        loss=custom_mse)
    return autoencoder

def create_heatmap(data, lat_col='latitude', lon_col='longitude', value_col='reconstruction_error'):
    heat_data = [[row[lat_col], row[lon_col], row[value_col]] for index, row in data.iterrows()]

    m = folium.Map(location=[data[lat_col].mean(), data[lon_col].mean()], zoom_start=10)
    HeatMap(heat_data).add_to(m)
    heatmap_path = os.path.join(os.getcwd(), 'heatmap.html')
    m.save(heatmap_path)
    print(f"Heatmap '{heatmap_path}' dosyasına kaydedildi.")

def load_and_prepare_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['trainDatas']
    collection = db['trainDatas']
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))

    data['max_acc_z'] = data['max_acc_z'].astype(str).str.replace(',', '.').astype(float)
    return data

def train_initial_model():
    data = load_and_prepare_data()
    X = data[['max_acc_z']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]

    best_hyperparameters = {
        "num_layers": 4,
        "units_0": 240,
        "learning_rate": 0.001,
        "units_1": 256,
        "units_2": 144,
        "units_3": 160
    }

    autoencoder = build_best_autoencoder(best_hyperparameters, input_dim)
    train_size = int(len(X_scaled) * 0.7)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]

    autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=False, validation_data=(X_test, X_test))

    model_path = os.path.join(os.getcwd(), 'best_autoencoder.h5')
    scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

    autoencoder.save(model_path)
    joblib.dump(scaler, scaler_path)

    print("Başlangıç modeli eğitildi ve kaydedildi.")
    return model_path, scaler_path

def update_and_retrain():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['trainDatas']
    collection = db['trainDatas']

    cursor = collection.find()
    data = pd.DataFrame(list(cursor))

    X = data[['max_acc_z']]
    
    scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    
    model_path = os.path.join(os.getcwd(), 'best_autoencoder.h5')
    best_autoencoder = tf.keras.models.load_model(model_path, custom_objects={'custom_mse': custom_mse})

    best_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=custom_mse)

    train_size = int(len(X_scaled) * 0.7)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]

    best_autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=False, validation_data=(X_test, X_test))
    best_autoencoder.save(model_path)

    print("Model güncellendi ve yeniden eğitildi.")
    
    reconstructed_all = best_autoencoder.predict(X_scaled)
    reconstruction_errors_all = np.mean(np.power(X_scaled - reconstructed_all, 2), axis=1)

    data['reconstruction_error'] = reconstruction_errors_all

    if 'latitude' in data.columns and 'longitude' in data.columns:
        create_heatmap(data)
    else:
        print("Latitude ve Longitude sütunları bulunamadı, ısı haritası oluşturulamıyor.")

train_initial_model()

schedule.every(5).seconds.do(update_and_retrain)

while True:
    schedule.run_pending()
    time.sleep(1)
