import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import mplcursors
from threading import Timer

# Veriyi yükleyin
data = pd.read_csv('machine/TMS/AnomalyDedection/tms_acceleration_with_errors.csv')

# İlgili verileri çıkarın
lat = data['latitude']
lon = data['longitude']
z_values = data['max_acc_z']
errors = data['reconstruction_error']
anomaly_degree = data['anomaly_degree']

# Eşik değerini belirleyin
threshold_value = np.percentile(errors, 10)

# Veriyi anomali derecesine göre sıralayın
data_sorted = data.sort_values(by='anomaly_degree')

# Anomali derecelerini renklendirme için normalize edin
norm = Normalize(vmin=0, vmax=anomaly_degree.max())
cmap = cm.RdYlGn_r

# Scatter plot ile anomali derecelerini renk kodlu olarak gösterin
plt.figure(figsize=(12, 8))
sc = plt.scatter(data_sorted['longitude'], data_sorted['latitude'], c=data_sorted['anomaly_degree'], 
                 cmap=cmap, norm=norm, marker='o', s=200, edgecolor='none')
plt.colorbar(sc, label='Anomaly Degree')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Anomaly Detection based on Z-axis Acceleration')

# Noktaların üzerine tıklandığında Z değerini ve anomali derecesini gösterin
cursor = mplcursors.cursor(sc)

def hide_annotation(sel):
    sel.annotation.set_visible(False)
    plt.draw()

@cursor.connect("add")
def on_add(sel):
    index = sel.target.index
    sel.annotation.set(text=f"Z: {z_values.iloc[index]:.2f}\nAnomaly: {anomaly_degree.iloc[index]:.2f}",
                       position=(0, 20), anncoords="offset points")
    sel.annotation.xy = (data_sorted['longitude'].iloc[index], data_sorted['latitude'].iloc[index])
    sel.annotation.set_visible(True)
    timer = Timer(5, hide_annotation, [sel])
    timer.start()

plt.show()

print("\nAnomali Tespiti Bilgileri:")
print(f"Toplam veri noktası sayısı: {len(data)}")
print(f"Anomalileri tespit edilen veri noktası sayısı: {len(np.where(anomaly_degree > 0)[0])}")
print(f"Belirlenen anomali eşiği değeri: {threshold_value}")

def print_groups(indices):
    for idx in indices:
        start_idx = max(0, idx - 2)
        end_idx = min(len(data), idx + 3)
        print(f"\nGrup (Index {idx}):")
        print(data.iloc[start_idx:end_idx][['max_acc_z', 'latitude', 'longitude', 'reconstruction_error', 'anomaly_degree']])

# En büyük
print("\nEn Büyük Error Değerleri ve Grupları:")
top_indices = errors.nlargest(3).index
print_groups(top_indices)

# En küçük
print("\nEn Küçük Error Değerleri ve Grupları:")
bottom_indices = errors.nsmallest(3).index
print_groups(bottom_indices)

# Rastgele
print("\nRastgele Error Değerleri ve Grupları:")
random_indices = np.random.choice(data.index, size=3, replace=False)
print_groups(random_indices)
