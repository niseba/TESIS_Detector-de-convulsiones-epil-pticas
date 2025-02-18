

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
import os

file = "P9.edf"
if not os.path.isfile(file):
    print(f"El archivo {file} no se encuentra en la ubicación especificada.")
else:
    # Código para cargar el archivo
    data = mne.io.read_raw_edf(file, preload=True)
    # Continúa con el resto del código
# Cargar archivo EDF
file = "P9.edf"
data = mne.io.read_raw_edf(file, preload=True)
raw_data = data.get_data()

# Número total de muestras
ntotalsamp = raw_data.size

# Obtener la tasa de muestreo de los datos
sfreq = int(data.info['sfreq'])
print(data.info['sfreq'])

# Número de canales
nchan = len(data.ch_names)

# Total de muestras por canal
nsamp_chann = int(ntotalsamp / nchan)

# Tiempo total en segundos
ntotal_chann = nsamp_chann / sfreq

# Vector de tiempo
t = np.linspace(0, ntotal_chann, num=nsamp_chann)

# Selección del canal de señal EMG
canal_emg = 31
if canal_emg >= nchan:
    raise IndexError(f"El archivo EDF solo tiene {nchan} canales. No se puede seleccionar el canal {canal_emg}.")
senalEMG = raw_data[canal_emg, 0:nsamp_chann]

# Parámetros del filtro Chebyshev1
N2 = 10  # Orden del filtro
rp2 = 1  # Máxima ganancia permitida en la banda de paso en dB
fc2 = 145  # Frecuencia de corte en Hz

# Normalizar la frecuencia de corte
wn2 = fc2 / (sfreq / 2)

b_iir, a_iir = signal.cheby1(N2, rp2, wn2, 'high', analog=False, output='ba')

print("El orden del filtro Chebyshev Tipo I es:", max(len(b_iir), len(a_iir)) - 1)
print("Cantidad de coeficientes del filtro:", len(a_iir) + len(b_iir) - 2)

w_iir, h_iir = signal.freqz(b_iir, a_iir)
h_iir = 20 * np.log10(abs(h_iir) + 0.000001)

# Filtro Chebyshev 1 con Fc 105 Hz
fc2_105 = 105  # Frecuencia de corte en Hz
wn2_105 = fc2_105 / (sfreq / 2)

b_iir2, a_iir2 = signal.cheby1(N2, rp2, wn2_105, 'high', analog=False, output='ba')
print("El orden del filtro Chebyshev Tipo I es:", max(len(b_iir2), len(a_iir2)) - 1)

w_iir2, h_iir2 = signal.freqz(b_iir2, a_iir2)
h_iir2 = 20 * np.log10(abs(h_iir2) + 0.000001)

# Filtro Chebyshev 1 con Fc desplazada
orden = 21
wc = 0.22
b_iir3, a_iir3 = signal.cheby1(orden, rp2, wc, 'high')

print("El orden del filtro Chebyshev 1 desplazado es:", max(len(b_iir3), len(a_iir3)) - 1)

w_iir3, h_iir3 = signal.freqz(b_iir3, a_iir3)
h_iir3 = 20 * np.log10(abs(h_iir3) + 0.000001)

# Respuesta en frecuencia
fig, axis = plt.subplots()
axis.plot((sfreq / (2 * np.pi)) * w_iir, h_iir, 'blue', linewidth=2, label='IIR Chebyshev Tipo I, Fc = 145 Hz')
axis.plot((sfreq / (2 * np.pi)) * w_iir2, h_iir2, 'green', linewidth=2, label='IIR Chebyshev Tipo I, Fc = 105 Hz')
axis.plot((sfreq / (2 * np.pi)) * w_iir3, h_iir3, 'red', linewidth=2, label='IIR Chebyshev Tipo I, Fc desplazada')

axis.set_title('RESPUESTA EN FRECUENCIA', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Atenuación (dB)', fontsize=12)
plt.xticks(range(0, 750, 50))
axis.grid()
axis.legend()
plt.show()

# Aplicación de filtros Chebyshev
senal_filtrada = signal.filtfilt(b_iir, a_iir, senalEMG, padlen=50)
senal_filtrada2 = signal.filtfilt(b_iir2, a_iir2, senalEMG, padlen=50)
senal_filtrada3 = signal.filtfilt(b_iir3, a_iir3, senalEMG, padlen=50)

# Gráficas de señal EMG filtrada
fig, axis = plt.subplots()
axis.plot(t, senal_filtrada * 1000, 'blue', linewidth=2, label='IIR Chebyshev Tipo I, Fc = 145 Hz')
axis.plot(t, senal_filtrada2 * 1000, 'green', linewidth=2, label='IIR Chebyshev Tipo I, Fc = 105 Hz')
axis.plot(t, senal_filtrada3 * 1000, 'red', linewidth=2, label='IIR Chebyshev Tipo I, Fc desplazada')

axis.set_title('Señal EMG Filtrada', fontsize=12)
axis.set_xlabel('Tiempo (s)', fontsize=12)
axis.set_ylabel('Amplitud (mV)', fontsize=12)
axis.grid()
axis.legend()
plt.show()

# Parámetros del filtro FIR
numtaps = 61
cutoff = 225
b_fir = signal.firwin(numtaps, cutoff, fs=sfreq, pass_zero=False)

w_fir, h_fir = signal.freqz(b_fir)
h_fir = 20 * np.log10(abs(h_fir) + 0.000001)

# Respuesta en frecuencia del filtro FIR
fig, axis = plt.subplots()
axis.plot((sfreq / (2 * np.pi)) * w_fir, h_fir, 'blue', linewidth=2, label='FIR')
axis.plot((sfreq / (2 * np.pi)) * w_iir, h_iir, 'red', linewidth=2, label='IIR Chebyshev Tipo I, Fc = 145 Hz')
axis.plot((sfreq / (2 * np.pi)) * w_iir2, h_iir2, 'green', linewidth=2, label='IIR Chebyshev Tipo I, Fc = 105 Hz')
axis.set_title('RESPUESTA EN FRECUENCIA', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Atenuación (dB)', fontsize=12)
plt.xticks(range(0, 750, 50))
axis.grid()
axis.legend()
plt.show()

# Gráfico de la señal EMG original
fig2, axis2 = plt.subplots()
axis2.plot(t, senalEMG * 1000, 'b', linewidth=2)
axis2.set_title('Señal EMG Original', fontsize=12)
axis2.set_xlabel('Tiempo (s)', fontsize=12)
axis2.set_ylabel('Amplitud (mV)', fontsize=12)
axis2.grid()
plt.show()

# Gráfico de la señal EMG filtrada
fig, axis = plt.subplots()
axis.plot(t, senal_filtrada * 1000, 'orange', linewidth=2)
axis.set_title('Señal EMG Filtrada', fontsize=12)
axis.set_xlabel('Tiempo (s)', fontsize=12)
axis.set_ylabel('Amplitud (mV)', fontsize=12)
axis.grid()
plt.show()

# Parámetros de la señal
muestras_por_segundo = 1024  # Frecuencia de muestreo

# Parámetros de la ventana
longitud_ventana = int(muestras_por_segundo)  # 1 segundo
solapamiento = int(longitud_ventana * 0.75)  # 75% de solapamiento

# Detectar los cruces por cero utilizando el método de ventana con histéresis
vol_t = 50e-6  # Umbral de voltaje para el cruce por cero
ventanas = np.array([senal_filtrada[i:i + longitud_ventana]
                     for i in range(0, len(senal_filtrada) - longitud_ventana + 1, solapamiento)])
cruces_por_cero = []

for ventana in ventanas:
    state = True
    cruces = 0

    for i in range(1, len(ventana)):
        if state and ventana[i - 1] > -vol_t and ventana[i] < -vol_t:
            # Cruce positivo por cero
            cruces += 1
            state = False

        if not state and ventana[i - 1] < vol_t and ventana[i] > vol_t:
            # Cruce negativo por cero
            cruces += 1
            state = True

    cruces_por_cero.append(cruces)

# Detección de epilepsia
umbral_cruces = 319
ventanas_consecutivas = 3

deteccion_epilepsia = False

if len(cruces_por_cero) >= ventanas_consecutivas:
    for i in range(len(cruces_por_cero) - ventanas_consecutivas + 1):
        if all(c > umbral_cruces for c in cruces_por_cero[i:i + ventanas_consecutivas]):
            deteccion_epilepsia = True
            break

if deteccion_epilepsia:
    print("Se detectó un episodio epiléptico.")
else:
    print("No se detectaron episodios epilépticos.")
