import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import butter, lfilter
import io

# Funciones
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Título
st.title("🎧 Identificación de patrones sonoros con Espectrogramas")

# Sidebar
st.sidebar.header("Configuración")
n_fft = st.sidebar.selectbox("Tamaño de ventana (n_fft)", [1024, 2046, 4096], index=2)
n_clusters = st.sidebar.slider("Cantidad de Clusters", 1, 5, 3)
ejemplos_por_cluster = st.sidebar.radio("Ejemplos por Cluster", [1, 2])
lowcut = 60
highcut = 15000
duracion_fragmento = 10  # segundos (MODIFICADO AQUÍ)
sr = 44100

# Cargar audio
archivo_audio = st.file_uploader("Sube tu archivo WAV", type=["wav"])

if archivo_audio:
    audio, sr_actual = librosa.load(archivo_audio, sr=sr)
    if len(audio) > sr * 60:
        audio = audio[:sr * 60]  # Recorta a 1 minuto
        st.warning("Audio recortado a 60 segundos.")
    else:
        st.info("Audio cargado completamente (menos de 1 minuto).")

    # Botón para ejecutar
    if st.button("▶️ RUN"):
        # Preprocesamiento
        audio_filtrado = bandpass_filter(audio, lowcut, highcut, sr)
        num_muestras = duracion_fragmento * sr
        trozos = []
        trozos_audio = []

        for i in range(0, len(audio_filtrado), num_muestras):
            fragmento = audio_filtrado[i:i + num_muestras]
            if len(fragmento) == num_muestras:
                trozos_audio.append(fragmento)
                mfcc = librosa.feature.mfcc(y=fragmento, sr=sr, n_mfcc=20)
                delta = librosa.feature.delta(mfcc)
                features = np.vstack([mfcc, delta])
                mean_features = np.mean(features.T, axis=0)
                log_features = np.log1p(np.abs(mean_features))
                trozos.append(log_features)

        if len(trozos) == 0:
            st.error("No se generaron fragmentos. Verifica el archivo.")
        else:
            # Clustering
            X = np.array(trozos)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            X_reducido = pca.fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_reducido)

            # Plot clusters
            fig1, ax1 = plt.subplots(figsize=(7, 6))
            colors = plt.cm.viridis(np.linspace(0.3, 1, n_clusters))
            for i in range(n_clusters):
                puntos = X_reducido[clusters == i]
                ax1.scatter(puntos[:, 0], puntos[:, 1], color=colors[i], label=f"Cluster {i}", s=50, alpha=0.6)
            ax1.set_title("Clustering de Fragmentos")
            ax1.set_xlabel("PCA 1")
            ax1.set_ylabel("PCA 2")
            ax1.legend()
            st.pyplot(fig1)

            # Espectrogramas
            st.subheader("Espectrogramas por Cluster")
            for cluster_id in range(n_clusters):
                st.markdown(f"**Cluster {cluster_id}**")
                idx = np.where(clusters == cluster_id)[0][:ejemplos_por_cluster]
                for j, i in enumerate(idx):
                    S = librosa.stft(trozos_audio[i], n_fft=n_fft, hop_length=256)
                    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    img = librosa.display.specshow(S_db, sr=sr, hop_length=256,
                                                   x_axis='time', y_axis='hz', ax=ax2)
                    ax2.set_title(f"Cluster {cluster_id} - Ejemplo {j+1}")
                    ax2.set_ylim(60, 15000)
                    plt.colorbar(img, ax=ax2, format="%+2.f dB")
                    st.pyplot(fig2)

                    # Botón para descargar imagen
                    buf = io.BytesIO()
                    fig2.savefig(buf, format="png")
                    st.download_button(
                        label="📥 Descargar imagen PNG",
                        data=buf.getvalue(),
                        file_name=f"cluster_{cluster_id}_ejemplo_{j+1}.png",
                        mime="image/png"
                    )
                    plt.close(fig2)
