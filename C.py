# ... (todo igual arriba)

    # Botón para ejecutar
    if st.button("▶️ RUN"):
        # Preprocesamiento
        audio_filtrado = bandpass_filter(audio, lowcut, highcut, sr)
        num_muestras = 8 * sr
        trozos = []
        trozos_audio = []
        posiciones = []  # Para saber en qué parte del audio va cada fragmento

        for i in range(0, len(audio_filtrado), num_muestras):
            fragmento = audio_filtrado[i:i + num_muestras]
            if len(fragmento) == num_muestras:
                trozos_audio.append(fragmento)
                posiciones.append(i)  # posición inicial del fragmento
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

            # Mostrar clustering
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

            # Mostrar espectrograma completo del audio filtrado
            st.subheader("🎛️ Espectrograma completo del audio filtrado")
            S_full = librosa.stft(audio_filtrado, n_fft=n_fft, hop_length=256)
            S_db_full = librosa.amplitude_to_db(np.abs(S_full), ref=np.max)
            fig_full, ax_full = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(S_db_full, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=ax_full)
            ax_full.set_title("Espectrograma completo")
            ax_full.set_ylim(100, 20000)
            plt.colorbar(img, ax=ax_full, format="%+2.f dB")
            st.pyplot(fig_full)

            # Espectrogramas por cluster
            st.subheader("🔎 Espectrogramas por Cluster")
            for cluster_id in range(n_clusters):
                st.markdown(f"**Cluster {cluster_id}**")
                idx = np.where(clusters == cluster_id)[0][:ejemplos_por_cluster]
                for j, i in enumerate(idx):
                    fragmento = trozos_audio[i]
                    S = librosa.stft(fragmento, n_fft=n_fft, hop_length=256)
                    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    img = librosa.display.specshow(S_db, sr=sr, hop_length=256,
                                                   x_axis='time', y_axis='hz', ax=ax2)
                    tiempo_inicio = posiciones[i] / sr
                    ax2.set_title(f"Cluster {cluster_id} - Ejemplo {j+1} (desde {tiempo_inicio:.1f} s)")
                    ax2.set_ylim(100, 20000)
                    plt.colorbar(img, ax=ax2, format="%+2.f dB")
                    st.pyplot(fig2)

                    # Botón de descarga
                    buf = io.BytesIO()
                    fig2.savefig(buf, format="png")
                    st.download_button(
                        label="📥 Descargar imagen PNG",
                        data=buf.getvalue(),
                        file_name=f"cluster_{cluster_id}_ejemplo_{j+1}.png",
                        mime="image/png"
                    )
                    plt.close(fig2)
