import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances_argmin_min
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# Load Data

gdf = gpd.read_file("indonesia.geojson")

url_pidana='https://drive.google.com/file/d/1-X5EC5lgPwS0hSEfBApD-SFlwYmjWacH/view?usp=drive_link'
url_pidana='https://drive.google.com/uc?id=' + url_pidana.split('/')[-2]

url_pendidikan='https://drive.google.com/file/d/1_neSxkDW_faVBJEWE6C52eEEVjK88lx3/view?usp=drive_link'
url_pendidikan='https://drive.google.com/uc?id=' + url_pendidikan.split('/')[-2]

url_pengangguran='https://drive.google.com/file/d/1K3EdFyG9LwE9Y6kow4ySDl3lME0FfK61/view?usp=drive_link'
url_pengangguran='https://drive.google.com/uc?id=' + url_pengangguran.split('/')[-2]

url_pendapatan='https://drive.google.com/file/d/1SLrnC_nL8BBFRB8wn1r8YUzpvTmMAM_6/view?usp=drive_link'
url_pendapatan='https://drive.google.com/uc?id=' + url_pendapatan.split('/')[-2]

df_pidana = pd.read_csv(url_pidana, delimiter=';')
df_pendidikan = pd.read_csv(url_pendidikan, delimiter=';')
df_pengangguran = pd.read_csv(url_pengangguran, delimiter=';')
df_pendapatan = pd.read_csv(url_pendapatan, delimiter=';')


# Pre-Processing

provinsi_to_drop = ["INDONESIA", "PAPUA SELATAN", "PAPUA BARAT DAYA", "PAPUA TENGAH", "PAPUA PEGUNUNGAN"]

df_pidana.drop(index=df_pidana[df_pidana["Provinsi"].isin(['INDONESIA'])].index, inplace=True)
df_pendidikan.drop(index=df_pendidikan[df_pendidikan["Provinsi"].isin(provinsi_to_drop)].index, inplace=True)
df_pengangguran.drop(index=df_pengangguran[df_pengangguran["Provinsi"].isin(provinsi_to_drop)].index, inplace=True)
df_pendapatan.drop(index=df_pendapatan[df_pendapatan["Provinsi"].isin(provinsi_to_drop)].index, inplace=True)

df_pidana = df_pidana.replace('-', np.nan)
print(df_pidana.isnull().sum())

for index, row in df_pidana.iterrows():
    if row.isnull().any():
        mean_row = row.iloc[1:].astype(float).mean()
        df_pidana.loc[index] = row.fillna(mean_row)

for col in df_pidana.columns[1:]:
    df_pidana[col] = pd.to_numeric(df_pidana[col])

for col in df_pendidikan.columns[1:]:
    df_pendidikan[col] = pd.to_numeric(df_pendidikan[col])

for col in df_pengangguran.columns[1:]:
    df_pengangguran[col] = pd.to_numeric(df_pengangguran[col])

for col in df_pendapatan.columns[1:]:
    df_pendapatan[col] = pd.to_numeric(df_pendapatan[col])



# Feature Engineering

# 1. Pidana
pidana_cols = [col for col in df_pidana.columns if col != 'Provinsi']
df_pidana['avg_pidana'] = df_pidana[pidana_cols].mean(axis=1)

years = list(range(len(pidana_cols)))
df_pidana['trend_pidana'] = 0.0

for idx, row in df_pidana.iterrows():
    y = row[pidana_cols].values.astype(float)
    if len(y) > 1:
        m, b = np.polyfit(years, y, 1)
        df_pidana.loc[idx, 'trend_pidana'] = m

df_pidana['std_pidana'] = df_pidana[pidana_cols].std(axis=1)

df_pidana_agg = df_pidana[['Provinsi', 'avg_pidana', 'trend_pidana', 'std_pidana']]

# 2. Pendidikan
pendidikan_cols = [col for col in df_pendidikan.columns if col != 'Provinsi']
df_pendidikan['avg_pendidikan'] = df_pendidikan[pendidikan_cols].mean(axis=1)

years = list(range(len(pendidikan_cols)))
df_pendidikan['trend_pendidikan'] = 0.0

for idx, row in df_pendidikan.iterrows():
    y = row[pendidikan_cols].values.astype(float)
    if len(y) > 1:
        m, b = np.polyfit(years, y, 1)
        df_pendidikan.loc[idx, 'trend_pendidikan'] = m

df_pendidikan['std_pendidikan'] = df_pendidikan[pendidikan_cols].std(axis=1)

df_pendidikan_agg = df_pendidikan[['Provinsi', 'avg_pendidikan', 'trend_pendidikan', 'std_pendidikan']]

# 3. Pengangguran
unemployment_by_year = {}
for year in range(2015, 2024):
    feb_col = f"{year}_Februari"
    ags_col = f"{year}_Agustus"
    if feb_col in df_pengangguran.columns and ags_col in df_pengangguran.columns:
        year_avg = (df_pengangguran[feb_col] + df_pengangguran[ags_col]) / 2
        unemployment_by_year[year] = year_avg

years = sorted(unemployment_by_year.keys())
df_pengangguran_yearly = pd.DataFrame()
df_pengangguran_yearly['Provinsi'] = df_pengangguran['Provinsi']

for year in years:
    df_pengangguran_yearly[str(year)] = unemployment_by_year[year]

df_pengangguran['avg_pengangguran'] = df_pengangguran_yearly.iloc[:, 1:].mean(axis=1)

trend_cols = [str(year) for year in years]
years_numeric = list(range(len(trend_cols)))
df_pengangguran['trend_pengangguran'] = 0.0

for idx, row in df_pengangguran_yearly.iterrows():
    y = row[trend_cols].values.astype(float)
    if len(y) > 1:
        m, b = np.polyfit(years_numeric, y, 1)
        df_pengangguran.loc[idx, 'trend_pengangguran'] = m

df_pengangguran['std_pengangguran'] = df_pengangguran_yearly.iloc[:, 1:].std(axis=1)

df_pengangguran_agg = df_pengangguran[['Provinsi', 'avg_pengangguran', 'trend_pengangguran', 'std_pengangguran']]

# 4. Pendapatan
income_cols = [col for col in df_pendapatan.columns if col != 'Provinsi']
df_pendapatan['avg_pendapatan'] = df_pendapatan[income_cols].mean(axis=1)

years = list(range(len(income_cols)))
df_pendapatan['trend_pendapatan'] = 0.0

for idx, row in df_pendapatan.iterrows():
    y = row[income_cols].values.astype(float)
    if len(y) > 1:
        m, b = np.polyfit(years, y, 1)
        df_pendapatan.loc[idx, 'trend_pendapatan'] = m

df_pendapatan['std_pendapatan'] = df_pendapatan[income_cols].std(axis=1)

df_pendapatan_agg = df_pendapatan[['Provinsi', 'avg_pendapatan', 'trend_pendapatan', 'std_pendapatan']]


# Merge all data

df_agg = pd.merge(df_pidana_agg, df_pendidikan_agg, on='Provinsi')
df_agg = pd.merge(df_agg, df_pengangguran_agg, on='Provinsi')
df_agg = pd.merge(df_agg, df_pendapatan_agg, on='Provinsi')


# Load geojson
gdf_map = gpd.read_file("indonesia.geojson")
gdf_map["state"] = gdf_map["state"].str.lower().str.strip()


# User Input

st.set_page_config(layout="wide")
st.title("Clustering Provinsi di Indonesia")
st.markdown("Pengelompokan provinsi di Indonesia berdasarkan faktor sosial-ekonomi dan kriminalitas dilakukan untuk mengelompokkan wilayah dengan karakteristik serupa, seperti tingkat pendapatan, pendidikan, pengangguran, dan tingkat kejahatan, sehingga memudahkan pemerintah dalam merancang kebijakan yang tepat sasaran untuk meningkatkan kesejahteraan dan keamanan masyarakat di setiap provinsi.")

pidana_features = ['avg_pidana', 'trend_pidana', 'std_pidana']
pendidikan_features = ['avg_pendidikan', 'trend_pendidikan', 'std_pendidikan']
pengangguran_features = ['avg_pengangguran', 'trend_pengangguran', 'std_pengangguran']
pendapatan_features = ['avg_pendapatan', 'trend_pendapatan', 'std_pendapatan']

# all_features = pidana_features + pendidikan_features + pengangguran_features + pendapatan_features
# selected_features = st.multiselect("Pilih fitur:", all_features, default=all_features[:4])

st.sidebar.header("Pilih Fitur")
use_pidana = st.sidebar.checkbox("Pidana", value=True)
use_pendidikan = st.sidebar.checkbox("Pendidikan", value=True)
use_pengangguran = st.sidebar.checkbox("Pengangguran", value=True)
use_pendapatan = st.sidebar.checkbox("Pendapatan", value=True)

all_features = pidana_features + pendidikan_features + pengangguran_features + pendapatan_features

selected_groups = []
if use_pidana:
    selected_groups.append("Pidana")
if use_pendidikan:
    selected_groups.append("Pendidikan")
if use_pengangguran:
    selected_groups.append("Pengangguran")
if use_pendapatan:
    selected_groups.append("Pendapatan")

selected_features = []
if use_pidana:
    selected_features += pidana_features
if use_pendidikan:
    selected_features += pendidikan_features
if use_pengangguran:
    selected_features += pengangguran_features
if use_pendapatan:
    selected_features += pendapatan_features

list_str = "Fitur yang dipilih:\n" + "\n".join(f"- {group}" for group in selected_groups)
st.markdown(list_str)
        


if len(selected_features) >= 2:
    df_selected = df_agg[["Provinsi"] + selected_features].copy()

    robust_scaler = RobustScaler()
    df_robust_scaled = robust_scaler.fit_transform(df_selected[selected_features])
    df_robust_scaled = pd.DataFrame(df_robust_scaled, columns=selected_features)

    n_components = 3

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_robust_scaled)

    def find_optimal_k(data, k_range=range(2, 11), title=""):
        wcss = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(data, labels))
            optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
            
            return optimal_k, wcss, silhouette_scores
    
    optimal_k_pca, _, _ = find_optimal_k(df_pca, title="PCA-transformed Data")
    optimal_k_pca = 3
    
    
    def interpret_clusters(data, labels, method_name):
        provinces = df_agg['Provinsi'].values
        
        basic_features = ['avg_pidana', 'avg_pendidikan', 'avg_pengangguran', 'avg_pendapatan']
        trend_features = ['trend_pidana', 'trend_pendidikan', 'trend_pengangguran', 'trend_pendapatan']
        
        overall_means = {}
        for feature in basic_features + trend_features:
            overall_means[feature] = df_agg[feature].mean()
        
        print(f"\n{method_name} Cluster Interpretations:")
        
        unique_clusters = [c for c in sorted(np.unique(labels)) if c >= 0]
        
        for cluster in unique_clusters:
            cluster_indices = np.where(labels == cluster)[0]
            cluster_provinces = provinces[cluster_indices]
            
            print(f"\nCluster {cluster+1} ({len(cluster_provinces)} provinces):")
            print(f"Provinces: {', '.join(cluster_provinces)}")
            
            cluster_data = df_agg.iloc[cluster_indices]
            
            characteristics = []
            
            crime_level = "tinggi" if cluster_data['avg_pidana'].mean() > overall_means['avg_pidana'] else "rendah"
            crime_trend = "meningkat" if cluster_data['trend_pidana'].mean() > 0 else "menurun"
            characteristics.append(f"Tingkat kriminalitas {crime_level} dengan tren {crime_trend}")
            
            education_level = "tinggi" if cluster_data['avg_pendidikan'].mean() > overall_means['avg_pendidikan'] else "rendah"
            education_trend = "meningkat" if cluster_data['trend_pendidikan'].mean() > 0 else "menurun"
            characteristics.append(f"Tingkat pendidikan {education_level} dengan tren {education_trend}")
            
            unemployment_level = "tinggi" if cluster_data['avg_pengangguran'].mean() > overall_means['avg_pengangguran'] else "rendah"
            unemployment_trend = "meningkat" if cluster_data['trend_pengangguran'].mean() > 0 else "menurun"
            characteristics.append(f"Tingkat pengangguran {unemployment_level} dengan tren {unemployment_trend}")
            
            income_level = "tinggi" if cluster_data['avg_pendapatan'].mean() > overall_means['avg_pendapatan'] else "rendah"
            income_trend = "meningkat" if cluster_data['trend_pendapatan'].mean() > 0 else "menurun"
            characteristics.append(f"Tingkat pendapatan {income_level} dengan tren {income_trend}")
            
            cluster_name = ""
            if income_level == "tinggi" and education_level == "tinggi":
                cluster_name = "Wilayah Berkembang Maju"
            elif income_level == "tinggi" and education_level == "rendah":
                cluster_name = "Wilayah Ekonomi Berbasis Sumber Daya"
            elif crime_level == "rendah" and unemployment_level == "rendah":
                cluster_name = "Wilayah Tradisional Stabil"
            elif unemployment_level == "tinggi" and crime_level == "tinggi":
                cluster_name = "Wilayah dengan Tantangan Sosial"
            else:
                cluster_name = "Wilayah Transisi"
                
            print(f"Karakteristik: {cluster_name}")
            for char in characteristics:
                print(f"- {char}")
            
            print("Nilai rata-rata:")
            for feature in basic_features:
                feature_mean = cluster_data[feature].mean()
                feature_diff = ((feature_mean - overall_means[feature]) / overall_means[feature]) * 100
                print(f"- {feature}: {feature_mean:.2f} ({feature_diff:+.1f}% dari rata-rata)")
        
        if -1 in labels:
            noise_indices = np.where(labels == -1)[0]
            noise_provinces = provinces[noise_indices]
            print(f"\nNoise Points ({len(noise_provinces)} provinces):")
            print(f"Provinces: {', '.join(noise_provinces)}")
            print("Karakteristik: Wilayah dengan pola unik yang tidak terklasifikasi dalam cluster manapun")


    def evaluate_clustering(data, labels, method_name):
        if -1 in labels:
            valid_indices = labels != -1
            if sum(valid_indices) < 2:
                print(f"\n{method_name} Metrics:")
                print("Could not calculate metrics - insufficient non-noise points.")
                return
            filtered_data = data[valid_indices]
            filtered_labels = labels[valid_indices]
        else:
            filtered_data = data
            filtered_labels = labels
            
        if len(np.unique(filtered_labels)) < 2:
            print(f"\n{method_name} Metrics:")
            print("Could not calculate metrics - need at least 2 clusters.")
            return
            
        try:
            silhouette = silhouette_score(filtered_data, filtered_labels)
            ch_score = calinski_harabasz_score(filtered_data, filtered_labels)
            
            print(f"\n{method_name} Metrics:")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Calinski-Harabasz Index: {ch_score:.2f}")
            print(f"Number of clusters: {len(np.unique(filtered_labels))}")
            
            unique, counts = np.unique(filtered_labels, return_counts=True)
            for cluster, count in zip(unique, counts):
                print(f"Cluster {cluster+1}: {count} provinces")
            
            if -1 in labels:
                noise_count = sum(labels == -1)
                print(f"Noise points: {noise_count} provinces")
                
            interpret_clusters(data, labels, method_name)
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    kmeans = KMeans(n_clusters=optimal_k_pca, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(df_pca)


    df_agg['kmeans_cluster'] = kmeans_labels + 1

    # Standarkan format huruf kecil dan hapus spasi
    gdf["state"] = gdf["state"].str.lower().str.strip()
    df_agg["Provinsi"] = df_agg["Provinsi"].str.lower().str.strip()

    # Merge berdasarkan nama provinsi
    gdf_merged = gdf.merge(df_agg, how="left", left_on="state", right_on="Provinsi")
    gdf_merged.rename(columns={'kmeans_cluster': 'Cluster'}, inplace=True)
    gdf_merged = gdf_merged.dropna()
    resultTable = pd.DataFrame({'Provinsi':gdf_merged['Provinsi'], 'Cluster':gdf_merged['Cluster']})
    resultTable = resultTable.dropna()

    # --- Output Tabel ---
    st.subheader("Tabel Hasil Clustering")
    st.dataframe(resultTable[['Provinsi', 'Cluster']].sort_values('Cluster'))

    # --- Output Map ---
    # gdf_merged['Cluster'] = gdf_merged['Cluster'].astype(str)  # MODIFIKASI

    # # Mapping warna diskret (boleh diganti sesuai selera)
    # color_discrete_map = {
    #     '1': '#1f77b4',
    #     '2': '#ff7f0e',
    #     '3': '#2ca02c',
    # }

    # st.subheader("Peta Cluster")
    # fig = px.choropleth_mapbox(
    #     gdf_merged,
    #     geojson=gdf_merged.geometry,
    #     locations=gdf_merged.index,
    #     color='Cluster',
    #     hover_name='Provinsi',
    #     hover_data={'Cluster': True},
    #     mapbox_style="carto-positron",
    #     center={"lat": -2.5, "lon": 118},
    #     zoom=4.3,
    #     opacity=0.8,
    #     height=700,
    #     color_discrete_map=color_discrete_map,  # MODIFIKASI
    #     category_orders={'Cluster': ['1', '2', '3']}  # MODIFIKASI
    # )
    # fig.update_layout(
    #     margin={"r":0,"t":0,"l":0,"b":0},
    #     legend_title_text='Cluster'  # MODIFIKASI (opsional untuk label legend)
    # )
    # st.plotly_chart(fig, use_container_width=True)


    import json

    # Pastikan kolom Cluster adalah int
    gdf_merged['Cluster'] = gdf_merged['Cluster'].astype(int)
    gdf_merged['Cluster_str'] = gdf_merged['Cluster'].astype(str)

    # Warna khusus untuk cluster
    color_discrete_map = {
        '1': '#e74c3c',   # Merah
        '2': '#f1c40f',   # Kuning
        '3': '#2ecc71',   # Hijau
    }

    # Convert to geojson
    geojson_data = json.loads(gdf_merged.to_json())

    st.subheader("Peta Hasil Clustering")

    # Buat map
    fig = px.choropleth_mapbox(
        gdf_merged,
        geojson=geojson_data,
        locations=gdf_merged.index,
        color='Cluster_str',
        hover_name='Provinsi',
        hover_data={'Cluster': True},
        mapbox_style="carto-positron",
        center={"lat": -2.5, "lon": 118},
        zoom=4.3,
        opacity=0.8,
        height=700,
        color_discrete_map=color_discrete_map,
        category_orders={'Cluster_str': ['1', '2', '3']}
    )

    # Update layout: pindah legend ke atas dan horizontal
    fig.update_layout(
        legend=dict(
            orientation="h",  # horizontal
            yanchor="bottom",
            y=1.05,           # posisi di atas plot
            xanchor="center",
            x=0.5,
            font=dict(
                size=16       # ukuran font lebih besar
            )
        ),
        legend_title_text='Cluster',
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)



else:
    st.warning("Pilih minimal 1 fitur untuk melakukan clustering.")