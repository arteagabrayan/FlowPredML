# Importamos Librerias
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import *
from sklearn.decomposition import PCA   
from sklearn.model_selection import train_test_split

# Ocultar header y footer que vienen por defecto
st.markdown("""
<style>
.css-nqowgj.edgvbvh3 {visibility: hidden;}           
.css-h5rgaw.egzxvld1 {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Agregamos CSS personalizado para cambiar el color del recuadro a naranja
st.markdown(
    """
    <style>
    div[data-baseweb="select"]>div:first-child {border-color: orange !important;}
    </style>
    """, unsafe_allow_html=True
)

st.title('Análisis de datos') 

# Leer datos
path_data = 'data' 
path_sampledata = path_data+"/12DB_6FP.csv"
data = pd.read_csv(path_sampledata)
name_clases = {0:"DB",1:"SS",2:"SW",3:"A",4:"I",5:"B"}
d = data.copy()
d['FlowPattern'] = d['FlowPattern'].replace(name_clases)
features_df = d.drop(['FlowPattern'], axis=1)
labels = d['FlowPattern'] 
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, 
                                                    test_size=0.20, random_state=1, stratify=labels)

# Trabajo de datos para PCA
scalar = preprocessing.StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(features_df)) 
# PCA con varianza acumulada
pca = PCA()
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)
pc_options = ['PC'+str(i) for i in range(1,10)]
data_pca = pd.DataFrame(scaled_data,columns=pc_options)
data_pca_labels = pd.concat([data_pca,d["FlowPattern"]],axis=1)

col1, col2 = st.columns(2)
with col1:
    st.header('Distribución')
    st.write("La distribución de datos se refiere a cómo se encuentran distribuidos o dispersos los datos dentro del conjunto de datos. En este caso, se puede observar que hay una gran cantidad de datos correspondientes al flujo intermitente (I), mientras que la cantidad de datos correspondientes al flujo de burbujas (B) es reducida.")
    def crear_grafico():
        cantidades = [153, 582, 816, 1093, 1664, 4721]
        nombres = ["B", "SS", "DB", "SW", "A", "I"]
        color_palette_list = ['#C1F0F6', '#007ACD', "#FFD97D", "#60D394", "#EE6055", '#0EBFE9']

        def label_formatter(pct, cantidad):
            cantidad_str = '{:.0f}'.format(cantidad)
            return '{:.0f}%\n{}'.format(pct, cantidad_str)

        fig, ax = plt.subplots(figsize=(13.45, 8))
        ax.pie(cantidades, labels=nombres, colors=color_palette_list[0:], 
            autopct=lambda pct: label_formatter(pct, cantidades[int(pct/100.*len(cantidades))]), 
            shadow=False, startangle=0,   
            pctdistance=1.2, labeldistance=1.4)
        ax.axis('equal')
        ax.figure.subplots_adjust(right=0.8)

        return fig

    st.pyplot(crear_grafico())

with col2:
    st.header('Matriz de correlación')
    st.write("La matriz de correlación es una herramienta estadística que muestra los coeficientes de correlación entre todas las posibles combinaciones de variables dentro de un conjunto de datos. Esto permite identificar patrones y tendencias, así como las variables que están más estrechamente relacionadas entre sí.")
    ################# MATRIZ DE CORRELACION #################
    def generar_matriz_correlacion(dataframe):
        corr = dataframe.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, cbar=False)
        ax.set_title('Matriz de correlación')
        return fig
    
    fig = generar_matriz_correlacion(d)
    st.pyplot(fig)

st.header('Análisis por característica')
st.write("A continuación, se presentan tres tipos de gráficos de distribución de datos para cada característica: histogramas, diagramas de caja y diagramas de violín. Estos gráficos permiten visualizar de manera efectiva cómo se distribuyen los datos en cada variable.")    
feature_selec = st.selectbox('Seleccionar característica a analizar', features_df.columns)

col1, col2, col3 = st.columns(3)
with col1:
    ################# HISTOGRAMA DE CARACTERISTICAS #################
    def generar_histograma(dataframe, columna):
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.histplot(dataframe[columna], bins=20)
        ax.set_xlabel(columna)
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de {}'.format(columna))
        st.pyplot(fig)
                
    generar_histograma(d, feature_selec)
    
with col2:        
    ################# DIAGRAMAS DE CAJA #################
    def boxplot(df, column):
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.boxplot(data=df, x="FlowPattern", y=column)
        ax.set_xlabel("FlowPattern")
        ax.set_ylabel(column)
        ax.set_title(f"{column} vs. FlowPattern")
        st.pyplot(fig)

    boxplot(d, feature_selec)

with col3:        
################# DIAGRAMAS DE VIOLIN #################
    def boxplot(df, column):
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.violinplot(data=df, x="FlowPattern", y=column)
        ax.set_xlabel("FlowPattern")
        ax.set_ylabel(column)
        ax.set_title(f"{column} vs. FlowPattern")
        st.pyplot(fig)

    boxplot(d, feature_selec)


st.header('Análisis de Componentes Principales')
st.write("En los gráficos del análisis de componentes principales (PCA) se muestra la varianza acumulada explicada por cada componente principal, lo que permite comprender cuánto contribuye cada componente a la variabilidad total del conjunto de datos. Además, se pueden seleccionar los componentes principales individuales para observar cómo afectan a las nueve características principales. También se incluye un gráfico de dispersión que muestra la relación entre dos componentes principales seleccionados.")

col1, col2, col3 = st.columns(3)
with col1:
################# PCA #################
    def plot_pca_variance(scaled_d, pca):
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.plot(list(range(scaled_d.shape[1])), np.cumsum(pca.explained_variance_ratio_), 'r-')
        ax.set_xlabel('Número de componente principal')
        ax.set_ylabel('Varianza acumulada explicada')
        ax.set_title('Varianza acumulada explicada con PCA')
        return fig

    pca = PCA()
    pca.fit(scaled_data)

    fig = plot_pca_variance(scaled_data, pca)
    st.pyplot(fig)

with col2:
    def plot_pc_loading(component):
        loadings = pd.DataFrame(
            data=pca.components_.T * np.sqrt(pca.explained_variance_), 
            columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
            index=X_train.columns
        )
        pc_loadings = abs(loadings).sort_values(by=component, ascending=False)[[component]]
        pc_loadings = pc_loadings.reset_index()
        pc_loadings.columns = ['Feature', 'CorrelationWithPC'+component]
        fig, ax = plt.subplots(figsize=(9,8))
        ax.bar(x=pc_loadings['Feature'], height=pc_loadings['CorrelationWithPC'+component], color="#87CEEB")
        ax.set_title('Impacto de las características - '+component, size=20)
        ax.tick_params(axis='x', labelrotation=90)
        return fig        
    
    selected_pc = st.selectbox('Seleccionar componente principal:', pc_options)
    fig = plot_pc_loading(selected_pc)
    st.pyplot(fig)

    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca,columns=pc_options)
    data_pca_labels = pd.concat([data_pca,d["FlowPattern"]],axis=1)

with col3:
    def plot_pca_jointplot(x_pca, y_pca):            
        jointplot = sns.jointplot(data=data_pca_labels, x=x_pca, y=y_pca, hue="FlowPattern")
        return jointplot.fig

    col1, col2 = st.columns(2)
    with col1:
        selected_pc1 = st.selectbox('Seleccionar componente X:', pc_options)
    with col2:
        selected_pc2 = st.selectbox('Seleccionar componente Y:', pc_options)

    fig = plot_pca_jointplot(selected_pc1, selected_pc2)
    st.pyplot(fig)

st.header("Importancia de características")
st.write("Los modelos de aprendizaje automático entrenados utilizan características específicas que proporcionan información relevante para predecir una clase en particular. A continuación, se presentan las características más relevantes detectadas por cada modelo.")

model_feature = st.selectbox('Seleccionar un modelo:', ['DT', 'ET','GB','RF'] )
if model_feature == 'DT':
    st.image("./images/feature_DT.png")
elif model_feature == 'ET':
    st.image("./images/feature_ET.png")
elif model_feature == 'GB':
    st.image("./images/feature_GB.png")
else:
    st.image("./images/feature_RF.png")