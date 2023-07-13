# Importamos Librerias
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import statistics
import time
import requests

predict_on = "STREAMLIT" # Para predicción en Streamlit
#predict_on = "AZURE" # Para predicción en Azure

#develop = True # Para desarrollo local
develop = False # Para producción en Streamlit Cloud

if predict_on == "AZURE":
    if develop:
        # LOCALHOST
        with open('config.txt', 'r') as file:
            url = next(line.split('=')[1].strip().strip('"') for line in file if line.startswith('url'))
    else:
        # STREAMLIT CLOUD
        url = st.secrets["url"]
else:
    # PREDICTION IN STREAMLIT
    # Cargamos modelos
    with open('models/modelGB.pkl','rb') as gb:
        GB = pickle.load(gb)
    with open('models/modelDT.pkl','rb') as et:
        ET = pickle.load(et)
    with open('models/modelB.pkl','rb') as b:
        B = pickle.load(b)
    with open('models/modelRF.pkl','rb') as rf:
        RF = pickle.load(rf)
    with open('models/modelDT.pkl','rb') as dt:
        DT = pickle.load(dt)        

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

st.title("Machine learning simulación")

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

#Imagen de inicio
imagen = Image.open("./images/flow_pattern.png")

st.subheader("Machine Learning por votación")
imagen = Image.open("./images/cerebro1.jpg") 
imagen = imagen.resize((1000, 1000))     
st.image(imagen)

# leer el archivo CSV en un DataFrame
df_du = pd.read_csv(path_data+'/datos_unificados.csv')
# separar el DataFrame y la serie
X_test_p_ = df_du.iloc[:, :-1]
y_test_p_ = df_du.iloc[:, -1]

# Crea contenedores vacíos para las tablas
tabla_X_test_p = st.empty()
tabla_predicciones = st.empty()

def mostrar_prediccion(current_row):
    X_test_p = X_test_p_.iloc[[current_row], :]

    # Crear un nuevo DataFrame
    df = X_test_p.T.reset_index()
    df.columns = ['Característica', 'Valor']
    df = df.set_index('Característica').T
    tabla_X_test_p.table(df)
    
    predicciones_dict = {'GB': [], 'ET': [], 'B': [], 'RF': [], 'DT': []}
    
    # Realizar la predicción para cada modelo y almacenarla en el diccionario
    if predict_on == "AZURE":
        # PREDICTION IN MICROSOFT AZURE
        for model in predicciones_dict.keys():
            try:                
                datos_a_testear = X_test_p.to_dict(orient='records')[0]
                datos_a_testear['selected_model'] = model
                response = requests.post(url, json=datos_a_testear)

                if response.status_code == 200:
                    prediccion = response.json()['prediccion']
                    predicciones_dict[model].append(prediccion)
                else:
                    print(f"Error en la solicitud para el modelo {model}: {response.text}")
                    predicciones_dict[model].append(None)
            except:
                print(f"Error al realizar la predicción para el modelo {model}")
                predicciones_dict[model].append(None)
    else:
        # PREDICTION IN STREAMLIT
        predicciones_dict['GB'].append(GB.predict(X_test_p)[0])
        predicciones_dict['ET'].append(ET.predict(X_test_p)[0])
        predicciones_dict['B'].append(B.predict(X_test_p)[0])
        predicciones_dict['RF'].append(RF.predict(X_test_p)[0])
        predicciones_dict['DT'].append(DT.predict(X_test_p)[0])
    
    # convierte el diccionario en un DataFrame y lo muestra en Streamlit
    df_predicciones = pd.DataFrame(predicciones_dict)

    # agrega una columna Resultado que muestre el resultado por votación
    df_predicciones['Resultado de votación'] = df_predicciones.apply(lambda x: statistics.mode(x.tolist()), axis=1)

    # muestra el DataFrame de predicciones
    df = df_predicciones.T.reset_index()
    df.columns = ['Modelo', 'Predicción']
    df = df.set_index('Modelo').T
    tabla_predicciones.table(df)

# Inicializa el estado de la sesión para almacenar el valor de current_row y el estado del modo automático
if "current_row" not in st.session_state:
    st.session_state.current_row = 0
    st.session_state.auto_mode = False

# Muestra la primera predicción y las tablas de la fila de datos 1 al cargar la página
mostrar_prediccion(st.session_state.current_row)

# Agrega botones para controlar el modo automático
next_button = st.button("Siguiente muestra")
auto_button = st.button("Automático")
stop_button = st.button("Stop")

# Llama a la función para mostrar las predicciones y actualizar las tablas cuando se presione el botón
if next_button:
    st.session_state.current_row += 1

    # Si se excede el número de filas en X_test_p_, reinicia la variable contadora a 0
    if st.session_state.current_row >= len(X_test_p_):
        st.session_state.current_row = 0

    mostrar_prediccion(st.session_state.current_row)

if auto_button:
    st.session_state.auto_mode = True

if stop_button:
    st.session_state.auto_mode = False

# Actualiza las tablas automáticamente si el modo automático está activo
while st.session_state.auto_mode:
    time.sleep(2.0)
    st.session_state.current_row += 1

    if st.session_state.current_row >= len(X_test_p_):
        st.session_state.current_row = 0
    
    mostrar_prediccion(st.session_state.current_row)
