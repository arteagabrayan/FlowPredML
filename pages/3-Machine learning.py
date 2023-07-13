# Importamos Librerias
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
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

st.title("Machine learning")

#Imagen de inicio
imagen = Image.open("./images/flow_pattern.png")

st.subheader("Machine Learning modelo seleccionado")
imagen = Image.open("./images/cerebro.png")
imagen = imagen.resize((1600, 900))     
st.image(imagen)

st.subheader("Algoritmo de Machine Learning")
st.write("Definición del algoritmo implementado para predecir los patrones de flujo.")
model_FP = ['GB', 'ET', 'B', 'RF', 'DT'] 
selected_Model = st.selectbox('Seleccionar un algoritmo de Machine Learning:', model_FP)

model_description = {
    'GB': "Gradient Boosting (GB) es un método basado en árboles de decisión que combina múltiples árboles iterativamente para mejorar la generalización.",
    'ET': "Extra Trees (ET) es un algoritmo de ensemble que utiliza árboles de decisión aleatorios para obtener predicciones precisas en clasificación, regresión y selección de características.",
    'B': "Bagging Classifier (B) es un método de ensamblaje que combina múltiples clasificadores independientes entrenados en diferentes subconjuntos de datos para mejorar la precisión y estabilidad.",
    'RF': "Random Forest (RF) es un algoritmo supervisado que construye múltiples árboles de decisión aleatorios y combina sus predicciones para obtener resultados precisos y estables en clasificación y regresión.",
    'DT': "Árbol de Decisión (DT) es un método de aprendizaje supervisado que utiliza un modelo de árbol para predecir la clase o valor de una variable objetivo basándose en múltiples características."
}

st.write(model_description[selected_Model])

st.subheader("Características de entrada")
features = ['Velocidad superficial del líquido (Vsl)', 'Velocidad superficial del gas (Vsg)', 'Viscosidad del líquido (VisL)', 'Viscosidad del gas (VisG)', 'Densidad del líquido (DenL)', 'Densidad del gas (DenG)', 'Tensión superficial (ST)', 'Ángulo de inclinación tubería (Ang)', 'Diámetro de la tubería (ID)']
st.write("A continuación, ingrese los valores de las características que serán utilizadas para la clasificación de patrones de flujo en los modelos de Machine Learning:")        

def user_input_parameters():
    inputs = {}
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        with st.container():
            if i % 3 == 0:
                cols = st.columns(3)
            inputs[feature] = cols[col].text_input(feature)
    data_features = {
            'Vsl' : inputs[features[0]],
            'Vsg' : inputs[features[1]],
            'VisL' : inputs[features[2]],
            'VisG' : inputs[features[3]],
            'DenL': inputs[features[4]],
            'DenG' : inputs[features[5]],
            'ST' : inputs[features[6]],
            'Ang' : inputs[features[7]],
            'ID' : inputs[features[8]]
            }
    features_df = pd.DataFrame(data_features, index = [0])
    return features_df


df = user_input_parameters()
###########################################################
###########################################################
st.subheader("Modelo " + selected_Model)

# Crear un nuevo DataFrame con una fila adicional 'Valor'
df = df.T.reset_index()
df.columns = ['Característica', 'Valor']
df = df.set_index('Característica').T

st.table(df)

# Crear dos botones 'PREDECIR' y 'LIMPIAR' en la misma fila
predict_button, clear_button = st.columns(2)
predict_clicked = predict_button.button('PREDECIR')

if predict_clicked:
    # Validar que todos los campos contengan valores numéricos
    for value in df.values.flatten():
        if not value or not value.isdigit():
            st.warning("Por favor, complete todos los datos con valores numéricos antes de hacer la predicción.")
            break
    else:
        prediction = None

        if predict_on == "AZURE": 
            # PREDICTION IN MICROSOFT AZURE
            datos_a_testear = df.to_dict(orient='records')[0]
            try:
                datos_a_testear['selected_model'] = selected_Model
                response = requests.post(url, json=datos_a_testear)

                if response.status_code == 200:
                    prediction = [response.json()['prediccion']]
                else:
                    st.warning('Error en la solicitud:'+str(response.text))
                    print('Error en la solicitud:', response.text)
            except:
                st.warning("Error al realizar la predicción. Verificar la conexión con la API en Azure")
                print("Error al realizar la predicción. Verificar la conexión con la API en Azure")
        else:
            # PREDICTION IN STREAMLIT
            if selected_Model == 'GB':
                prediction = GB.predict(df)
            elif selected_Model == 'ET':
                prediction = ET.predict(df)
            elif selected_Model == 'B':
                prediction = B.predict(df)
            elif selected_Model == 'RF':
                prediction = RF.predict(df)
            elif selected_Model == 'DT':
                prediction = DT.predict(df)

        # Crear un diccionario para asociar las predicciones con sus descripciones
        prediction_descriptions = {
            'DB': 'Flujo de burbujas dispersas (DB)',
            'SS': 'Flujo estratificado uniforme (SS)',
            'SW': 'Flujo estratificado ondulado (SW)',
            'A': 'Flujo anular (A)',
            'I': 'Flujo intermitente (I)',
            'B': 'Flujo de burbujas (B)'
        }

        # Mostrar la descripción completa de la predicción
        st.success(prediction_descriptions[prediction[0]])

        if prediction[0] in ["I", "A"]:
            st.warning("Alerta: Presta atención a posibles fallos en la tubería.") 