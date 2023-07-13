# Importamos Librerias
import streamlit as st
import pandas as pd
from PIL import Image

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

st.title('Métricas')

st.header('Métricas de los modelos')
st.write('Las métricas son herramientas fundamentales para evaluar el rendimiento de un modelo de Machine Learning en un conjunto de datos de prueba, con el objetivo de medir su capacidad de generalización. Esta habilidad se refiere a la capacidad del modelo de hacer predicciones precisas y útiles en nuevos datos que no fueron utilizados durante su entrenamiento. Las métricas de evaluación de modelos permiten comparar diferentes modelos y seleccionar el mejor para abordar un problema específico. Algunas métricas comunes incluyen:')
st.write("""
            * Accuracy (Exactitud)
            * Precision (Precisión)
            * Recall (Recuperación)
            * F1-score (Puntuación F1)
            * Cross Validation (Validación cruzada)
            * Class Prediction Error (Error de predicción de clase)
            * Confusion Matrix (Matriz de confusión)
            * Precision-Recall Curves (Curvas de precisión-recuperación)
            * ROC Curves (Curvas ROC)""")    

model_FP = ['GB', 'ET','B','RF','DT'] 
select_Model = st.selectbox('Puede elegir uno de los modelos para evaluar sus métricas:', model_FP)

metric_GB = {
            'Accuracy': 0.9446,
            'Precision': 0.9252,
            'Recall': 0.9341,
            'F1-score': 0.9291,
            'Cross Validation' : "0.9442 ± 0.0075"
            }
metric_ET = {
            'Accuracy' : 0.9402,
            'Precision' : 0.9145,
            'Recall' : 0.9350,
            'F1-score' : 0.9237,
            'Cross Validation' : "0.9424 ± 0.0048"                
            }
metric_B = {
            'Accuracy': 0.9374,
            'Precision': 0.9311,
            'Recall': 0.9264,
            'F1-score': 0.9286,
            'Cross Validation' : "0.9327 ± 0.0066" 
            }
metric_RF = {
            'Accuracy': 0.9358,
            'Precision': 0.9167,
            'Recall': 0.9300,
            'F1-score': 0.9224,
            'Cross Validation' : "0.9369 ± 0.0057"  
            }
metric_DT = {
            'Accuracy': 0.9269,
            'Precision': 0.8990,
            'Recall': 0.9293,
            'F1-score': 0.9125,
            'Cross Validation' : "0.9234 + 0.0089"
            }

if select_Model == 'GB':
    df = pd.DataFrame.from_dict(metric_GB, orient='index')
    df.columns = ['Valor']
    df = df.T  # Transpone el DataFrame para que las métricas estén en las filas        
    st.table(df)
    col1, col2 = st.columns(2)
    with col1:
        imagen = Image.open("./images/class_pred_GB.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

        imagen = Image.open("./images/prec_recall_GB.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

    with col2:
        imagen = Image.open("./images/conf_GB.png")
        imagen = imagen.resize((700, 500))       
        st.image(imagen)

        imagen = Image.open("./images/ROC_GB.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)
        


elif select_Model == 'ET':
    df = pd.DataFrame.from_dict(metric_ET, orient='index')
    df.columns = ['Valor']
    df = df.T  # Transpone el DataFrame para que las métricas estén en las filas        
    st.table(df)        
    col1, col2 = st.columns(2)
    with col1:
        imagen = Image.open("./images/class_pred_ET.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

        imagen = Image.open("./images/prec_recall_ET.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)


    with col2:
        imagen = Image.open("./images/conf_ET.png")
        imagen = imagen.resize((700, 500))       
        st.image(imagen)


        imagen = Image.open("./images/ROC_ET.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)


elif select_Model == 'B':
    df = pd.DataFrame.from_dict(metric_B, orient='index')
    df.columns = ['Valor']
    df = df.T  # Transpone el DataFrame para que las métricas estén en las filas        
    st.table(df)          
    col1, col2 = st.columns(2)
    with col1:
        imagen = Image.open("./images/class_pred_B.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)
        
        imagen = Image.open("./images/prec_recall_B.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)
    

    with col2:
        imagen = Image.open("./images/conf_B.png")
        imagen = imagen.resize((700, 500))       
        st.image(imagen)

        imagen = Image.open("./images/ROC_B.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

elif select_Model == 'RF':
    df = pd.DataFrame.from_dict(metric_RF, orient='index')
    df.columns = ['Valor']
    df = df.T  # Transpone el DataFrame para que las métricas estén en las filas        
    st.table(df)            
    col1, col2 = st.columns(2)
    with col1:
        imagen = Image.open("./images/class_pred_RF.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

        imagen = Image.open("./images/prec_recall_RF.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)


    with col2:
        imagen = Image.open("./images/conf_RF.png")
        imagen = imagen.resize((700, 500))       
        st.image(imagen)

        imagen = Image.open("./images/ROC_RF.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)


else:    
    df = pd.DataFrame.from_dict(metric_DT, orient='index')
    df.columns = ['Valor']
    df = df.T  # Transpone el DataFrame para que las métricas estén en las filas        
    st.table(df)   
    col1, col2 = st.columns(2)
    with col1:
        imagen = Image.open("./images/class_pred_DT.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

        imagen = Image.open("./images/prec_recall_DT.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)

    with col2:
        imagen = Image.open("./images/conf_DT.png")
        imagen = imagen.resize((700, 500))       
        st.image(imagen)

        imagen = Image.open("./images/ROC_DT.png")
        imagen = imagen.resize((700, 500))            
        st.image(imagen)
    