# Importamos Librerias
import streamlit as st
from PIL import Image

#Imagen de inicio
imagen = Image.open("./images/flow_pattern.png")

st.set_page_config(page_title="Home - FlowPredML", page_icon="./images/logoBDT.png", layout="centered") #layout="wide")#layout="centered")

# Ocultar header y footer que vienen por defecto
st.markdown("""
<style>
.css-nqowgj.edgvbvh3 {visibility: hidden;}           
.css-h5rgaw.egzxvld1 {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Aplicación de machine learning para predecir patrones de flujo en tuberias")

st.header("Los patrones de flujo")

imagen = imagen.resize((1400, 500))
st.image(imagen)

st.write("Un patrón de flujo describe cómo fluye un fluido a través de una tubería. ")
st.write("Seis patrones comunes en fluidos de tubería son:")
text = "**Flujo de burbujas dispersas (DB):** Gas fluye en pequeñas burbujas dispersas dentro de un líquido, moviéndose aleatoriamente y con distribución irregular, se encuentra en sistemas de aireación y procesos de mezcla. **Flujo estratificado uniforme (SS):** Dos fluidos diferentes, como líquido y gas, fluyen paralelos sin mezclarse, con una superficie de separación plana y uniforme, común en torres de destilación y sistemas de absorción. **Flujo estratificado ondulado (SW):** Similar al SS, pero la superficie de separación es ondulada debido a la acción de la gravedad y la fricción entre los fluidos, se encuentra en tuberías inclinadas y procesos de transferencia de calor. **Flujo anular (A):** Líquido fluye en el centro y gas en la periferia de la tubería, formando un anillo alrededor del líquido con superficie de separación curva, común en sistemas de inyección de gas y torres de absorción. **Flujo intermitente (I):** Diferentes patrones de flujo se alternan a lo largo de la tubería, común en tuberías con cambios de sección y en procesos de transporte de líquidos y gases. **Flujo de burbujas (B):** Gas fluye en grandes burbujas dentro del líquido, formando un patrón distintivo y moviéndose hacia la superficie de la tubería, se encuentra en sistemas de agitación y procesos de aireación"
items = text.split(".")
for item in items:
    st.write("- " + item + ".")
st.write("***Predecir patrones de flujo en tuberías es importante para prevenir fallas en el sistema, mejorar eficiencia de procesos y garantizar seguridad de operaciones; además, conocer los patrones de flujo es crucial para el diseño y optimización de sistemas de transporte de fluidos***")

st.header('Análisis del Modelo Canvas en MLOps')
st.write('El modelo Canvas para MLOps es una representación visual de alto nivel que ilustra las actividades principales, herramientas y flujos de trabajo requeridos para implementar y mantener modelos de aprendizaje automático. Este modelo ofrece una guía para planificar y llevar a cabo proyectos de aprendizaje automático de forma efectiva, garantizando la calidad y transparencia del modelo. El enfoque del modelo Canvas se centra en el ciclo de vida completo del modelo, desde la definición del problema hasta su implementación y monitoreo en producción.')
st.write('**A continuación, se presenta el modelo canvas propuesto:**')
st.image("./images/CANVAS.png",caption="Modelo CANVAS para MLOps.")
