#  MinTexto v5 — Skip-gram 

Aplicación web interactiva para aprender cómo funcionan los **embeddings de palabras** con el modelo **Skip-gram**, construido completamente con NumPy (sin PyTorch ni TensorFlow).


## ¿Qué hace la app?

-  **Carga tu propio texto** (`.txt`, `.pdf` o `.docx`) o escríbelo directamente
-  **Preprocesa** el corpus: limpieza, tokenización y eliminación de stopwords
-  **Genera parejas Skip-gram** con ventana de contexto configurable
-  **Entrena** la red neuronal en tiempo real con barra de progreso
-  **Similitud coseno** entre palabras (las más similares semánticamente)
-  **Predicción de contexto** (co-ocurrencia)
-  **Explorador interactivo**: escribe cualquier palabra y analízala

##  Instalación local

```bash
git clone https://github.com/tu-usuario/mintexto-deploy
cd mintexto-deploy
pip install -r requirements.txt
streamlit run app.py
```

##  Hiperparámetros configurables

| Parámetro | Rango | Descripción |
|---|---|---|
| Ventana de contexto | 1–5 | Cuántas palabras vecinas considera cada par |
| Dimensiones del embedding | 10–200 | Tamaño del vector que representa cada palabra |
| Tasa de aprendizaje | 0.001–0.1 | Velocidad de ajuste de los pesos |
| Épocas | 100–2000 | Cuántas veces recorre el corpus completo |
| Top N resultados | 3–15 | Cuántos resultados mostrar en las gráficas |

##  Stack técnico

- **Python 3.10+**
- `streamlit` — interfaz web
- `numpy` — algebra lineal y entrenamiento
- `plotly` — gráficas interactivas
- `pandas` — tablas de datos
- `pdfplumber` — lectura de PDFs
- `python-docx` — lectura de archivos Word
