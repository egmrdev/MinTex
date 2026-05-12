"""
MinTexto_v4.py 
Formatos soportados: .txt  |  .pdf  |  .docx
"""

import re
import sys
import string
import pathlib
import numpy as np

# PARAMETROS DEL MODELO  (ajusta estos valores segun tu corpus)

VENTANA        = 2      # Cuantas palabras a cada lado se consideran "vecinas"
DIMENSIONES    = 50     # Tamaño de cada vector de palabra (embedding)
TASA_APREND    = 0.01   # Que tan grande es cada paso de ajuste (learning rate)
EPOCAS         = 500    # Cuantas veces recorremos todos los pares de palabras
TOP_N          = 5      # Cuantos resultados mostrar en similitud y prediccion
SEMILLA        = 42     # Para que los resultados sean reproducibles

np.random.seed(SEMILLA)

# Cargar stop-words desde archivo externo
_sw_ruta = pathlib.Path("stopwords-es.txt")
if _sw_ruta.exists():
    STOP_WORDS = {linea.strip().lower() for linea in _sw_ruta.read_text(encoding="utf-8").splitlines() if linea.strip()}
else:
    # Fallback si no existe el archivo
    STOP_WORDS = {"a","al","de","del","el","en","es","la","las","lo","los","no","o","para","por","que","se","si","su","un","una","y"}
    print("[AVISO] stopwords-es.txt no encontrado, usando lista minima")


# PASO 1: LEER EL ARCHIVO
# Detecta automaticamente si es .txt, .pdf o .docx
# y extrae el texto sin formato como una cadena de caracteres.

def leer_archivo(ruta_str: str) -> str:
    ruta = pathlib.Path(ruta_str)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: '{ruta}'")

    ext = ruta.suffix.lower()

    if ext == ".txt":
        texto = ruta.read_text(encoding="utf-8")

    elif ext == ".pdf":
        import pdfplumber
        paginas = []
        with pdfplumber.open(ruta) as pdf:
            for pagina in pdf.pages:
                contenido = pagina.extract_text()
                if contenido:
                    paginas.append(contenido)
        texto = "\n".join(paginas)

    elif ext == ".docx":
        from docx import Document
        doc = Document(ruta)
        texto = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError(f"Formato no soportado '{ext}'. Usa .txt, .pdf o .docx")

    print(f"[1] Archivo leido ({ext})  ->  {len(texto):,} caracteres en total")
    return texto


# PASO 2: PREPROCESAMIENTO
# El texto se limpia para que el modelo solo vea palabras utiles:
#   - Convertimos todo a minusculas
#   - Eliminamos puntuacion (comas, puntos, etc.)
#   - Quitamos las stop-words (palabras sin significado util)
#   - Descartamos palabras de un solo caracter

def preprocesar(texto: str) -> list:
    # Minusculas
    texto = texto.lower()

    # Reemplazar signos de puntuacion por espacio
    texto = re.sub(f"[{re.escape(string.punctuation)}]", " ", texto)

    # Quitar espacios multiples
    texto = re.sub(r"\s+", " ", texto).strip()

    # Dividir en palabras y filtrar stop-words y palabras muy cortas
    palabras = [
        p for p in texto.split()
        if p not in STOP_WORDS and len(p) > 1
    ]

    print(f"[2] Tokens utiles encontrados: {len(palabras)}")
    return palabras


# PASO 3: CONSTRUIR EL VOCABULARIO
# El vocabulario es un diccionario que asigna un numero unico
# a cada palabra diferente que aparece en el corpus.

# Ejemplo:
#   mineria -> 0
#   texto   -> 1
#   modelo  -> 2
#   ...


def construir_vocabulario(palabras: list) -> tuple:
    # Asignamos indices empezando desde 0
    vocab = {}
    for palabra in palabras:
        if palabra not in vocab:
            vocab[palabra] = len(vocab)   # indice = posicion actual

    # Vocabulario inverso: de numero a palabra (util para imprimir resultados)
    vocab_inverso = {indice: palabra for palabra, indice in vocab.items()}

    print(f"[3] Vocabulario: {len(vocab)} palabras unicas")
    return vocab, vocab_inverso


# PASO 4: ONE-HOT ENCODING
# Representa cada palabra como un vector de ceros con un solo 1
# en la posicion que le corresponde en el vocabulario.

# Ejemplo con vocabulario de 5 palabras:
#   mineria (indice 0)  -> [1, 0, 0, 0, 0]
#   texto   (indice 1)  -> [0, 1, 0, 0, 0]
#   modelo  (indice 2)  -> [0, 0, 1, 0, 0]

def one_hot(indice: int, tamaño_vocab: int) -> np.ndarray:
    vector = np.zeros(tamaño_vocab)
    vector[indice] = 1.0
    return vector


# PASO 5: GENERAR PAREJAS SKIP-GRAM
# Para cada palabra del corpus, generamos pares (objetivo, contexto).
# "Contexto" son las palabras que aparecen cerca dentro de la ventana.
#
# Ejemplo con ventana=2 sobre la frase: "mineria texto descubre patrones ocultos"
#
#   Palabra objetivo: "descubre" (posicion 2)
#   Vecinos dentro de la ventana: "mineria", "texto", "patrones", "ocultos"
#   Parejas generadas: (descubre, mineria), (descubre, texto),
#                      (descubre, patrones), (descubre, ocultos)

def generar_parejas(secuencia: list, ventana: int) -> list:
    parejas = []
    total = len(secuencia)

    for posicion, palabra_objetivo in enumerate(secuencia):
        # Definir los limites de la ventana
        inicio = max(0, posicion - ventana)
        fin    = min(total, posicion + ventana + 1)

        # Agregar un par por cada vecino
        for vecino in range(inicio, fin):
            if vecino != posicion:   # no emparejamos la palabra consigo misma
                parejas.append((palabra_objetivo, secuencia[vecino]))

    print(f"[4] Parejas (objetivo, contexto) generadas: {len(parejas)}")
    return parejas


# PASO 6: ENTRENAMIENTO DEL MODELO SKIP-GRAM
# La red neuronal tiene dos matrices de pesos:
#
#   P_entrada  (vocab x dimensiones)
#   -> Convierte una palabra objetivo en su vector de embedding
#
#   P_salida   (dimensiones x vocab)
#   -> Convierte el embedding en probabilidades sobre todo el vocabulario
#
# Por cada par (objetivo, contexto) hacemos:
#
#   1. FORWARD PASS:
#      - Extraemos el embedding de la palabra objetivo de P_entrada
#      - Multiplicamos por P_salida para obtener puntuaciones
#      - Aplicamos softmax para convertir en probabilidades
#
#   2. CALCULO DEL ERROR:
#      - Medimos que tan equivocada esta la prediccion (cross-entropy)
#
#   3. BACKPROPAGATION:
#      - Calculamos cuanto contribuyo cada peso al error
#
#   4. ACTUALIZACION:
#      - Ajustamos los pesos en la direccion que reduce el error

def softmax(puntuaciones: np.ndarray) -> np.ndarray:
    """Convierte puntuaciones en probabilidades que suman 1."""
    # Restamos el maximo para estabilidad numerica (evita numeros muy grandes)
    exp = np.exp(puntuaciones - puntuaciones.max())
    return exp / exp.sum()


def entrenar(vocab: dict, parejas: list) -> tuple:
    V = len(vocab)    # tamaño del vocabulario
    E = DIMENSIONES   # tamaño de cada embedding

    # Inicializamos los pesos con valores aleatorios pequeños
    P_entrada = np.random.uniform(-0.5, 0.5, (V, E))   # (palabras x dimensiones)
    P_salida  = np.random.uniform(-0.5, 0.5, (E, V))   # (dimensiones x palabras)

    print(f"\n[5] Iniciando entrenamiento...")
    print(f"    Vocabulario    : {V} palabras")
    print(f"    Dimensiones    : {E}")
    print(f"    Pares a revisar: {len(parejas)}")
    print(f"    Epocas         : {EPOCAS}")
    print(f"    Tasa aprendizaje: {TASA_APREND}\n")

    ancho = len(str(EPOCAS))

    for epoca in range(EPOCAS):
        perdida_total = 0.0

        for idx_objetivo, idx_contexto in parejas:

            # ── FORWARD PASS ──────────────────────────────────────────
            # 1. Obtener el embedding de la palabra objetivo
            #    (simplemente tomamos la fila correspondiente de P_entrada)
            embedding = P_entrada[idx_objetivo]              # vector de E valores

            # 2. Calcular puntuaciones para cada palabra del vocabulario
            puntuaciones = P_salida.T @ embedding            # vector de V valores

            # 3. Convertir puntuaciones en probabilidades
            probabilidades = softmax(puntuaciones)           # vector de V valores, suma 1

            # ── ERROR ─────────────────────────────────────────────────
            # Medimos el error: -log(probabilidad de la palabra correcta)
            # Cuanto menor sea la perdida, mejor predice el modelo
            perdida_total += -np.log(probabilidades[idx_contexto] + 1e-9)

            # ── BACKPROPAGATION ───────────────────────────────────────
            # Calculamos cuanto se equivoco cada probabilidad
            error = probabilidades.copy()
            error[idx_contexto] -= 1.0   # la palabra correcta deberia tener prob=1

            # Gradiente de P_salida: como cambiar cada peso para reducir el error
            gradiente_P_salida = np.outer(embedding, error)

            # Gradiente del embedding: como cambiar el vector de la palabra objetivo
            gradiente_embedding = P_salida @ error

            # ── ACTUALIZACION (Gradient Descent) ──────────────────────
            P_salida          -= TASA_APREND * gradiente_P_salida
            P_entrada[idx_objetivo] -= TASA_APREND * gradiente_embedding

        # Mostrar progreso cada 100 epocas
        if epoca == 0 or (epoca + 1) % 100 == 0:
            perdida_promedio = perdida_total / len(parejas)
            print(f"  Epoca {epoca+1:{ancho}}/{EPOCAS}  |  Perdida promedio: {perdida_promedio:.4f}")

    # Guardamos los pesos aprendidos en archivos de texto
    np.savetxt("PEnt.txt", P_entrada)
    np.savetxt("PSal.txt", P_salida.T)
    print(f"\n  Pesos guardados en PEnt.txt y PSal.txt")

    return P_entrada, P_salida


# PASO 7: SIMILITUD COSENO
# Despues del entrenamiento, palabras con significados similares
# tendran vectores apuntando en direcciones parecidas.
#
# La similitud coseno mide el angulo entre dos vectores:
#   - Valor cercano a 1.0 = palabras muy similares
#   - Valor cercano a 0.0 = palabras sin relacion
#   - Valor cercano a -1  = palabras opuestas

def similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    norma = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norma) if norma > 0 else 0.0


def palabras_similares(palabra: str, vocab: dict, vocab_inv: dict,
                       P_entrada: np.ndarray) -> None:
    """Encuentra las palabras con el vector mas cercano al de 'palabra'."""
    if palabra not in vocab:
        print(f"  '{palabra}' no esta en el vocabulario.")
        return

    vector_objetivo = P_entrada[vocab[palabra]]

    # Calcular similitud contra todas las palabras del vocabulario
    similitudes = [
        (similitud_coseno(vector_objetivo, P_entrada[i]), vocab_inv[i])
        for i in range(len(vocab_inv))
        if i != vocab[palabra]
    ]

    # Ordenar de mayor a menor similitud
    similitudes.sort(reverse=True)

    print(f"\n  Palabras similares a '{palabra}':")
    print(f"  {'Palabra':<24} {'Similitud':>9}")
    print(f"  {'-'*24} {'-'*9}")
    for sim, pal in similitudes[:TOP_N]:
        print(f"  {pal:<24} {sim:>9.4f}")


# PASO 8: PREDICCION DE CONTEXTO
# Usamos los pesos entrenados para responder la pregunta:
# "Si la palabra objetivo es X, ¿que palabras es mas probable
#  encontrar cerca de ella en el texto?"
#
# Esto es exactamente lo que el modelo aprendio a hacer:
#   embedding(X) -> puntuaciones -> probabilidades -> top palabras

def predecir_contexto(palabra: str, vocab: dict, vocab_inv: dict,
                      P_entrada: np.ndarray, P_salida: np.ndarray) -> None:
    """Predice las palabras de contexto mas probables para 'palabra'."""
    if palabra not in vocab:
        print(f"  '{palabra}' no esta en el vocabulario.")
        return

    # Forward pass completo con los pesos ya entrenados
    embedding     = P_entrada[vocab[palabra]]   # vector de la palabra
    puntuaciones  = P_salida.T @ embedding      # puntuaciones para cada palabra
    probabilidades = softmax(puntuaciones)      # convertir a porcentajes

    # Armar lista de candidatos (excluir la misma palabra)
    candidatos = [
        (probabilidades[i], vocab_inv[i])
        for i in range(len(vocab_inv))
        if i != vocab[palabra]
    ]
    candidatos.sort(reverse=True)

    print(f"\n  Prediccion de contexto para '{palabra}':")
    print(f"  {'Palabra':<24} {'Probabilidad':>12}")
    print(f"  {'-'*24} {'-'*12}")
    for prob, pal in candidatos[:TOP_N]:
        print(f"  {pal:<24} {prob:>11.4%}")


# COMPLETO

def pipeline(ruta_archivo: str) -> None:
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  MinTexto v4")
    print(SEP)

    # ── Paso 1: Leer ────────────────────────────────────────────
    texto = leer_archivo(ruta_archivo)

    # ── Paso 2: Preprocesar ─────────────────────────────────────
    palabras = preprocesar(texto)
    print(f"     Muestra: {palabras[:8]}")

    # ── Paso 3: Vocabulario ─────────────────────────────────────
    vocab, vocab_inv = construir_vocabulario(palabras)

    # ── Paso 4: One-Hot (muestra visual) ────────────────────────
    print(f"\n     One-Hot encoding (muestra de 3 palabras):")
    for pal in list(vocab.keys())[:3]:
        vec = one_hot(vocab[pal], len(vocab))
        bits = " ".join(str(int(v)) for v in vec[:8])
        print(f"     '{pal}' -> [{bits} ...]")

    # ── Paso 5: Parejas skip-gram ───────────────────────────────
    secuencia = [vocab[p] for p in palabras]    # convertir palabras a numeros
    parejas   = generar_parejas(secuencia, VENTANA)

    print(f"     Muestra de parejas (objetivo -> contexto):")
    for obj, ctx in parejas[:5]:
        print(f"     '{vocab_inv[obj]}' -> '{vocab_inv[ctx]}'")

    # ── Paso 6: Entrenamiento ───────────────────────────────────
    P_entrada, P_salida = entrenar(vocab, parejas)

    # ── Paso 7: Similitud coseno ─────────────────────────────────
    print(f"\n{SEP}")
    print("[6] SIMILITUD COSENO  (palabras semanticamente cercanas)")
    print(SEP)
    for pal in palabras[:3]:
        palabras_similares(pal, vocab, vocab_inv, P_entrada)

    # ── Paso 8: Prediccion de contexto ───────────────────────────
    print(f"\n{SEP}")
    print("[7] PREDICCION DE CONTEXTO  (que palabras aparecen cerca)")
    print(SEP)
    for pal in palabras[:3]:
        predecir_contexto(pal, vocab, vocab_inv, P_entrada, P_salida)

    # ── Modo interactivo ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MODO INTERACTIVO  (escribe 'salir' para terminar)")
    print(SEP)
    while True:
        entrada = input("\n  Ingresa una palabra para predecir su contexto: ").strip().lower()
        if entrada in ("salir", "exit", "q"):
            break
        predecir_contexto(entrada, vocab, vocab_inv, P_entrada, P_salida)
        palabras_similares(entrada, vocab, vocab_inv, P_entrada)

    print(f"\n{SEP}")
    print("[OK] Pipeline finalizado.")
    print(SEP)


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else "Documento.txt"
    pipeline(archivo)
