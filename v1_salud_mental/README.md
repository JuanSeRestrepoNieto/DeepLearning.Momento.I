# Versión 1 — Clasificación tabular: salud mental estudiantil

> Trabajo previo, reemplazado por la versión de clasificación de imágenes que
> está en la raíz del repositorio. Se conserva como referencia.
>
> Para ejecutarlo hay que copiar estos archivos a la raíz (esperan encontrar
> `data/AI_SocialMedia_Student_Dataset.csv` y escribir en `figures/`).

Clasificación del **nivel de salud mental** de estudiantes (`Bajo` / `Medio` /
`Alto`) a partir de sus hábitos de vida y su uso de tecnología, con un MLP en
PyTorch.

## Dataset

**AI & Social Media Impact: Student Health & Grades** (Kaggle)
<https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades>

- **16.000** filas y **10** columnas en el archivo original, sin valores nulos.
- Tras eliminar `Student_ID` aparecen **1.000 filas duplicadas** (registros
  idénticos que solo se diferenciaban por el identificador). Al eliminarlas
  quedan **m = 15.000** muestras.
- Columnas: `Age`, `Gender`, `Education_Level`, `Daily_Social_Media_Hours`,
  `Daily_AI_Tool_Usage_Hours`, `Sleep_Hours`, `Physical_Activity_Hours`,
  `Mental_Health_Score`, `Physical_Health_Score`.

**Nota:** pese a lo que sugiere el título en Kaggle, el dataset **no incluye
rendimiento académico, burnout ni aislamiento social**.

## Definición de la tarea

`Mental_Health_Score` es continua (32,6 – 91,8), por lo que se **discretiza en
tres clases usando terciles**:

| Clase | Etiqueta | Rango | Muestras |
|-------|----------|-------|----------|
| 0 | Bajo  | ≤ 69,41 | 5.000 |
| 1 | Medio | 69,41 – 77,64 | 5.000 |
| 2 | Alto  | > 77,64 | 5.000 |

Clases perfectamente balanceadas, así que el *accuracy* es informativo y la línea
base de un clasificador aleatorio es **33,3 %**.

## Preprocesamiento

1. Se elimina `Student_ID` (identificador sin poder predictivo).
2. **One-Hot Encoding** para `Gender` y `Education_Level` → `n = 12` features.
3. **Partición estratificada 70 / 15 / 15** con `random_state=42`
   (10.500 / 2.250 / 2.250).
4. **Z-Score** con `StandardScaler`, ajustado **solo con el conjunto de
   entrenamiento** para evitar *data leakage*.

## Arquitectura

`Entrada (12) → 64 (ReLU, Dropout 0.2) → 32 (ReLU, Dropout 0.2) → 3 (logits)`

- Pérdida: `CrossEntropyLoss`. Optimizador: Adam, `lr = 0.001`, `batch_size = 64`.
- Early stopping (`patience = 15`) sobre la pérdida de validación.

## Resultados

Early stopping en la época 29; mejor modelo en la **época 14**
(`Val Loss = 0.9807`).

| Métrica | Valor |
|---------|-------|
| Accuracy (test) | **49,96 %** |
| F1-Score macro | **0,4918** |
| Línea base (azar) | 33,3 % |

| Clase | Precision | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Bajo  | 0,5321 | 0,5973 | 0,5628 | 750 |
| Medio | 0,3915 | 0,3076 | 0,3445 | 751 |
| Alto  | 0,5440 | 0,5941 | 0,5680 | 749 |

Importancia por permutación (caída de accuracy al barajar cada variable):

| Variable | Caída |
|----------|-------|
| `Daily_Social_Media_Hours` | +0,0853 |
| `Sleep_Hours` | +0,0368 |
| `Physical_Activity_Hours` | +0,0271 |
| `Daily_AI_Tool_Usage_Hours` | +0,0059 |
| `Gender`, `Education_Level`, `Age`, `Physical_Health_Score` | ≈ 0 |

## Interpretación de resultados

### Señal disponible en los datos

Correlación de cada variable con el puntaje de salud mental:

| Variable | Correlación |
|----------|-------------|
| `Daily_Social_Media_Hours` | −0,400 |
| `Sleep_Hours` | +0,311 |
| `Physical_Activity_Hours` | +0,246 |
| `Physical_Health_Score` | +0,244 |
| `Daily_AI_Tool_Usage_Hours` | −0,108 |
| `Age` | +0,004 |

El uso de redes sociales es el predictor negativo dominante y las horas de sueño
el positivo más fuerte. La edad es prácticamente ruido. Como ninguna correlación
es alta, existe un **techo de desempeño**: un accuracy cercano al 100 % sería
señal de fuga de información, no de un buen modelo.

El resultado lo confirma: **49,96 % frente a un 33,3 % de azar**. El modelo
captura señal real —supera la línea base en unos 17 puntos— pero está lejos de
ser determinista. Las curvas muestran que la pérdida de validación se estanca
hacia la época 14 mientras la de entrenamiento sigue bajando: el modelo ya
extrajo lo que los datos permiten, y añadir capas o épocas solo produciría
sobreajuste. Es el techo del problema, no una falla de la arquitectura.

La importancia por permutación coincide con el análisis de correlación:
`Daily_Social_Media_Hours` domina (barajarla cuesta 8,5 puntos de accuracy),
seguida por sueño y actividad física. En contraste, `Age`, `Gender`,
`Education_Level` y `Physical_Health_Score` tienen importancia nula o levemente
negativa: **el modelo no usa variables demográficas para decidir**, lo cual es
deseable desde el punto de vista ético porque evita sesgos por género o nivel
educativo.

### Falsos positivos y falsos negativos

El error no es simétrico. Tomando como clase crítica el nivel **Bajo** (los
estudiantes que requieren apoyo):

- **Falso positivo** — clasificar como `Bajo` a un estudiante que está bien. El
  costo es una intervención innecesaria: se consume tiempo de un servicio
  limitado y hay riesgo de estigmatización. Es un costo real pero **reversible**.
- **Falso negativo** — no detectar a un estudiante que sí está en riesgo. Queda
  fuera del sistema de apoyo justo cuando lo necesita y el deterioro puede
  agravarse sin supervisión. Costo **alto y potencialmente irreversible**.

Por esa asimetría debe **priorizarse el recall sobre la precision en la clase
`Bajo`**: es preferible revisar de más que dejar pasar un caso. Se logra
ponderando la clase en la función de pérdida o desplazando el umbral de decisión.

Los resultados por clase reflejan la estructura del problema. `Bajo` y `Alto`
alcanzan F1 de 0,56 y 0,57, mientras `Medio` se queda en 0,34 con recall 0,31:
**la clase intermedia es la que el modelo no logra separar**, y sus muestras se
reparten hacia los extremos. Esa confusión es la esperable y la menos
preocupante: los terciles imponen fronteras artificiales sobre una variable
continua, de modo que un estudiante con puntaje 69,3 y otro con 69,5 son casi
idénticos pero caen en clases distintas. Los errores graves son los de la esquina
de la matriz de confusión: confundir `Bajo` con `Alto`.

### Limitaciones

- **Correlación no implica causalidad.** El modelo detecta que más horas de redes
  sociales acompañan a peores puntajes, pero no permite concluir que las causen;
  la relación puede ser inversa o deberse a factores no medidos.
- Las distribuciones son muy regulares y no hay ningún valor faltante, lo que
  sugiere datos **sintéticos o fuertemente depurados**. El desempeño no se
  trasladaría a datos reales de una institución.
- La variable objetivo es un autorreporte convertido en puntaje, con la
  subjetividad que eso implica.
- Un accuracy del 50 % en tres clases **no basta para tomar decisiones
  automáticas sobre estudiantes concretos**. Sirve como herramienta de
  priorización o tamizaje, nunca como diagnóstico.
