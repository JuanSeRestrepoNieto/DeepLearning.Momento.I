# Momento Evaluativo I — Clasificación de imágenes con un MLP

Clasificación de la **subcategoría de un producto de moda** (Topwear, Shoes,
Bags, Watches, …) a partir de su imagen, usando un Perceptrón Multicapa
implementado en PyTorch.

## Dataset

**Fashion Product Images (Small)** — Kaggle, `paramaggarwal/fashion-product-images-small`

Se descarga con `kagglehub` (ver `download_data.py`). Se usa la variante *small*
(60×80 px, ~565 MB) en vez de la full-res de 25 GB: contiene exactamente las
mismas imágenes y el mismo `styles.csv`, y como el MLP reescala a 32×32 de todos
modos, la resolución original es irrelevante para esta tarea.

| | |
|---|---|
| Imágenes en disco | 44.441 |
| Filas utilizables tras cruzar `styles.csv` con las imágenes y filtrar clases raras | **43.974** |
| Clases (`subCategory`) | **27** |
| Resolución de entrada | 32 × 32 × 3 |
| Features tras aplanar (n) | **3.072** |

### Definición de la tarea

La etiqueta es la columna `subCategory` de `styles.csv`. De las 45 subcategorías
originales se **descartan las que tienen menos de 100 muestras**, porque no
permiten una partición estratificada en tres subconjuntos ni una evaluación
estadísticamente significativa. Quedan 27 clases.

### Desbalance

El dataset está fuertemente desbalanceado:

| Clase | Muestras | % |
|-------|----------|---|
| Topwear | 15.398 | 35,02 % |
| Shoes | 7.343 | 16,70 % |
| Bags | 3.055 | 6,95 % |
| Bottomwear | 2.693 | 6,12 % |
| Watches | 2.542 | 5,78 % |
| … | … | … |
| Apparel Set | 106 | 0,24 % |
| Free Gifts | 104 | 0,24 % |

**Razón de desbalance: 148 a 1.** Esto tiene dos consecuencias metodológicas:

1. El *accuracy* deja de ser una métrica confiable — un modelo que prediga
   siempre `Topwear` acertaría el 35 % sin haber aprendido nada. Por eso se
   reporta también el **F1-Score macro**, que promedia las clases sin ponderar
   por su tamaño.
2. Se aplican **pesos por clase** en la función de pérdida
   (`CrossEntropyLoss(weight=...)`), inversamente proporcionales a su frecuencia.

## Preprocesamiento

1. Se cruzan las filas de `styles.csv` con las imágenes realmente presentes en
   disco (`on_bad_lines="skip"` porque algunas filas tienen comas de más en
   `productDisplayName`).
2. Cada imagen se convierte a RGB, se reescala a 32×32 y se guarda como `uint8`
   (cuatro veces menos memoria que `float32`). El resultado se cachea en
   `data/fashion_cache_32.npz` para no repetir la decodificación.
3. **Partición estratificada 70 / 15 / 15** con `random_state=42`.
4. **Normalización Z-Score por canal**, con media y desviación calculadas
   **únicamente sobre el conjunto de entrenamiento**, para evitar fuga de
   información (*data leakage*).
5. La imagen se aplana a un vector de 3.072 componentes, que es la entrada del MLP.

## Arquitectura

```
Entrada 3072 → 512 → 256 → 128 → 27 (logits)
```

Cada capa oculta lleva `BatchNorm1d → ReLU → Dropout(0.3)`. BatchNorm estabiliza
el entrenamiento con una entrada de dimensión alta; Dropout contiene el
sobreajuste.

- Pérdida: `CrossEntropyLoss` con pesos por clase.
- Optimizador: Adam, `lr = 0.001`, `batch_size = 128`.
- **Early stopping sobre el F1-Score macro de validación** (`patience = 12`).
  No se usa la pérdida de validación como criterio: al ponderar las clases queda
  demasiado ruidosa y llegó a seleccionar una época con 4 puntos menos de
  accuracy que la mejor. El F1 macro es el criterio coherente con el desbalance.

## Cómo ejecutar

Credenciales de Kaggle: Kaggle → Settings → API → *Create New API Token*, y
guardar el archivo en `~/.kaggle/kaggle.json`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python download_data.py
```

```bash
.venv/bin/python eda.py
```

```bash
.venv/bin/python main.py
```

```bash
.venv/bin/python evaluate.py
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `download_data.py` | Descarga del dataset desde Kaggle con `kagglehub` |
| `dataset.py` | Cruce con `styles.csv`, decodificación y reescalado de imágenes, caché, partición estratificada, Z-Score y `DataLoader`s |
| `model.py` | Definición del MLP |
| `main.py` | Entrenamiento, validación por época, early stopping por F1 macro y curvas |
| `evaluate.py` | Evaluación final sobre test: métricas, matriz de confusión, peores clases y confusiones más frecuentes |
| `eda.py` | Análisis exploratorio y figuras |
| `Ejemplo/` | Código de referencia con MNIST visto en clase |
| `v1_salud_mental/` | Versión anterior del trabajo (clasificación tabular de salud mental estudiantil) |

## Resultados

Partición: 30.781 train / 6.596 validación / 6.597 test.
El entrenamiento **alcanzó el tope de 60 épocas sin disparar el early stopping**;
el mejor checkpoint es el de la época 58 (F1 macro de validación 0,8223).

| Métrica | Valor |
|---------|-------|
| Accuracy (test) | **89,31 %** |
| F1-Score macro | **0,7993** |
| F1-Score ponderado | 0,9011 |
| Línea base (predecir siempre `Topwear`) | 35,0 % |

### Mejores y peores clases

| Clase | Precision | Recall | F1 | Soporte |
|-------|-----------|--------|-----|---------|
| Ties | 1,0000 | 1,0000 | **1,0000** | 38 |
| Eyewear | 0,9877 | 1,0000 | 0,9938 | 161 |
| Belts | 0,9683 | 1,0000 | 0,9839 | 122 |
| Nails | 0,9600 | 0,9796 | 0,9697 | 49 |
| Topwear | 0,9845 | 0,9065 | 0,9439 | 2.310 |
| Shoes | 0,9793 | 0,9020 | 0,9391 | 1.102 |
| … | | | | |
| Scarves | 0,5882 | 0,5882 | 0,5882 | 17 |
| Dress | 0,3000 | 0,7606 | 0,4303 | 71 |
| Free Gifts | 0,0976 | 0,2667 | **0,1429** | 15 |

### Confusiones más frecuentes

| Real → Predicha | Casos |
|-----------------|-------|
| Topwear → Dress | 107 |
| Shoes → Sandal | 79 |
| Bags → Wallets | 40 |
| Topwear → Innerwear | 30 |
| Watches → Free Gifts | 29 |
| Shoes → Flip Flops | 20 |

Figuras en `figures/`: `eda_distribucion_clases.png`, `eda_ejemplos.png`,
`eda_imagen_promedio.png`, `curva_loss.png`, `curva_accuracy.png`,
`curva_f1.png`, `matriz_confusion.png`, `peores_clases.png`.

## Interpretación de resultados

### El accuracy engaña, el F1 macro no

89,31 % de accuracy suena excelente, pero la brecha con el **F1 macro de 0,7993**
—casi diez puntos— es el dato importante: el modelo es muy bueno en las clases
grandes y bastante peor en las pequeñas. El F1 ponderado (0,9011) prácticamente
coincide con el accuracy justamente porque ambos están dominados por `Topwear` y
`Shoes`, que juntas son más de la mitad del conjunto de prueba. **Reportar solo el
accuracy en un dataset con desbalance de 148 a 1 ocultaría el verdadero
comportamiento del modelo.**

El efecto de los pesos por clase se ve en la asimetría entre precision y recall
macro: 0,7777 frente a 0,8422. Al penalizar más los errores sobre clases
minoritarias, el modelo se vuelve más agresivo prediciéndolas — recupera más
casos verdaderos (recall alto) a costa de generar falsas alarmas (precision baja).
`Dress` es el ejemplo extremo: recall 0,76 pero precision 0,30, es decir, captura
la mayoría de los vestidos reales pero dos de cada tres cosas que llama "vestido"
no lo son. Fue una decisión deliberada; sin los pesos, esas clases simplemente
desaparecerían de las predicciones.

### Qué revelan las confusiones sobre las limitaciones del MLP

Los errores no son aleatorios y ahí está lo interesante:

- **Topwear → Dress (107 casos)** y **Shoes → Sandal (79)** son confusiones entre
  clases con siluetas casi idénticas a 32×32. Un vestido y una camiseta larga
  producen prácticamente el mismo vector de píxeles a esa resolución.
- **Bags → Wallets (40)** es el mismo objeto a distinta escala. Como las
  fotografías de catálogo están todas encuadradas de forma similar, el MLP pierde
  la referencia de tamaño real.
- **Watches → Free Gifts (29)** revela un problema de la propia etiqueta:
  `Free Gifts` no es una categoría *visual*, es una categoría *comercial*. Sus
  imágenes son relojes, perfumes o bolsos. Ninguna arquitectura puede aprender
  esa clase desde el píxel, y su F1 de 0,14 lo confirma. **No es un fallo del
  modelo sino de la definición del problema**, y una versión posterior debería
  excluirla explícitamente.

La limitación de fondo es arquitectónica: **el MLP destruye la estructura
espacial al aplanar la imagen**. Cada uno de los 3.072 píxeles es una feature
independiente, sin ninguna noción de que dos píxeles vecinos están relacionados.
Que funcione tan bien pese a eso se debe a que las fotos de catálogo son
extremadamente uniformes —producto centrado, fondo blanco, encuadre constante—,
así que la posición absoluta de cada píxel sí es informativa. Sobre fotos reales,
con fondos y encuadres variables, este mismo modelo se derrumbaría. Ahí es donde
una CNN, que aprende filtros invariantes a la traslación, marcaría la diferencia.

### Falsos positivos y falsos negativos

En un catálogo de e-commerce, que es el uso natural de este modelo, el costo de
los errores depende de la clase:

- **Falso positivo** — etiquetar un producto en una categoría que no le
  corresponde. El producto aparece en búsquedas equivocadas: el cliente que filtra
  por "vestidos" ve camisetas. Genera fricción y desconfianza, pero se corrige
  editando la ficha.
- **Falso negativo** — no asignar la categoría correcta. El producto queda
  **invisible** en la navegación por categorías, que es como la mayoría de
  usuarios compra. Comercialmente es el error más caro: un producto que no
  aparece no se vende.

Como el catálogo se navega principalmente por categorías, aquí conviene
**priorizar el recall**, y por eso la configuración con pesos de clase —que
sacrifica precision para elevar recall— es la adecuada para este caso de uso. La
alternativa sensata en producción sería usar el modelo como **sugeridor**: que
proponga las 3 categorías más probables y un operador confirme, en lugar de
etiquetar automáticamente.

### Limitaciones

- Las 18 subcategorías con menos de 100 muestras fueron **excluidas**, no
  resueltas. El modelo no sabe que existen y en producción las clasificaría mal
  con total confianza.
- `Free Gifts` es una categoría comercial sin correlato visual y contamina la
  métrica macro; debería eliminarse del planteamiento.
- La resolución de 32×32 es una restricción impuesta por el MLP, no por los
  datos: elimina detalles (texturas, estampados, logos) que distinguirían
  `Topwear` de `Dress`. Subirla haría crecer la capa de entrada cuadráticamente,
  que es exactamente el problema que las CNN resuelven con pesos compartidos.
- El entrenamiento se cortó por el límite de épocas, no por convergencia: el F1
  de validación aún subía. Con más épocas el resultado mejoraría algo, aunque la
  brecha entre train (90,1 %) y validación (89,0 %) indica que el margen restante
  es pequeño.
- Todas las imágenes provienen de un único catálogo (Myntra) con condiciones de
  fotografía homogéneas. **El desempeño no se traslada a imágenes tomadas por
  usuarios.**
