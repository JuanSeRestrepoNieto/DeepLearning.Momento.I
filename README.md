# Momento Evaluativo I — Clasificación de imágenes con un MLP

Clasificación de la **subcategoría de un producto de moda** (Topwear, Shoes,
Bags, Watches, …) a partir de su imagen, con un Perceptrón Multicapa en PyTorch.

## El problema

**Fashion Product Images (Small)** — Kaggle, `paramaggarwal/fashion-product-images-small`.

De las 44.441 imágenes del catálogo se usan **43.974**, repartidas en **27
subcategorías**. Se descartaron las categorías con menos de 100 muestras porque
con tan pocos casos no es posible repartirlas entre entrenamiento, validación y
prueba, ni medir nada confiable sobre ellas.

El dato que condiciona todo el trabajo es el **desbalance**:

Distribución de clases (figures/eda_distribucion_clases.png)

`Topwear` concentra 15.398 imágenes (35 % del total) y `Free Gifts` apenas 104.
Son **148 a 1**. Eso significa que un modelo que respondiera siempre "Topwear"
acertaría el 35 % de las veces sin haber aprendido absolutamente nada, y obliga a
medir el desempeño con algo más que el porcentaje de aciertos.

## Qué se hizo

Cada imagen se reduce a 32 × 32 píxeles y se entrega a una red de cuatro capas
que devuelve una de las 27 categorías. Los datos se repartieron en **30.781 de
entrenamiento, 6.596 de validación y 6.597 de prueba**, manteniendo la misma
proporción de clases en los tres grupos y reservando el de prueba sin tocarlo
hasta el final.

Como el dataset está tan desbalanceado, durante el entrenamiento se le dio **más
peso a los errores sobre las categorías pequeñas**, para que el modelo no las
ignorara. El entrenamiento corrió 70 épocas y se conservó el modelo de la época
**69**, el mejor sobre el conjunto de validación.

## Resultados

| Métrica | Valor |
|---|---|
| **Accuracy** (aciertos sobre el total) | **90,50 %** |
| **F1-Score macro** (todas las categorías pesan igual) | **0,8250** |
| F1-Score ponderado (cada categoría pesa según su tamaño) | 0,9089 |
| Precision macro | 0,8027 |
| Recall macro | 0,8630 |
| Línea base (responder siempre `Topwear`) | 35,0 % |

### Las mejores y las peores categorías

| Mejores | F1 | | Peores | F1 |
|---|---|---|---|---|
| Eyewear | 0,9875 | | Free Gifts | 0,4091 |
| Belts | 0,9799 | | Dress | 0,5222 |
| Nails | 0,9796 | | Loungewear and Nightwear | 0,6260 |
| Ties | 0,9744 | | Scarves | 0,6341 |
| Shoes | 0,9543 | | Makeup | 0,7103 |
| Topwear | 0,9453 | | Sandal | 0,7351 |

Peores clases (figures/peores_clases.png)

### Los errores más frecuentes

| Real → Predicha | Casos |
|---|---|
| Topwear → Dress | 67 |
| Topwear → Innerwear | 58 |
| Bags → Wallets | 40 |
| Shoes → Sandal | 32 |
| Topwear → Jewellery | 24 |
| Flip Flops → Shoes | 21 |
| Watches → Free Gifts | 14 |

Matriz de confusión (figures/matriz_confusion.png)

## Interpretación

### El 90 % de aciertos es un número engañoso

A primera vista 90,50 % parece un resultado muy bueno. Pero al comparar las dos
formas de calcular el F1-Score aparece lo que ese número esconde: **0,9089 cuando
cada categoría pesa según su tamaño, pero 0,8250 cuando todas pesan igual**.

La diferencia de ocho puntos se explica sola: `Topwear` y `Shoes` son más de la
mitad del conjunto de prueba, así que el accuracy mide sobre todo qué tan bien
clasificamos camisetas y zapatos. Cuando se obliga a que una categoría de 17
prendas cuente lo mismo que una de 2.310, el desempeño real baja.

Las dos cifras no se contradicen: responden preguntas distintas. El **F1
ponderado** dice qué tan bien funciona el catálogo para el cliente promedio, que
navega sobre todo las categorías grandes. El **F1 macro** dice qué tan bien
funciona la categoría peor atendida. Si al negocio le importa que las líneas
pequeñas también se vendan, la segunda es la que hay que mirar.

### Las curvas muestran que el modelo dejó de aprender a mitad de camino

 Curva de pérdida (figures/curva_loss.png)

Esta es la gráfica más reveladora, y no dice nada bueno. El error sobre los datos
de entrenamiento (azul) baja sin parar. El error sobre los datos de validación
(naranja) toca su mejor punto alrededor de la **época 27** y a partir de ahí se
queda estancado, e incluso empeora al final.

Que las dos curvas se separen significa que **desde la época 27 el modelo dejó de
aprender y empezó a memorizar**. Siguió mejorando sobre las imágenes que ya había
visto, pero eso ya no se tradujo en acertar mejor sobre imágenes nuevas.

 Curva de accuracy (figures/curva_accuracy.png)

Lo interesante es que esta segunda gráfica parece decir lo contrario: las dos
líneas van pegadas las 70 épocas y terminan juntas cerca del 90 %. Leída sola,
diría que todo está perfecto.

**Ambas gráficas son correctas; lo que falla es el accuracy como termómetro.**
Como está dominado por las categorías grandes, sigue subiendo mientras el modelo
mantenga `Topwear` y `Shoes` bien clasificados, y no se entera de que el
desempeño sobre el resto se está deteriorando. Si hubiéramos evaluado solo con
accuracy —que era lo intuitivo—, este problema habría pasado desapercibido.

Curva de F1 (figures/curva_f1.png)

El F1 macro sube de 0,55 a 0,83 y se aplana claramente a partir de la época 40.
Los saltos de arriba abajo vienen de las categorías chicas: basta que `Free Gifts`
pase de 4 a 7 aciertos —sobre 15 imágenes— para mover el promedio varios puntos.

La conclusión práctica es que **alargar el entrenamiento ya no sirve**. Subir el
tope de 60 a 70 épocas mejoró el F1 macro de 0,7993 a 0,8250, pero la curva de
error indica que a partir de ahí solo se profundizaría la memorización.

### Por qué falla donde falla

Los errores no son aleatorios. Al revisarlos aparecen **tres causas distintas**,
y conviene no mezclarlas porque cada una se arregla de forma diferente.

**1. Una categoría que no se puede aprender.** `Free Gifts` es la peor de todas
(F1 0,4091), y el modelo no tiene la culpa. No es una categoría *visual* sino
*comercial*: sus imágenes son relojes, perfumes o bolsos que la tienda regalaba en
alguna promoción. Lo que define esa categoría no está en la foto. Por eso el
modelo confunde relojes con "regalos" 14 veces. Ninguna red podría resolverlo, y
lo correcto sería **sacar esa categoría del problema**.

**2. Objetos que a 32 × 32 píxeles se ven iguales.** Es el grupo más numeroso:
`Topwear → Dress` (67 casos), `Topwear → Innerwear` (58), `Shoes → Sandal` (32).
Un vestido y una camiseta larga, reducidos a una miniatura, son prácticamente la
misma silueta. Lo que los distingue —el largo exacto, la textura de la tela, el
estampado— se pierde al encoger la imagen. `Bags → Wallets` (40 casos) es una
variante del mismo problema: es el mismo objeto a distinta escala, y como todas
las fotos del catálogo están encuadradas igual, el modelo pierde la referencia de
tamaño real. **Este grupo sí se arregla**, con más resolución o con un modelo que
sepa mirar detalles.

**3. Categorías con muy pocos ejemplos.** `Scarves` tiene 17 imágenes de prueba y
`Accessories` 19. Con tan pocos casos, un solo acierto de más o de menos mueve la
métrica varios puntos. Aquí el problema no es el modelo sino **que la medición no
es confiable**; se resuelve consiguiendo más datos.

### El modelo prefiere equivocarse por exceso

Al comparar precision (0,8027) con recall (0,8630) se ve que el modelo es **más
propenso a asignar de más que a dejar sin asignar**. Eso fue intencional: al darle
más peso a las categorías pequeñas durante el entrenamiento, se volvió más
arriesgado al predecirlas.

`Dress` es el ejemplo extremo. Encuentra tres de cada cuatro vestidos reales, pero
de todo lo que llama "vestido", seis de cada diez no lo son. Sin ese ajuste las
categorías pequeñas simplemente **no aparecerían nunca** en las predicciones, lo
cual sería peor.

La decisión tiene sentido pensando en para qué sirve esto. En una tienda en línea
los dos errores no cuestan lo mismo:

- Si un producto queda **en la categoría equivocada**, el cliente que filtra por
  "vestidos" ve camisetas. Molesta, pero se corrige editando la ficha.
- Si un producto **no recibe su categoría**, queda invisible para quien navega por
  categorías. Y un producto que no aparece, no se vende.

El segundo error es el caro, así que conviene pecar de más y no de menos.

Aun así, con 90,5 % de aciertos uno de cada diez productos quedaría mal
clasificado: en un catálogo de 44.000 artículos son unos **4.400 errores**.
Demasiados para etiquetar automáticamente. El uso sensato sería como
**asistente**: que el modelo proponga las tres categorías más probables y una
persona confirme.

### La limitación de fondo

El modelo recibe la imagen como una lista plana de píxeles, sin ninguna noción de
que dos píxeles vecinos tienen que ver entre sí. Le pasa algo parecido a alguien
que intentara reconocer una foto leyendo los colores uno por uno, en fila.

Que aun así funcione bien se debe a que **las fotos de catálogo son todas
iguales**: producto centrado, fondo blanco, mismo encuadre. En esas condiciones,
saber qué hay en cada posición fija sí sirve. Pero eso también marca el límite del
resultado: sobre fotos tomadas por usuarios, con fondos y ángulos variables,
**este modelo se derrumbaría**, y el 90,5 % dejaría de ser una cifra válida.

Ese es justamente el problema que resuelven las redes convolucionales, que
aprenden a reconocer formas sin importar en qué parte de la imagen aparezcan.

## Limitaciones

- Las 18 subcategorías con menos de 100 muestras quedaron **excluidas**, no
  resueltas. El modelo no sabe que existen.
- `Free Gifts` no debería formar parte del problema y arrastra la métrica macro
  hacia abajo.
- La resolución de 32 × 32 elimina los detalles que distinguirían `Topwear` de
  `Dress`.
- Todas las imágenes vienen de un único catálogo (Myntra) con fotografía
  homogénea. **El resultado no se traslada a fotos de usuarios.**

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
| `download_data.py` | Descarga del dataset desde Kaggle |
| `dataset.py` | Preparación de los datos: carga, reescalado, partición y normalización |
| `model.py` | Definición de la red |
| `main.py` | Entrenamiento y curvas |
| `evaluate.py` | Evaluación final sobre el conjunto de prueba |
| `eda.py` | Análisis exploratorio |
| `Ejemplo/` | Código de referencia con MNIST visto en clase |
