# ML1_ExamenAplicado_Valderrama_Alexander

Examen individual — Machine Learning I. Predicción del valor medio de vivienda en California mediante modelos de regresión penalizada y de ensamble, con análisis exploratorio, tratamiento de outliers, PCA y clustering K-Means.

## Dataset

- **Nombre:** California Housing Dataset
- **Fuente:** `sklearn.datasets.fetch_california_housing` (scikit-learn), originalmente de StatLib (Carnegie Mellon University)
- **URL:** https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Filas / columnas:** 20.640 filas × 9 columnas (8 predictoras + 1 objetivo)
- **Variable objetivo:** `MedHouseVal` (valor medio de vivienda, en cientos de miles de USD)
- **Tipo de tarea:** Regresión

## Metodología (resumen)

1. **EDA:** revisión de tipos, nulos (0%), outliers vía método IQR (tratados con capping/winsorización para no distorsionar PCA/K-Means), distribución del target (skewness=0,98), correlaciones de Pearson y multicolinealidad (Latitude–Longitude, r=-0,92).
2. **Preprocesamiento:** split train/test (80/20, `random_state=42`) antes de cualquier transformación; `ColumnTransformer` con `SimpleImputer` + `StandardScaler` (sin variables categóricas en este dataset).
3. **No supervisado:** PCA (5 componentes, 90,16% de varianza explicada) y K-Means (K=4, seleccionado por interpretabilidad del perfil de clusters, con 4 segmentos geográfico-socioeconómicos claros de California).
4. **Modelado supervisado:** Ridge (penalizado, `GridSearchCV` sobre alpha) y Random Forest (`GridSearchCV` sobre n_estimators/max_depth/min_samples_split), ambos con validación cruzada de 5 folds.
5. **Evaluación:** métricas sobre el conjunto de test reservado desde el inicio (nunca usado para ajustar ningún modelo).

## Resultados del mejor modelo (Random Forest)

| Métrica | Valor (test) |
|---|---|
| RMSE | 0,5050 |
| MAE | 0,3297 |
| R² | 0,8054 |
| MAPE | 0,1904 |
| R² train (comparación sobreajuste) | 0,9727 |
| Tiempo de entrenamiento | ~48 s |

Random Forest se seleccionó por sobre Ridge (RMSE=0,6811, R²=0,6460) al ofrecer una mejora de desempeño de ~26% en RMSE. El detalle completo de la comparación, la justificación y el análisis de errores están en el notebook.

## Video

https://youtu.be/bj9DIw34t4g

## Declaración de uso de Inteligencia Artificial generativa

Este trabajo fue desarrollado con apoyo de **Claude (Anthropic)** como asistente de IA generativa, utilizado para estructurar el pipeline conforme a los puntos exigidos en la ficha del examen, redactar y depurar el código Python, y redactar las interpretaciones en Markdown a partir de los resultados numéricos obtenidos por la ejecución real del código. Todas las decisiones metodológicas (tratamiento de outliers, selección de K, selección de modelo final) y las conclusiones fueron revisadas y validadas por el autor.

## Cómo reproducir el análisis

```bash
pip install -r requirements.txt
jupyter notebook Valderrama_codigo_examen.ipynb
```

## Estructura del repositorio

```
├── Valderrama_codigo_examen.ipynb   # Notebook ejecutado, con outputs
├── README.md
├── requirements.txt
├── tabla_comparativa_modelos.csv    # Tabla comparativa exportada
└── figures/                        # Todos los graficos generados (dpi=150)
```
