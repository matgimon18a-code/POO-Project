# Práctica OOP – Linear Regression (C++)

**Integrantes:**  
- **Juliana Sepúlveda**  
- **Matías Gil**

Este repositorio implementa, sin librerías externas, un **modelo de Regresión Lineal** con un enfoque **orientado a objetos**. Se incluyen funciones auxiliares para lectura de CSV, partición *train/test*, estandarización (z‑score), álgebra de matrices (transpuesta, producto, inversión por Gauss‑Jordan con pivoteo parcial) y métricas (R², MSE, MAE).

> **Archivos de ejemplo (“setlists”)**:  
> - `Ice_cream_selling_data.csv` (Regresión simple: temperatura → ventas)  
> - `student_exam_scores.csv` (Regresión múltiple: horas estudio, sueño, asistencia, notas previas → nota examen)

---

## 1) Objetivo del reto
- Desarrollar una solución **POO** en C++ para regresión lineal **sin** usar librerías externas (Eigen, BLAS, etc.).  
- Demostrar entreno, predicción y evaluación con dos *setlists* (datasets) provistos.  
- Mantener el código **portable** (GCC/Clang/MSVC) y **didáctico** (comentado en español).

---

## 2) Cómo compilar y ejecutar

### Opción A — Linux / MinGW / macOS (GCC/Clang)
```bash
g++ -std=c++17 -O2 -o oop_lr main.cpp
./oop_lr
```

### Opción B — Windows (Visual Studio – Developer Command Prompt)
```bat
cl /std:c++17 /O2 /EHsc main.cpp
main.exe
```

> **Nota:** Si usas Visual Studio IDE, crea un **Console App (C++)**, agrega `main.cpp`, y configura `/std:c++17` en Propiedades → C/C++ → Lenguaje.

---

## 3) Configurar las “setlists” (datasets)
El `main()` llama a:
```cpp
demo_dataset("Ice_cream_selling_data.csv", true, LinearRegression::NORMAL_EQUATION);
demo_dataset("student_exam_scores.csv", true, LinearRegression::NORMAL_EQUATION);
```
Asegúrate de que los CSV estén en la **misma carpeta** que el ejecutable (o usa rutas absolutas, por ejemplo en Windows):
```cpp
demo_dataset("C:\\\\Users\\\\TuUsuario\\\\Documents\\\\Ice_cream_selling_data.csv", true, LinearRegression::NORMAL_EQUATION);
demo_dataset("C:\\\\Users\\\\TuUsuario\\\\Documents\\\\student_exam_scores.csv", true, LinearRegression::NORMAL_EQUATION);
```

> Si el profe se refiere a “**setlists**” como **rutas** a los archivos, **cámbialas aquí**.

---

## 4) Estructura general del código

- **`class StandardScaler`**  
  - `fit(X)`: calcula medias y desviaciones por columna.  
  - `transform(X)`: aplica z‑score usando las estadísticas guardadas.  
- **Álgebra de matrices (helpers)**  
  - `transpose(A)`, `matmul(A,B)`, `identity(n)`, `inverse(A)` (Gauss‑Jordan con pivoteo).  
- **`class LinearRegression`**  
  - Atributos: `weights_` (vector<double>), `bias_` (double), `scaler_`, `method_`, etc.  
  - `fit(X,y)`: entrena por **Ecuación Normal** o **Gradiente Descendente**.  
  - `predict(X)`: usa `weights_` y `bias_` para generar `y_hat` (aplica el mismo *scaling* si procede).  
  - `score(X,y)`: calcula **R²**, **MSE**, **MAE**.  
- **Carga de CSV y partición**  
  - `read_csv_xy(path)`: asume **última columna** = `y`.  
  - `train_test_split(...)`: *shuffle* reproducible (semilla fija).  
- **`demo_dataset(path, scale, method)`**  
  - Entrena, evalúa y muestra métricas para cada setlist.

---

## 5) Explicación detallada (POO y flujo)

### 5.1. Data Scaling (z‑score)
- En `fit(X,y)`, si `scale_features_` = `true`:  
  1. `scaler_.fit(X)` calcula media y desviación por columna (muestral, divide por *n‑1*).  
  2. `X = scaler_.transform(X)` normaliza las variables.  
- En `predict(X_nuevo)`, se reaplica **el mismo** `transform()` con las estadísticas del entrenamiento.  
- Beneficios: mejor **condicionamiento** de \( X^T X \) (inversión más estable) y **convergencia** del gradiente.

### 5.2. Entrenamiento – `fit(X,y)`
- **Ecuación Normal**:  
  \[ \theta = (X_b^T X_b)^{-1} X_b^T y \]
  con \( X_b = [X \mid \mathbf{1}] \) (columna de unos). Se separa \( \theta = [w_1,\dots,w_d,b] \).  
- **Gradiente Descendente** (opcional):  
  Minimiza MSE con actualizaciones:  
  \( b \leftarrow b - \alpha \frac{2}{n} \sum_i e_i \),  
  \( w_j \leftarrow w_j - \alpha \frac{2}{n} \sum_i e_i X_{ij} \).

### 5.3. Predicción – `predict(X)`
- Para cada fila \( x_i \): \( \hat{y}_i = b + \sum_j w_j x_{ij} \).  
- Devuelve un `vector<double>` con todas las predicciones.

### 5.4. Evaluación – `score(X,y)`
- Calcula \( \bar{y} \), \( SS_{res}=\sum (y-\hat{y})^2 \), \( SS_{tot}=\sum (y-\bar{y})^2 \).  
- **R²** \(=1-SS_{res}/SS_{tot}\), **MSE**, **MAE**.  
- Llama internamente a `predict(X)`.

### 5.5. Lectura CSV – `read_csv_xy(path)`
- `split_csv_line(...)` maneja comillas para celdas con comas.  
- `to_double(...)` limpia símbolos (`%`, `°`, no‑ASCII) y usa `stod`.  
- Supone que la **última columna** es el *target* `y` y el resto son *features* `X`.

### 5.6. Inversión de matrices – `inverse(A)`
- Gauss‑Jordan con **pivoteo parcial** (mejora estabilidad): intercambia filas por máximo pivote absoluto, normaliza y elimina.  
- Lanza `runtime_error` si la matriz es *singular o mal condicionada*.

---

## 6) Uso esperado y salida

Con los dos setlists de ejemplo, el programa imprime algo como:
```
=== Ejecutando demo con: Ice_cream_selling_data.csv ===
Características (1): Temperature (°C)
Variable objetivo: Ice Cream Sales (units)
R^2=0.94.. | MSE=... | MAE=...
 
=== Ejecutando demo con: student_exam_scores.csv ===
Características (4): hours_studied sleep_hours attendance_percent previous_scores
Variable objetivo: exam_score
R^2=0.84.. | MSE=... | MAE=...
```
> Los números exactos varían levemente por el *shuffle* (semilla fija reduce la variación).

---

## 7) Retos y soluciones

1. **Invertir \( X^T X \) sin Eigen/BLAS**  
   - *Reto*: estabilidad numérica.  
   - *Solución*: **Gauss‑Jordan con pivoteo parcial** + *scaling* previo de features.

2. **Lectura de CSV con campos con comas/porcentajes**  
   - *Reto*: parsear filas con comillas y símbolos extra.  
   - *Solución*: `split_csv_line` sensible a comillas + `to_double` que limpia `%`, `°`, no‑ASCII.

3. **Portabilidad GCC/MSVC**  
   - *Reto*: `<bits/stdc++.h>` no existe en MSVC.  
   - *Solución*: incluir solo **cabeceras estándar** (`<vector>`, `<numeric>`, etc.) y usar `std::` explícito.

---

## 8) Conclusiones

1. La **estandarización** de variables mejora la estabilidad de la **Ecuación Normal** y la convergencia del **Gradiente**.  
2. Implementar álgebra básica (transpuesta, producto, inversión) es suficiente para **prototipos educativos** sin librerías externas.  
3. Un pipeline POO claro (`fit` → `predict` → `score`, con `StandardScaler`) facilita **mantenibilidad** y **extensión** (ej. regularización L2, k‑fold).

---

## 9) Nota importante (recordatorio sobre *setlists*)
> **Recordatorio:** Antes de compilar/ejecutar, **ajusta en `main.cpp` las rutas de las “setlists”** (datasets) para que el programa pueda leerlas. Si los CSV no están en la misma carpeta del ejecutable, **usa rutas absolutas** en las llamadas a `demo_dataset(...)`.
