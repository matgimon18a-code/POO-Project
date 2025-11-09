// ===============================================================
//  Práctica OOP - Regresión Lineal en C++ (sin librerías externas)
//  Autor: (tu nombre aquí)
//  Compilación: g++ -std=c++17 -O2 main.cpp -o oop_lr
//  Ejecución:   ./oop_lr
// ===============================================================

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// ---------------------------------------------------------------
//  ESTRUCTURA PARA GUARDAR LAS MÉTRICAS DE EVALUACIÓN
// ---------------------------------------------------------------
struct Score {
    double r2;   // Coeficiente de determinación
    double mse;  // Error cuadrático medio
    double mae;  // Error absoluto medio
};

// ===============================================================
//  BLOQUE DE DATA SCALING (Estandarización de datos)
// ===============================================================
class StandardScaler {
public:
    vector<double> mean_, std_;
    bool fitted = false;

    // --- Método fit(): calcula la media y desviación estándar de cada columna
    void fit(const vector<vector<double>>& X) {
        size_t n = X.size(), d = X[0].size();
        mean_.assign(d, 0.0);
        std_.assign(d, 0.0);

        // Calcular medias
        for (size_t j = 0; j < d; ++j) {
            for (size_t i = 0; i < n; ++i)
                mean_[j] += X[i][j];
            mean_[j] /= (double)n;
        }

        // Calcular desviaciones estándar
        for (size_t j = 0; j < d; ++j) {
            for (size_t i = 0; i < n; ++i) {
                double diff = X[i][j] - mean_[j];
                std_[j] += diff * diff;
            }
            std_[j] = sqrt(std_[j] / max<size_t>(1, n - 1));
            if (std_[j] == 0.0) std_[j] = 1.0; // Evitar división por 0
        }
        fitted = true;
    }

    // --- Método transform(): aplica la estandarización (z-score)
    vector<vector<double>> transform(const vector<vector<double>>& X) const {
        if (!fitted) return X;
        vector<vector<double>> Z = X;
        for (size_t i = 0; i < Z.size(); ++i) {
            for (size_t j = 0; j < Z[i].size(); ++j) {
                Z[i][j] = (Z[i][j] - mean_[j]) / std_[j];
            }
        }
        return Z;
    }
};

// ===============================================================
//  FUNCIONES AUXILIARES DE MATRICES (sin librerías externas)
// ===============================================================

using Matrix = vector<vector<double>>;
using Vec = vector<double>;

static Matrix transpose(const Matrix& A) {
    size_t n = A.size(), m = A[0].size();
    Matrix T(m, vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            T[j][i] = A[i][j];
    return T;
}

static Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t n = A.size(), m = A[0].size(), p = B[0].size();
    Matrix C(n, vector<double>(p, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < m; ++k)
            for (size_t j = 0; j < p; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

static Matrix identity(size_t n) {
    Matrix I(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

// --- Inversa de matriz mediante método de Gauss-Jordan con pivoteo parcial
static Matrix inverse(Matrix A) {
    size_t n = A.size();
    Matrix I = identity(n);
    for (size_t i = 0; i < n; ++i) {
        // Buscar pivote
        size_t pivot = i;
        double mx = fabs(A[i][i]);
        for (size_t r = i + 1; r < n; ++r)
            if (fabs(A[r][i]) > mx) { mx = fabs(A[r][i]); pivot = r; }

        if (mx < 1e-12) throw runtime_error("Matriz singular o mal condicionada.");
        if (pivot != i) { swap(A[i], A[pivot]); swap(I[i], I[pivot]); }

        // Normalizar fila
        double diag = A[i][i];
        for (size_t j = 0; j < n; ++j) {
            A[i][j] /= diag;
            I[i][j] /= diag;
        }

        // Eliminar otros elementos de la columna
        for (size_t r = 0; r < n; ++r)
            if (r != i) {
                double factor = A[r][i];
                for (size_t c = 0; c < n; ++c) {
                    A[r][c] -= factor * A[i][c];
                    I[r][c] -= factor * I[i][c];
                }
            }
    }
    return I;
}

// ===============================================================
//  CLASE PRINCIPAL: LINEAR REGRESSION
// ===============================================================
class LinearRegression {
public:
    enum FitMethod { NORMAL_EQUATION, GRADIENT_DESCENT };

private:
    Vec weights_;           // Pesos del modelo
    double bias_ = 0.0;     // Término independiente
    bool fitted_ = false;   // Indica si el modelo fue entrenado
    bool scale_features_ = true;
    StandardScaler scaler_; // Escalador de datos
    FitMethod method_ = NORMAL_EQUATION;
    double lr_ = 0.01;      // Tasa de aprendizaje
    size_t epochs_ = 2000;  // Iteraciones para gradiente

public:
    // ===========================================================
    //  MÉTODO FIT() - Entrenamiento del modelo
    // ===========================================================
    void fit(const vector<vector<double>>& X_raw, const vector<double>& y) {
        if (X_raw.empty()) throw runtime_error("X vacío");

        size_t n = X_raw.size(), d = X_raw[0].size();
        vector<vector<double>> X = X_raw;

        // --- BLOQUE DE DATA SCALING: normaliza los datos si está activado
        if (scale_features_) {
            scaler_.fit(X);           // Calcula medias y desviaciones
            X = scaler_.transform(X); // Estandariza X
        }

        // --- OPCIÓN 1: Resolver por ecuación normal
        if (method_ == NORMAL_EQUATION) {
            Matrix Xb(n, vector<double>(d + 1, 1.0));
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < d; ++j)
                    Xb[i][j] = X[i][j];

            Matrix Xt = transpose(Xb);
            Matrix XtX = matmul(Xt, Xb);
            Matrix XtX_inv = inverse(XtX);

            Matrix Y(n, vector<double>(1));
            for (size_t i = 0; i < n; ++i)
                Y[i][0] = y[i];

            Matrix theta = matmul(matmul(XtX_inv, Xt), Y);

            weights_.assign(d, 0.0);
            for (size_t j = 0; j < d; ++j)
                weights_[j] = theta[j][0];
            bias_ = theta[d][0];
            fitted_ = true;
        }
        // --- OPCIÓN 2: Resolver por gradiente descendente
        else {
            weights_.assign(d, 0.0);
            bias_ = 0.0;
            for (size_t it = 0; it < epochs_; ++it) {
                double db = 0.0;
                vector<double> dw(d, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    double yhat = bias_;
                    for (size_t j = 0; j < d; ++j)
                        yhat += weights_[j] * X[i][j];
                    double err = yhat - y[i];
                    db += err;
                    for (size_t j = 0; j < d; ++j)
                        dw[j] += err * X[i][j];
                }
                db = (2.0 / n) * db;
                for (size_t j = 0; j < d; ++j)
                    dw[j] = (2.0 / n) * dw[j];
                bias_ -= lr_ * db;
                for (size_t j = 0; j < d; ++j)
                    weights_[j] -= lr_ * dw[j];
            }
            fitted_ = true;
        }
    }

    // ===========================================================
    //  MÉTODO PREDICT() - Realiza predicciones con el modelo
    // ===========================================================
    vector<double> predict(const vector<vector<double>>& X_raw) const {
        if (!fitted_) throw runtime_error("El modelo no ha sido entrenado");

        vector<vector<double>> X = X_raw;

        // --- BLOQUE DE DATA SCALING (usar el mismo escalado del fit)
        if (scale_features_)
            X = scaler_.transform(X);

        vector<double> yhat(X.size(), 0.0);
        for (size_t i = 0; i < X.size(); ++i) {
            double pred = bias_;
            for (size_t j = 0; j < X[i].size(); ++j)
                pred += weights_[j] * X[i][j];
            yhat[i] = pred;
        }
        return yhat;
    }

    // ===========================================================
    //  MÉTODO SCORE() - Evalúa el rendimiento del modelo
    // ===========================================================
    Score score(const vector<vector<double>>& X, const vector<double>& y) const {
        vector<double> yhat = predict(X);  // Predicciones del modelo
        size_t n = y.size();
        double mean_y = accumulate(y.begin(), y.end(), 0.0) / (double)n;

        double ss_tot = 0.0, ss_res = 0.0, mae = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = y[i] - yhat[i];
            ss_res += diff * diff;       // Error cuadrático
            mae += fabs(diff);           // Error absoluto
            double d2 = y[i] - mean_y;
            ss_tot += d2 * d2;           // Varianza total
        }

        Score s;
        s.mse = ss_res / (double)n;
        s.mae = mae / (double)n;
        s.r2 = (ss_tot == 0.0) ? 1.0 : 1.0 - ss_res / ss_tot;
        return s;
    }
};

// ===============================================================
//  FUNCIONES PARA LEER CSV Y DIVIDIR TRAIN/TEST
// ===============================================================
struct Dataset {
    vector<vector<double>> X;
    vector<double> y;
    vector<string> feature_names;
    string target_name;
};

static vector<string> split_csv_line(const string& line) {
    vector<string> cells;
    string cell;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '\"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { cells.push_back(cell); cell.clear(); }
        else cell.push_back(c);
    }
    cells.push_back(cell);
    return cells;
}

static double to_double(string s) {
    s.erase(remove(s.begin(), s.end(), '%'), s.end());
    s.erase(remove_if(s.begin(), s.end(),
        [](unsigned char ch){ return ch < 32 || ch > 126; }),
        s.end());
    if (s.empty()) return 0.0;
    return stod(s);
}

static Dataset read_csv_xy(const string& path) {
    ifstream fin(path);
    if (!fin) throw runtime_error("No se pudo abrir el archivo: " + path);
    string line;
    vector<string> headers;
    getline(fin, line);
    headers = split_csv_line(line);

    vector<vector<double>> rows;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto cells = split_csv_line(line);
        if (cells.size() != headers.size()) continue;
        vector<double> row;
        for (auto& c : cells) row.push_back(to_double(c));
        rows.push_back(row);
    }

    Dataset ds;
    size_t n = rows.size(), m = rows[0].size();
    ds.feature_names.assign(headers.begin(), headers.end() - 1);
    ds.target_name = headers.back();
    ds.X.assign(n, vector<double>(m - 1));
    ds.y.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m - 1; ++j)
            ds.X[i][j] = rows[i][j];
        ds.y[i] = rows[i][m - 1];
    }
    return ds;
}

static void train_test_split(const vector<vector<double>>& X, const vector<double>& y,
                             double test_ratio,
                             vector<vector<double>>& Xtr, vector<double>& ytr,
                             vector<vector<double>>& Xte, vector<double>& yte,
                             unsigned seed = 42) {
    size_t n = X.size();
    vector<size_t> idx(n);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(seed);
    shuffle(idx.begin(), idx.end(), rng);
    size_t ntest = (size_t)round(test_ratio * n);
    for (size_t k = 0; k < n; ++k) {
        size_t i = idx[k];
        if (k < ntest) { Xte.push_back(X[i]); yte.push_back(y[i]); }
        else { Xtr.push_back(X[i]); ytr.push_back(y[i]); }
    }
}

// ===============================================================
//  FUNCIÓN DEMO - Entrena y evalúa con un dataset CSV
// ===============================================================
static void demo_dataset(const string& path, bool scale, LinearRegression::FitMethod method) {
    cout << "\n=== Ejecutando demo con: " << path << " ===\n";
    Dataset ds = read_csv_xy(path);

    cout << "Características (" << ds.feature_names.size() << "): ";
    for (auto& n : ds.feature_names) cout << n << " ";
    cout << "\nVariable objetivo: " << ds.target_name << "\n";

    vector<vector<double>> Xtr, Xte;
    vector<double> ytr, yte;
    train_test_split(ds.X, ds.y, 0.2, Xtr, ytr, Xte, yte, 2025);

    LinearRegression lr(scale, method, 0.05, 5000);
    lr.fit(Xtr, ytr);

    Score sc = lr.score(Xte, yte);
    cout << fixed << setprecision(4);
    cout << "R^2=" << sc.r2 << " | MSE=" << sc.mse << " | MAE=" << sc.mae << "\n";
}

// ===============================================================
//  FUNCIÓN PRINCIPAL - MAIN
// ===============================================================
int main() {
    try {
        // Cambiar las rutas a donde tengas tus CSV
        demo_dataset("Ice_cream_selling_data.csv", true, LinearRegression::NORMAL_EQUATION);
        demo_dataset("student_exam_scores.csv", true, LinearRegression::NORMAL_EQUATION);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}