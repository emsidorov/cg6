#include "matrix.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>


class Layer {
public:
    virtual ~Layer() {}
    virtual void loadWeights(std::ifstream& file) = 0;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad) = 0;
    virtual void printWeights() const = 0; // Добавленный метод
};


class DenseLayer : public Layer {
public:
    Matrix weights, biases;
    float learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, beta1_t = beta1, beta2_t = beta2;
    Matrix m_weights, v_weights;
    Matrix m_biases, v_biases;
    Matrix input_cache;

    DenseLayer(size_t input_size, size_t output_size) : weights(output_size, input_size), biases(1, output_size),
               m_weights(output_size, input_size), v_weights(output_size, input_size), m_biases(1, output_size), 
               v_biases(1, output_size) {}

    void loadWeights(std::ifstream& file) {
        for (size_t i = 0; i < weights.rows; ++i) {
            for (size_t j = 0; j < weights.cols; ++j) {
                file.read(reinterpret_cast<char*>(&weights(i, j)), sizeof(float));
            }
        }
        for (size_t i = 0; i < biases.rows; ++i) {
            for (size_t j = 0; j < biases.cols; ++j) {
                file.read(reinterpret_cast<char*>(&biases(i, j)), sizeof(float));
            }
        }
    }

    void printWeights() const override {
        for (size_t i = 0; i < weights.rows; ++i) {
            for (size_t j = 0; j < weights.cols; ++j) {
                std::cout << weights(i, j) << " ";
            }
            std::cout << std::endl;
        }
        for (size_t i = 0; i < biases.rows; ++i) {
            std::cout << biases(i, 0) << std::endl;
        }
    }

    Matrix forward(const Matrix& input) {
        input_cache = input;
        Matrix output = Matrix::multiply(input, Matrix::transpose(weights));
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                output(i, j) += biases.data[j];
            }
        }
        return output;
    }

    void updateWeights(const Matrix& dW, const Matrix& db) {
        // Обновление моментов для весов
        m_weights = m_weights * beta1 + dW * (1 - beta1);
        v_weights = v_weights * beta2 + (dW * dW) * (1 - beta2); // Предполагаем, что операция ^ выполняет поэлементное возведение в квадрат

        // Исправление смещения для моментов
        Matrix m_hat_weights = m_weights / (1 - beta1_t);
        Matrix v_hat_weights = v_weights / (1 - beta2_t);

        // Обновление весов
        weights = weights - (m_hat_weights / (v_hat_weights.sqrt() + epsilon)) * learning_rate;

        // Похожие шаги для смещений
        m_biases = m_biases * beta1 + db * (1 - beta1);
        v_biases = v_biases * beta2 + (db * db) * (1 - beta2);

        Matrix m_hat_biases = m_biases / (1 - beta1_t);
        Matrix v_hat_biases = v_biases / (1 - beta2_t);

        biases = biases - (m_hat_biases / (v_hat_biases.sqrt() + epsilon)) * learning_rate;

        beta1_t *= beta1;
        beta2_t *= beta2;
    }

    Matrix backward(const Matrix& grad) {
        Matrix dW = Matrix::multiply(Matrix::transpose(grad), input_cache);

        Matrix db(biases.rows, biases.cols);
        for (int j = 0; j < grad.cols; ++j) {
            for (int i = 0; i < grad.rows; ++i) {
                db.data[j] += grad(i, j);
            }
        }

        updateWeights(dW, db);
        Matrix dInput = Matrix::multiply(grad, weights);
        return dInput;
    }
};


class SineLayer : public Layer {
private:
    float w0; // Масштабирующий коэффициент
    Matrix prod_cache;

public:
    SineLayer(float w0 = 30.0) : w0(w0) {}

    void loadWeights(std::ifstream& file) {
        return;
    }

    void printWeights() const override {
        return;
    }

    Matrix forward(const Matrix& input) override {
        Matrix prod = input * w0;
        prod_cache = prod;
        Matrix output(input.rows, input.cols); // Создаём выходную матрицу такого же размера, как и входная
        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = std::sin(prod.data[i]);
        }
        return output;
    }

    Matrix backward(const Matrix& grad) {
        Matrix dSine_dInput(prod_cache.rows, prod_cache.cols);
        for (size_t i = 0; i < prod_cache.data.size(); ++i) {
            dSine_dInput.data[i] = w0 * std::cos(prod_cache.data[i]);
        }
        return dSine_dInput * grad;
    }
};


class MSE {
public:
    Matrix diff;

    Matrix forward(const Matrix& predictions, const Matrix& targets) {
        assert(predictions.rows == targets.rows && predictions.cols == targets.cols);
        diff = predictions - targets;
        float N = static_cast<float>(predictions.data.size());
        auto mse = (diff * diff).sum() / N;
        return mse;
    }

    Matrix backward() {
        float N = static_cast<float>(diff.data.size());
        Matrix grad = diff * (2.0f / N);
        return grad;
    }
};


class SIREN {
private:
    std::vector<Layer*> layers;

public:
    SIREN(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string layerType;
            size_t inputSize, outputSize;

            iss >> layerType;

            if (layerType == "Dense") {
                iss.ignore(std::numeric_limits<std::streamsize>::max(), '(');
                iss >> inputSize;
                iss.ignore(std::numeric_limits<std::streamsize>::max(), ')');
                iss.ignore(std::numeric_limits<std::streamsize>::max(), '(');
                iss >> outputSize;

                layers.push_back(new DenseLayer(inputSize, outputSize));
            } else if (layerType == "Sin") {
                float w0 = 30.0;
                layers.push_back(new SineLayer(w0));
            }
        }
        file.close();
    }

    ~SIREN() {
        for (Layer* layer : layers) {
            delete layer;
        }
    }

    void loadWeights(const std::string& filename) {
        std::ifstream weightsFile(filename, std::ios::binary);
        for (auto layer : layers) {
            layer->loadWeights(weightsFile);
        }
        return;
    }

    void printWeights() const {
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "Layer " << i << ":" << std::endl;
            layers[i]->printWeights();
            std::cout << std::endl;
        }
    }


    Matrix forward(const Matrix& input) {
        Matrix output = input;
        for (Layer* layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    Matrix backward(const Matrix& grad) {
        auto layer_grad = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            layer_grad = (*it)->backward(layer_grad); // Обновляем grad на каждом шаге
        }
        return layer_grad;
    }
};
