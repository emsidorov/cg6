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
    virtual void printWeights() const = 0; // Добавленный метод
};



class DenseLayer : public Layer {
public:
    Matrix weights;
    Matrix biases;

    DenseLayer(size_t input_size, size_t output_size) : weights(output_size, input_size), biases(1, output_size) {}

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
        std::cout << "Weights:" << std::endl;
        for (size_t i = 0; i < weights.rows; ++i) {
            for (size_t j = 0; j < weights.cols; ++j) {
                std::cout << weights(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Biases:" << std::endl;
        for (size_t i = 0; i < biases.rows; ++i) {
            std::cout << biases(i, 0) << std::endl;
        }
    }

    Matrix forward(const Matrix& input) {
        Matrix output = Matrix::multiply(input, Matrix::transpose(weights));
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                output(i, j) += biases.data[j];
            }
        }
        return output;
    }
};


class SineLayer : public Layer {
private:
    float w0; // Масштабирующий коэффициент

public:
    SineLayer(float w0 = 30.0) : w0(w0) {}

    void loadWeights(std::ifstream& file) {
        return;
    }

    void printWeights() const override {
        return;
    }


    Matrix forward(const Matrix& input) override {
        Matrix output(input.rows, input.cols); // Создаём выходную матрицу такого же размера, как и входная
        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = std::sin(w0 * input.data[i]);
        }
        return output;
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
};