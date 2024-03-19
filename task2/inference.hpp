#include "network.hpp"


struct Data {
    Matrix x;
    Matrix y;
};

Data loadData(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open test data file.");
    }

    int N;
    file.read(reinterpret_cast<char*>(&N), sizeof(N));

    Matrix x(N, 3);
    Matrix y(N, 1);

    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j) {
            file.read(reinterpret_cast<char*>(&x(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < y.rows; ++i) {
        for (int j = 0; j < y.cols; ++j) {
            file.read(reinterpret_cast<char*>(&y(i, j)), sizeof(float));
        }
    }

    return {x, y};
}

void test(SIREN& model, const Data& data) {
    Matrix output = model.forward(data.x);
    float error = (output - data.y).abs().max();

    if (error < 1e-5) {
        std::cout << "Test passed successfully. Maximum error: " << error << std::endl;
    } else {
        std::cout << "Test failed. Maximum error: " << error << std::endl;
    }
}