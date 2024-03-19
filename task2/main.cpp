#include "trace.hpp"
#include <random>


struct TrainParams {
    int batch_size, num_steps, log_iter = 100;
};


Data getBatch(const Data& data, int batchSize) {
    std::random_device rd;
    std::mt19937 gen(rd());

    int N = data.x.rows, input_size = data.x.cols, output_size = data.y.cols;
    std::uniform_int_distribution<> dis(0, N - 1);

    Matrix batchX(batchSize, input_size);
    Matrix batchY(batchSize, output_size);

    for (int i = 0; i < batchSize; ++i) {
        int idx = dis(gen);
        for (int j = 0; j < input_size; ++j) {
            batchX(i, j) = data.x(idx, j);
        }
        for (int j = 0; j < output_size; ++j) {
            batchY(i, j) = data.y(idx, j);
        }
    }

    return {batchX, batchY};
}

void printRandomSamples(const Data& data, int num_samples=10) {
    std::random_device rd;
    std::mt19937 gen(rd());

    int N = data.x.rows, input_size = data.x.cols, output_size = data.y.cols;
    std::uniform_int_distribution<> dis(0, N - 1);

    std::cout << N << std::endl;

    for (int i = 0; i < num_samples; ++i) {
        int idx = dis(gen);
        std::cout << "Point " << i << ":" << std::endl;
        for (int j = 0; j < input_size; ++j) {
            std::cout << data.x(idx, j) << " ";
        }
        std::cout << std::endl;
        for (int j = 0; j < output_size; ++j) {
            std::cout << data.y(idx, j) << " ";
        }
        std::cout << std::endl;
    }

    return;
}


void train(
    SIREN& model, 
    Data& data,
    const TrainParams& params
) {
    int N = data.x.rows;
    auto mse = MSE();
    float running_loss = 0.0f;

    for (int i = 0; i < params.num_steps; ++i) {
        Data batch = getBatch(data, params.batch_size);

        auto output = model.forward(batch.x);
        auto loss = mse.forward(output, batch.y);
        auto mse_grad = mse.backward();
        auto model_grad = model.backward(mse_grad);

        if (i == 0) {
            running_loss = loss(0, 0);
        } else {
            running_loss = running_loss * 0.9 + loss(0, 0) * 0.1;
        }

        if ((i > 0) && (i % params.log_iter == 0)) {
            std::cout << "Iter: " << i << ", Loss: " << loss(0, 0) << ", Prediction: " << output(0, 0) << std::endl;
        }
    }
}


int main() {

    SIREN model("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_arch.txt");
    std::cout << "Built model" << std::endl;

    model.loadWeights("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_weights.bin");
    std::cout << "Loaded weights" << std::endl;

    Data data = loadData("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_points.bin");
    std::cout << "Loaded data" << std::endl;

    printRandomSamples(data);
    train(model, data, {512, 15000, 10});

    // test(model, data);
    //  std::cout << "Tested model" << std::endl;

    // generate_scene(model, "task2_references/cam2.txt", "task2_references/light.txt", 32);

    return 0;
}
