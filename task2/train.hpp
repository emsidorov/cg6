#include "trace.hpp"

struct TrainParams {
    int batch_size, num_steps, log_iter, checkpoint_iter, render_iter;
    float lr;

    // Конструктор, который считывает параметры из файла
    TrainParams(const std::string& filePath) : log_iter(100), checkpoint_iter(100), lr(0.00005f), render_iter(1000) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Не удалось открыть файл: " << filePath << std::endl;
            // Возможно, стоит здесь добавить логику обработки ошибки открытия файла
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            if (!(iss >> key)) { continue; } // Пропустить пустые строки

            if (key == "batch_size") {
                iss >> batch_size;
            } else if (key == "num_steps") {
                iss >> num_steps;
            } else if (key == "log_iter") {
                iss >> log_iter;
            } else if (key == "checkpoint_iter") {
                iss >> checkpoint_iter;
            } else if (key == "render_iter") {
                iss >> render_iter;
            } else if (key == "learning_rate") {
                iss >> lr;
            } else {
                std::cerr << "Неизвестный параметр: " << key << std::endl;
            }
        }
    }
};


Data sampleData(Mesh& mesh, int num_samples = 50000) {
    Matrix points(num_samples, 3);
    Matrix distances(num_samples, 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // auto start = std::chrono::high_resolution_clock::now();
    int count = 0;

    #pragma omp parallel for
    for (int i = 0; i < num_samples; ++i) {
        points(i, 0) = dis(gen);
        points(i, 1) = dis(gen);
        points(i, 2) = dis(gen);

        distances(i, 0) = mesh.distance(glm::vec3(points(i, 0), points(i, 1), points(i, 2)));
        if (distances(i, 0) < 0) {
            count++;
        }
    }
    std::cout << "Num negatives: " << count << std::endl;

    return {points, distances};
}


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

    int count = 0;

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

    for (int i = 0; i < data.x.rows; ++i) {
        if (data.y(i, 0) < 0) {
            count++;
        }
    }

    std::cout << "N samples: " << N << ", Negatives: " << count << std::endl;

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
    model.setLR(params.lr);

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

        if ((i + 1) % params.log_iter == 0) {
            std::cout << "Iter: " << i + 1 << ", Loss: " << loss(0, 0) << ", Prediction: " << output(0, 0) << std::endl;
        }

        if ((i + 1) % params.checkpoint_iter == 0) {
            std::ostringstream ckptPath;
            ckptPath << "weights/ckpt" << i + 1 << ".bin";
            model.saveWeights(ckptPath.str());
            std::cout << "Saved checkpoint to " << ckptPath.str() << std::endl;
        }

        if ((i + 1) % params.render_iter == 0) {
            std::ostringstream ckptPath;
            ckptPath << "renders2/step" << i + 1 << ".png";
            Mesh mesh;
            render(model, mesh, "task2_references/cam1.txt", "task2_references/light.txt", ckptPath.str(), 64);
        }
    }
}