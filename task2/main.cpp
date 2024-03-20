#include "train.hpp"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Недостаточно аргументов" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        if (argc != 8) {
            std::cerr << "Для режима обучения требуется arch.txt, file.obj, train_params.txt, cam.txt, light.txt, num_threads" << std::endl;
            return 1;
        }
        std::string archPath = argv[2];
        std::string objPath = argv[3];
        std::string trainPath = argv[4];
        std::string camPath = argv[5];
        std::string lightPath = argv[6];
        int num_threads = std::stoi(argv[7]);
        omp_set_num_threads(num_threads);

        SIREN model(archPath);
        Mesh mesh(objPath);
        Data data = sampleData(mesh, 50000);
        TrainParams params(trainPath);
        train(model, data, params, camPath, lightPath);
        render(model, camPath, lightPath, "train_results/render.png", 512);
        model.saveWeights("train_results/weights.bin");
    } else if (mode == "render") {
        if (argc != 7) {
            std::cerr << "Для режима рендера требуются arch.txt, weights.bin, cam.txt, light.txt, num_threads" << std::endl;
            return 1;
        }
        std::string archPath = argv[2];
        std::string weightsPath = argv[3];
        std::string camPath = argv[4];
        std::string lightPath = argv[5];
        int num_threads = std::stoi(argv[6]);
        omp_set_num_threads(num_threads);

        SIREN model(archPath);
        model.loadWeights(weightsPath);
        render(model, camPath, lightPath, "render_results/out_cpu.png", 512);
    } else {
        std::cerr << "Неизвестный режим. Используйте 'train' для обучения или 'render' для рендера." << std::endl;
        return 1;
    }

    return 0;
}
