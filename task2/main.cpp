#include "train.hpp"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Недостаточно аргументов" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        if (argc != 6) {
            std::cerr << "Для режима обучения требуется arch.txt, file.obj, train_params.txt num_threads" << std::endl;
            return 1;
        }
        std::string archPath = argv[2];
        std::string objPath = argv[3];
        std::string trainPath = argv[4];
        int num_threads = std::stoi(argv[5]);

        SIREN model(archPath);
        std::cout << "Built model" << std::endl;
        Mesh mesh(objPath);
        std::cout << "Built mesh" << std::endl;
        Data data = sampleData(mesh, num_threads=num_threads);
        std::cout << "Sample Data" << std::endl;
        TrainParams params(trainPath);
        std::cout << "Built train params" << std::endl;
        train(model, data, params);
        render(model, "task2_references/cam1.txt", "task2_references/light.txt", num_threads);
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

        SIREN model(archPath);
        model.loadWeights(weightsPath);
        render(model, camPath, lightPath, num_threads);
    } else {
        std::cerr << "Неизвестный режим. Используйте 'train' для обучения или 'render' для рендера." << std::endl;
        return 1;
    }

    return 0;
}


// int main() {

//     omp_set_num_threads(16);

//     SIREN model("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf2_arch.txt");
//     std::cout << "Built model" << std::endl;

//     // model.loadWeights("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf2_weights.bin");
//     // std::cout << "Loaded weights" << std::endl;

//     // model.loadWeights("/home/evgeny.sidorov/Graph6/task2/weights/ckpt10000.bin");
//     // std::cout << "Loaded weights" << std::endl;

//     // Data data = loadData("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf2_points.bin");
//     // std::cout << "Loaded data" << std::endl;

//     // train(model, data, {512, 15000, 10});

//     // test(model, data);
//     //  std::cout << "Tested model" << std::endl;

//     // generate_scene(model, "task2_references/cam1.txt", "task2_references/light.txt", 32);

//     // Triangle triangle({0, 0, 0}, {1, 1, 0}, {1, 0, 0});
//     Mesh mesh("task2_references/cup_1n.obj");
//     std::cout << "Loaded mesh" << std::endl;
//     // // Mesh mesh(triangle);

//     Data data = sampleData(mesh, 50000);
//     printRandomSamples(data);

//     TrainParams params("task2_references/train_params.txt");
//     train(model, data, params);  

//     render(model, mesh, "task2_references/cam1.txt", "task2_references/light.txt", "output.png", 64);

//     return 0;
// }
