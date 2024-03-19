#include "train.hpp"


Data sampleData(Mesh& mesh, int num_samples = 50000) {
    Matrix points(num_samples, 3);
    Matrix distances(num_samples, 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    omp_set_num_threads(64);

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


int main() {

    SIREN model("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_arch.txt");
    std::cout << "Built model" << std::endl;

    // model.loadWeights("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf2_weights.bin");
    // std::cout << "Loaded weights" << std::endl;

    // model.loadWeights("/home/evgeny.sidorov/Graph6/task2/weights/ckpt2000.bin");
    // std::cout << "Loaded weights" << std::endl;

    Data data = loadData("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf2_points.bin");
    std::cout << "Loaded data" << std::endl;

    // train(model, data, {512, 15000, 10});

    // test(model, data);
    //  std::cout << "Tested model" << std::endl;

    // generate_scene(model, "task2_references/cam1.txt", "task2_references/light.txt", 32);

    // Triangle triangle({0, 0, 0}, {1, 1, 0}, {1, 0, 0});
    Mesh mesh("task2_references/cup_1n.obj");
    std::cout << "Loaded mesh" << std::endl;
    // // Mesh mesh(triangle);

    // Data data = sampleData(mesh);
    // printRandomSamples(data);

    train(model, data, {512, 15000, 10});  

    generate_scene(model, mesh, "task2_references/cam1.txt", "task2_references/light.txt", 64);

    return 0;
}
