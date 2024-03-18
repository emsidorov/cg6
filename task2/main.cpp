#include "inference.hpp"
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "public_image.h"
#include <chrono>
#include <omp.h>
#include <memory>
#include <fstream>
#include "public_camera.h"


float sphereDistance(const glm::vec3& point) {
    return glm::length(point - glm::vec3(0.0f, 0.0f, 0.0f)) - 1.0f;
}


glm::vec3 sphereNormal(const glm::vec3& point) {
    return glm::normalize(point - glm::vec3(0.0f, 0.0f, 0.0f));
}


struct Scene {
    nsdf::Camera camera;
    nsdf::DirectedLight light;
};


glm::vec3 cross(glm::vec3 &vec1, glm::vec3 &vec2) 
{ 
    return glm::vec3(
        vec1.y * vec2.z - vec1.z * vec2.y, 
        vec1.z * vec2.x - vec1.x * vec2.z, 
        vec1.x * vec2.y - vec1.y * vec2.x
    ); 
}


float sdf(SIREN& model, const glm::vec3 &point) {
    float distance;
    if (point.x < -1 || point.x > 1 || point.y < -1 || point.y > 1 || point.z < -1 || point.z > 1) {
        // Подсчитываем расстояние до границы куба, т.к. объекторв вне единичного куба нет (или используем минимальный шаг)
        glm::vec3 outsideDist = max(glm::abs(point) - glm::vec3(1.0, 1.0, 1.0), 0.01f);
        distance = outsideDist.length(); 
    } else {
        Matrix x(point);
        Matrix y = model.forward(x);
        distance = y(0, 0);
    }
    return distance;
}


glm::vec3 getNormal(const glm::vec3& p, SIREN& model, float epsilon = 1e-4) {
    float sdfX = sdf(model, glm::vec3(p.x + epsilon, p.y, p.z)) - sdf(model, glm::vec3(p.x - epsilon, p.y, p.z));
    float sdfY = sdf(model, glm::vec3(p.x, p.y + epsilon, p.z)) - sdf(model, glm::vec3(p.x, p.y - epsilon, p.z));
    float sdfZ = sdf(model, glm::vec3(p.x, p.y, p.z + epsilon)) - sdf(model, glm::vec3(p.x, p.y, p.z - epsilon));

    glm::vec3 normal(sdfX, sdfY, sdfZ);
    return glm::normalize(normal);
}


glm::vec3 trace(
    SIREN& model,
    glm::vec3 &lightDir,
    glm::vec3 &cameraPos,
    glm::vec3 &rayDir
) {
    float t = 0.0f, distance;
    for (int i = 0; i < 100; ++i) {
        glm::vec3 point = cameraPos + t * rayDir;
        float distance = sdf(model, point);
        // float distance = sphereDistance(point);

        if (distance < 0.00001f) {
            glm::vec3 normal = getNormal(point, model);
            // glm::vec3 normal = sphereNormal(point);
            float diffuse = std::max(glm::dot(normal, lightDir), 0.08f);

            return glm::vec3(diffuse, diffuse, diffuse);
        }
        t += distance;
        if (t >= 100.0f) break;
    }

    return glm::vec3(0.0f, 0.0f, 0.0f);
}


void generate_scene(SIREN& model, Scene scene, int numThreads) {
    int width = 512, height = 512;
    float* output = new float[width * height * 3];
    size_t imageSize = width * height * 3 * sizeof(float);
    glm::vec3 cameraPos(scene.camera.pos_x, scene.camera.pos_y, scene.camera.pos_z), lightDir(scene.light.dir_x, scene.light.dir_y, scene.light.dir_z);

    // Расчёт базиса камеры
    glm::vec3 pos(scene.camera.pos_x, scene.camera.pos_y, scene.camera.pos_z); // Позиция камеры
    glm::vec3 target(0.0f, 0.0f, 0.0f); // Цель камеры
    glm::vec3 up(0.0f, 1.0f, 0.0f); // Вектор "вверх"

    glm::vec3 view = glm::normalize(target - pos); // Направление взгляда
    glm::vec3 right = glm::normalize(cross(view, up));
    up = cross(right, view);

    // Расчёт направления луча для каждого пикселя
    float aspectRatio = float(width) / height;
    float scale = tan(scene.camera.fov_rad / 2.0f); // FOV в радианах

    omp_set_num_threads(numThreads);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {
            float x = (2.0f * (i + 0.5f) / width - 1.0f) * aspectRatio * scale;
            float y = (2.0f * (j + 0.5f) / height - 1.0f) * scale;

            glm::vec3 rayDir = glm::normalize(view + right * x + up * y);

            int index = 3 * (i + j * width);

            glm::vec3 color = trace(model, lightDir, cameraPos, rayDir);

            output[index] = color.x;
            output[index + 1] = color.y;
            output[index + 2] = color.z;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() << " ms\n";

    unsigned char* image = new unsigned char[width * height * 3];
    for (int i = 0; i < width * height * 3; ++i) {
        image[i] = static_cast<unsigned char>(255 * output[i]); // Предполагается, что output нормализован от 0 до 1
    }

    // Сохранение изображения
    stbi_write_png("out_cpu.png", width, height, 3, image, width * 3);

    std::cout << "Image saved as out_cpu.png" << std::endl;

    delete[] image;
}

Scene loadScene(const std::string& cameraFile, const std::string& lightFile) {
    Scene scene;
    scene.camera.from_file(cameraFile.c_str());
    scene.light.from_file(lightFile.c_str());
    return scene;
}


int main() {

    SIREN model("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_arch.txt");
    std::cout << "Built model" << std::endl;

    model.loadWeights("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_weights.bin");
    std::cout << "Loaded weights" << std::endl;

    Data data = loadData("/home/evgeny.sidorov/Graph6/NAIR_2024/task2_references/sdf1_test.bin");
    std::cout << "Loaded data" << std::endl;

    test(model, data);
     std::cout << "Tested model" << std::endl;

    auto scene = loadScene("task2_references/cam1.txt", "task2_references/light.txt");
    generate_scene(model, scene, 8);

    return 0;
}
