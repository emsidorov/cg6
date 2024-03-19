#include "inference.hpp"

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>


float dot2( glm::vec3 v ) { return glm::dot(v,v); }
float clamp( float v, float min, float max) { return std::min(max, std::max(min, v)); }

class Triangle {
public:
    glm::vec3 v1, v2, v3;
    
    Triangle(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3) : v1(v1), v2(v2), v3(v3) {}

    float distance(const glm::vec3& p) const {
        glm::vec3 v21 = v2 - v1; glm::vec3 p1 = p - v1;
        glm::vec3 v32 = v3 - v2; glm::vec3 p2 = p - v2;
        glm::vec3 v13 = v1 - v3; glm::vec3 p3 = p - v3;
        glm::vec3 nor = cross( v21, v13 );


        return sqrt( // inside/outside test    
                 (glm::sign(glm::dot(glm::cross(v21,nor),p1)) + 
                  glm::sign(glm::dot(glm::cross(v32,nor),p2)) + 
                  glm::sign(glm::dot(glm::cross(v13,nor),p3))<2.0) 
                  ?
                  // 3 edges    
                  std::min( std::min( 
                  dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
                  dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0,1.0)-p2) ), 
                  dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0,1.0)-p3) )
                  :
                  // 1 face    
                  dot(nor,p1)*dot(nor,p1)/dot2(nor) );
    }

    bool is_inside(const glm::vec3& p) const {
        glm::vec3 nor = glm::normalize(glm::cross(v2 - v1, v3 - v1));
        glm::vec3 p_proj = p - glm::dot(p - v1, nor) * nor;
        float dotProduct = glm::dot(nor, p_proj - p);

        if (dotProduct > 0) {
            return false; // Точка лежит в плоскости по направлению нормали от треугольника
        } else {
            return true; // Точка лежит в противоположном направлении
        }
    }

};


class Mesh {
public:
    std::vector<Triangle> triangles;

    Mesh(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return;
        }

        std::string line;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> vertix_ids;
        while (getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;
            if (prefix == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(glm::vec3(x, y, z));
            } else if (prefix == "f") {
                int i0, i1, i2;
                int tmp; char slash;
                iss >> i0 >> slash >> tmp >> slash >> tmp;
                iss >> i1 >> slash >> tmp >> slash >> tmp;
                iss >> i2 >> slash >> tmp >> slash >> tmp;
                // std::cout << i0 << " " << i1 << " " << i2 << std::endl;
                // OBJ индексы начинаются с 1, поэтому вычитаем 1 для корректного индексирования
                vertix_ids.push_back(glm::vec3(i0, i1, i2));
            }
        }

        for (auto& vertex_idx : vertix_ids) {
            triangles.push_back(
                Triangle(vertices[vertex_idx.x - 1], vertices[vertex_idx.y - 1], vertices[vertex_idx.z - 1])
            );
        }
    }

    Mesh() {}

    Mesh(const Triangle& triangle) {
        triangles.push_back(triangle);
    }

    void addTriangle(const Triangle& triangle) {
        triangles.push_back(triangle);
    }

    float distance(const glm::vec3& point) const {
        float minDistance = std::numeric_limits<float>::max();
        bool isInside = true;
        for (const auto& triangle : triangles) {
            float dist = triangle.distance(point);
            if (dist < minDistance) {
                minDistance = dist;
                isInside = triangle.is_inside(point);
            }
        }
        // if (!isInside) {
        //     return -minDistance;
        // }
        return minDistance;
    }
};