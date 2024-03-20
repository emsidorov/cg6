# Сборка программы

```bash
g++ -fopenmp -o main main.cpp
```

# Запуск программы

## Обучение

```bash
./main train arch.txt, file.obj, train_params.txt, cam.txt, light.txt, num_threads
```
- **arch.txt** - файл с описанием архитектуры сети
- **file.obj** - файл с мешом
- **train_params.txt** - файл с параметрами обучения
- **cam.txt** - файл с параметрами камеры
- **light.txt** - файл с параметрами источника света
- **num_threads** - количество OpenMP нитей для ускорения программы

**train_params.txt** имеет следующую структру

```txt
batch_size 512
num_steps 20000
log_iter 10
checkpoint_iter 500
render_iter 500
learning_rate 0.00005
```

## Рендер

```bash
./main train arch.txt, weights.bin, cam.txt, light.txt, num_threads
```
- **arch.txt** - файл с описанием архитектуры сети
- **weights.bin** - файл с весами модели
- **cam.txt** - файл с параметрами камеры
- **light.txt** - файл с параметрами источника света
- **num_threads** - количество OpenMP нитей для ускорения программы

# Результаты работы программы

## Обучение

Результат сохраняется в директорию `train_results/`
Структуру директории

- `train_results/`
    - `renders` 
        - `stepN` - отрисовка сцены после N шага обучения (для ускорения промежуточная отрисовка делается 128x128)
    - `weights` 
        - `ckptN` - веса модели после N шага обучения
    - `render.png` - отрисовка сцены после окончания обучения
    - `weights.bin` - веса модели после окончания обучения

## Рендер

Результат сохраняется в `render_results/out_cpu.png`

# Примеры запуска программы

## Обучение

1. Обучние по мешу чашки

```bash
./main train task2_references/sdf2_arch.txt task2_references/cup_1n.obj task2_references/train_params.txt task2_references/cam1.txt task2_references/light.txt 32
```

2. Обучние по мешу сферы

```bash
./main train task2_references/sdf2_arch.txt task2_references/sphere.obj task2_references/train_params.txt task2_references/cam1.txt task2_references/light.txt 32
```

## Рендер

1. Рендер 1 SDF

```bash
./main render task2_references/sdf1_arch.txt task2_references/sdf1_weights.bin task2_references/cam1.txt task2_references/light.txt 64
```

2. Обучние по мешу сферы

```bash
./main render task2_references/sdf2_arch.txt task2_references/sdf2_weights.bin task2_references/cam1.txt task2_references/light.txt 64
```

# Спецификация сервера, где тестировалось
- Ubuntu 20.04
- Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
- NVIDIA A40