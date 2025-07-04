```markdown
# Квантовый симулятор с гибридным управлением

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)

**Полнофункциональный симулятор квантовых вычислений** с поддержкой GPU/CPU/MPI, визуализацией и коррекцией ошибок.

## 🔥 Ключевые особенности

- **Автоматическая оптимизация** под железо (анализ CPU/GPU/RAM)
- **3 режима работы**: плотные матрицы (до 28 кубитов), разреженные (до 50), тензорные сети (50+)
- **Реальные квантовые API**: IBMQ, Rigetti, IonQ через единый интерфейс
- **Готовые алгоритмы**: Шор, Гровер, VQE, QAOA
- **NISQ-реализм**: моделирование шумов и утечек
- **3D-визуализация** состояний и запутанности

## 🚀 Быстрый старт

### Установка
```bash
pip install -r requirements.txt
```

### Запуск GUI
```bash
python quantum_controller.py
```

### Пример в коде
```python
from quantum_core import QuantumSimulatorCUDA

sim = QuantumSimulatorCUDA(num_qubits=5)
sim.apply_gate('h', 0)
print("Результат:", sim.measure(0))
```

## 📊 Возможности

### 1. Аппаратно-оптимизированные вычисления
```python
# Автоподбор режима
sim = QuantumSimulatorCUDA(num_qubits=30)  # Автоматически использует sparse-режим

# Принудительный выбор
sim.quantum_state.change_backend('tensor_network')
```

### 2. Работа с реальными квантовыми устройствами
```python
from quantum_core import QuantumAPIHandler

api = QuantumAPIHandler()
api.set_api_key('ibmq', 'ваш_ключ')
job = api.execute(sim, backend='ibmq_qasm_simulator')
```

### 3. Визуализация
```python
vis = QuantumVisualizer(sim)
vis.plot_3d_state("state.png")
vis.plot_entanglement_graph(threshold=0.2)
```

## 🛠 Технологический стек

| Компонент | Описание |
|-----------|----------|
| **CUDA** | GPU-ускорение через CuPy |
| **MPI** | Распределенные вычисления |
| **PyQt5** | Графический интерфейс |
| **NetworkX** | Анализ графов запутанности |

## 📈 Производительность

Конфигурация | Макс. кубитов | Время гейта (мс)
------------|--------------|-------------
CPU (i9) | 28 | 0.5
GPU (RTX 3090) | 30 | 0.1
MPI (4 узла) | 35 | 0.3

## 📚 Документация

- [API Reference](docs/api.md)
- [Примеры алгоритмов](examples/)
- [Настройка оборудования](docs/hardware.md)

## 🤝 Как внести вклад

1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/AmazingFeature`)
3. Закоммитьте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Запушите (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📜 Лицензия

MIT © 2023 Quantum Research Team

---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your/repo)
[![Report Bug](https://img.shields.io/badge/Report-Bug-red.svg)](https://github.com/your/repo/issues)
```