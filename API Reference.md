```markdown
# API Reference (русская версия)

## Основные классы

### `QuantumSimulatorCUDA` - Главный класс симулятора

**Инициализация:**
```python
sim = QuantumSimulatorCUDA(
    num_qubits=5,               # Количество кубитов
    qubit_props=None,           # Свойства кубитов (опционально)
    optimization_level=2,       # Уровень оптимизации (0-2)
    sparse_threshold=None       # Порог для разреженного режима
)
```

**Основные методы:**
| Метод | Описание | Пример |
|-------|----------|--------|
| `apply_gate()` | Применение квантового гейта | `sim.apply_gate('h', 0)` |
| `measure()` | Измерение кубита | `result = sim.measure(0)` |
| `get_state()` | Получение вектора состояния | `state = sim.get_state()` |
| `reset()` | Сброс состояния | `sim.reset()` |

### `QuantumController` - Графический интерфейс

**Запуск:**
```python
app = QApplication(sys.argv)
controller = QuantumController()
controller.show()
sys.exit(app.exec_())
```

## Квантовые операции

### Базовые гейты
```python
# Однокубитные
sim.apply_gate('h', 0)       # Адамара
sim.apply_gate('x', 1)       # Паули-X
sim.apply_gate('ry', 2, theta=np.pi/4)  # Вращение Y

# Двухкубитные
sim.apply_gate('cx', 0, 1)   # CNOT
sim.apply_gate('cz', 1, 2)   # CZ
```

### Алгоритмы (из `quantum_extensions.py`)

**Алгоритм Гровера:**
```python
def oracle(state):
    return state == [1,0,1]  # Ищем |101>

result = grovers_algorithm(
    simulator=sim,
    oracle=oracle,
    iterations=None  # Автоподбор
)
```

**Алгоритм Шора:**
```python
factors = shors_algorithm_fixed(sim, N=15)
# Возвращает [3, 5] для N=15
```

## Работа с API (`QuantumAPIHandler`)

### Настройка подключения
```python
api = QuantumAPIHandler()
api.set_api_key(
    provider='ibmq',          # или 'rigetti', 'ionq'
    key='ваш_api_ключ',
    encrypt=True              # Шифрование ключа
)
```

### Отправка заданий
```python
job = api.execute(
    simulator=sim,
    backend='ibmq_qasm_simulator',
    shots=1024,
    parameters={...}
)
```

## Визуализация (`QuantumVisualizer`)

**Примеры:**
```python
vis = QuantumVisualizer(sim)

# 3D график амплитуд
vis.plot_3d_state("state.png", angle=(30, 45))

# Граф запутанности
vis.plot_entanglement_graph(
    filename="entanglement.png",
    threshold=0.1  # Порог для отображения связей
)
```

## Управление ошибками

### Поверхностный код
```python
from quantum_extensions import SurfaceCodeCorrector

corrector = SurfaceCodeCorrector(
    simulator=sim,
    distance=3  # Размер кода
)
corrector.stabilize(rounds=5)
```

### Модель шумов NISQ
```python
noise_model = NISQNoiseModel({
    0: QubitProperties(t1=50.0, t2=30.0),
    1: QubitProperties(t1=70.0, t2=40.0)
})
sim = QuantumSimulatorCUDA(noise_model=noise_model)
```

## Системные методы

### Аппаратная конфигурация
```python
hw_info = sim.get_hardware_report()
"""
{
    'max_theoretical_qubits': 28,
    'used_qubits': 5,
    'backend': 'GPU',
    'gpu_memory_used': '2.1 GB'
}
"""
```

### Производительность
```python
stats = sim.get_performance_stats()
"""
{
    'total_operations': 120,
    'average_gate_time': 0.0032,
    'error_rate': 0.001
}
"""
```

## Исключения

| Класс ошибки | Описание |
|--------------|----------|
| `QubitValidationError` | Некорректный индекс кубита |
| `BackendError` | Проблемы с GPU/MPI |
| `MemoryError` | Недостаточно памяти |
| `APIError` | Ошибки подключения к квантовым API |

---

Для более детальной информации смотрите docstrings в коде или [примеры использования](examples/).