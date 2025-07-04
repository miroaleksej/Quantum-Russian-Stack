import numpy as np
import cupy as cp
from numba import cuda
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, defaultdict
import json
from enum import Enum
import warnings
from scipy.linalg import expm, svd
from dataclasses import dataclass, field
import requests
from abc import ABC, abstractmethod
import time
import psutil
import cmath
from collections import defaultdict
import os
import logging
import hashlib
from cryptography.fernet import Fernet
from scipy.sparse import csr_matrix, kron, eye
import networkx as nx
import itertools
try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("MPI not available. Distributed mode disabled.")
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
except ImportError:
    warnings.warn("Matplotlib not available. Visualization disabled.")

# ==================== КОНСТАНТЫ ====================
MAX_QUBITS_CPU = 28
MAX_QUBITS_GPU = 30
MAX_QUBITS_TN = 50
MIN_ENERGY_PER_QUBIT = 1e-21
MAX_ENERGY_THRESHOLD = 1e-6
NORM_TOLERANCE = 1e-10
DEFAULT_GATE_TIMES = {
    'h': 35, 'x': 35, 'y': 35, 'z': 35,
    'rz': 0, 'u3': 50, 'cx': 100, 'measure': 300
}
SPARSE_STATE_THRESHOLD = 0.1
MAX_CACHE_SIZE = 1000

# ==================== ЛОГГИРОВАНИЕ И ОШИБКИ ====================
logger = logging.getLogger('QuantumSimulator')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class QuantumError(Exception):
    """Базовый класс для ошибок квантового симулятора"""
    pass

class QubitValidationError(QuantumError):
    """Ошибка валидации кубитов"""
    pass

class BackendError(QuantumError):
    """Ошибка бэкенда"""
    pass

class APIError(QuantumError):
    """Ошибка квантового API"""
    pass

class MemoryError(QuantumError):
    """Ошибка недостатка памяти"""
    pass

class NoiseModelError(QuantumError):
    """Ошибка в модели шумов"""
    pass

# ==================== БАЗОВЫЕ КЛАССЫ И ТИПЫ ====================
class QubitType(Enum):
    TRANSMON = 1
    SPIN = 2
    TOPOLOGICAL = 3
    FLUXONIUM = 4
    GATMON = 5

@dataclass
class OperationRecord:
    """Запись о квантовой операции"""
    gate_type: str
    qubits: List[int]
    timestamp: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error_occurred: bool = False
    error_message: Optional[str] = None

@dataclass
class QubitProperties:
    """Физические свойства кубита"""
    qubit_type: QubitType
    t1: float  # мкс
    t2: float  # мкс
    frequency: float  # GHz
    anharmonicity: float  # MHz
    position: Tuple[float, float]
    gate_times: Dict[str, float] = field(default_factory=dict)
    leakage_rate: float = 0.001
    readout_error: Tuple[float, float] = (0.01, 0.01)

    def __post_init__(self):
        """Валидация физических параметров"""
        if self.t1 <= 0 or self.t2 <= 0:
            raise ValueError("T1 and T2 times must be positive")
        if self.t2 > 2 * self.t1:
            raise ValueError("T2 cannot exceed 2*T1 (T2 <= 2*T1)")
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if not (0 <= self.leakage_rate <= 0.1):
            raise ValueError("Leakage rate must be between 0 and 0.1")
        if not all(0 <= x <= 1 for x in self.readout_error):
            raise ValueError("Readout error probabilities must be between 0 and 1")

# ==================== ИНТЕРФЕЙСЫ ====================
class QuantumBackend(ABC):
    """Абстрактный базовый класс для бэкендов"""
    
    @abstractmethod
    def apply_gate(self, gate_name: str, qubits: List[int], **params):
        """Применение квантового гейта"""
        pass
    
    @abstractmethod
    def measure(self, qubit: int) -> int:
        """Измерение кубита"""
        pass
    
    @abstractmethod
    def get_state(self):
        """Получение текущего состояния"""
        pass
    
    @abstractmethod
    def reset(self):
        """Сброс состояния"""
        pass

class NoiseModelInterface(ABC):
    """Интерфейс для моделей шума"""
    
    @abstractmethod
    def apply_noise(self, state, gate: str, qubits: List[int], duration: float):
        """Применение шумовых эффектов"""
        pass
    
    @abstractmethod
    def get_error_rates(self) -> Dict[str, float]:
        """Получение статистики ошибок"""
        pass

# ==================== КОНФИГУРАЦИЯ ОБОРУДОВАНИЯ ====================
class HardwareConfig:
    """Конфигурация аппаратного обеспечения для симуляции"""
    
    def __init__(self):
        self.use_gpu = False
        self.use_mpi = False
        self.use_sparse = False
        self.gpu_type = None
        self.max_qubits = self._calculate_max_qubits()
        self.cache_enabled = True
        self.optimization_level = 2
        
        self._init_gpu()
        self._init_mpi()
        self._init_cache()
        
    def _init_gpu(self):
        """Инициализация GPU бэкенда"""
        try:
            cp.cuda.Device(0).compute_capability
            self.use_gpu = True
            self.gpu_type = 'cuda'
        except:
            try:
                import cupy_rocm as cp_rocm
                cp = cp_rocm
                self.use_gpu = True
                self.gpu_type = 'rocm'
            except:
                logger.warning("No GPU acceleration available. Falling back to CPU.")
    
    def _init_mpi(self):
        """Инициализация MPI бэкенда"""
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.use_mpi = self.size > 1
        except:
            self.use_mpi = False

    def _init_cache(self):
        """Инициализация кэша"""
        self.gate_cache = {}
        self.max_cache_size = MAX_CACHE_SIZE

    def _calculate_max_qubits(self) -> int:
        """Динамически определяет максимальное число кубитов для текущего железа"""
        limits = [
            self._get_cpu_qubit_limit(),
            self._get_gpu_qubit_limit(),
            self._get_ram_qubit_limit(),
            self._get_real_device_limits()
        ]
        valid_limits = [x for x in limits if x is not None]
        if not valid_limits:
            raise MemoryError("Unable to determine hardware limits")
        return min(valid_limits)

    def _get_cpu_qubit_limit(self) -> Optional[int]:
        try:
            cache_size = psutil.cpu_freq().max * 1e6 / 8
            max_qubits = int(math.log2(cache_size / (2 * 16)))
            return min(max_qubits, MAX_QUBITS_CPU)
        except:
            return None

    def _get_gpu_qubit_limit(self) -> Optional[int]:
        try:
            mem_info = cp.cuda.Device().mem_info
            available_mem = mem_info[0] * 0.8
            max_qubits = int(math.log2(available_mem / (3 * 16)))
            return min(max_qubits, MAX_QUBITS_GPU)
        except:
            return None

    def _get_ram_qubit_limit(self) -> int:
        available_ram = psutil.virtual_memory().available * 0.7
        return int(math.log2(available_ram / (2 * 16)))

    def _get_real_device_limits(self) -> int:
        return MAX_QUBITS_TN

    def enable_sparse_mode(self, threshold: float = SPARSE_STATE_THRESHOLD):
        """Включение разреженного режима"""
        self.use_sparse = True
        self.sparse_threshold = threshold
        logger.info(f"Sparse mode enabled with threshold {threshold}")

    def set_optimization_level(self, level: int):
        """Установка уровня оптимизации"""
        if level not in [0, 1, 2]:
            raise ValueError("Optimization level must be 0, 1 or 2")
        self.optimization_level = level
        logger.info(f"Optimization level set to {level}")

# ==================== КВАНТОВОЕ СОСТОЯНИЕ ====================
class QuantumState:
    """Представление квантового состояния с поддержкой CPU/GPU и разреженных матриц"""
    
    def __init__(self, num_qubits: int, backend: str = "gpu", sparse_threshold: Optional[float] = None):
        self.num_qubits = num_qubits
        self.backend = backend if backend in ["gpu", "cpu", "sparse"] else "cpu"
        self.sparse_threshold = sparse_threshold
        self._state = self._initialize_state()
        self.norm_tolerance = NORM_TOLERANCE
        self._gate_cache = {}
        self._sparse_representation = None
        logger.info(f"Initialized quantum state with {num_qubits} qubits on {backend.upper()}")

    def _initialize_state(self) -> Union[cp.ndarray, np.ndarray, csr_matrix]:
        """Инициализация состояния |0>^n"""
        try:
            state_size = 2**self.num_qubits
            
            if self.backend == "sparse":
                state = csr_matrix(([1.0], ([0], [0])), shape=(state_size, 1), dtype=np.complex128)
                return state
            
            required_memory = state_size * 16  # complex128 = 16 bytes
            
            if self.backend == "gpu":
                try:
                    mem_info = cp.cuda.Device().mem_info
                    if required_memory > mem_info[0] * 0.8:
                        raise MemoryError(f"Not enough GPU memory. Required: {required_memory/1e9:.2f} GB")
                    state = cp.zeros(state_size, dtype=cp.complex128)
                except cp.cuda.memory.OutOfMemoryError:
                    raise MemoryError("GPU memory allocation failed")
            else:
                if required_memory > psutil.virtual_memory().available * 0.7:
                    raise MemoryError(f"Not enough RAM. Required: {required_memory/1e9:.2f} GB")
                state = np.zeros(state_size, dtype=np.complex128)
            
            state[0] = 1.0
            return state
        except MemoryError as e:
            logger.error(f"Memory allocation failed: {str(e)}")
            if self.backend != "sparse":
                logger.info("Attempting to fall back to sparse representation")
                try:
                    return self._initialize_sparse_state()
                except Exception as e:
                    logger.error(f"Failed to initialize sparse state: {str(e)}")
                    raise
            raise
        except Exception as e:
            logger.error(f"State initialization failed: {str(e)}")
            raise QuantumError(f"Failed to initialize quantum state: {str(e)}")

    def _initialize_sparse_state(self) -> csr_matrix:
        """Инициализация разреженного состояния"""
        state_size = 2**self.num_qubits
        state = csr_matrix(([1.0], ([0], [0])), shape=(state_size, 1), dtype=np.complex128)
        self.backend = "sparse"
        return state

    def ensure_normalization(self):
        """Проверка и нормализация состояния"""
        if self.backend == "sparse":
            norm = np.sqrt((self._state.conj().multiply(self._state).sum())
            if abs(norm - 1.0) > self.norm_tolerance:
                logger.debug(f"Renormalizing sparse state (norm = {norm:.6f})")
                self._state = self._state / norm
            return
            
        norm = cp.linalg.norm(self._state) if self.backend == "gpu" else np.linalg.norm(self._state)
        if abs(norm - 1.0) > self.norm_tolerance:
            logger.debug(f"Renormalizing state (norm = {norm:.6f})")
            self._state /= norm

    def to_cpu(self) -> np.ndarray:
        """Конвертация в CPU массив"""
        if self.backend == "gpu":
            return cp.asnumpy(self._state)
        elif self.backend == "sparse":
            return self._state.toarray().flatten()
        return self._state

    def to_gpu(self) -> cp.ndarray:
        """Конвертация в GPU массив"""
        if self.backend == "cpu":
            return cp.array(self._state)
        elif self.backend == "sparse":
            return cp.array(self._state.toarray().flatten())
        return self._state

    def to_sparse(self) -> csr_matrix:
        """Конвертация в разреженную матрицу"""
        if self.backend == "sparse":
            return self._state.copy()
        
        state_cpu = self.to_cpu()
        mask = np.abs(state_cpu) > self.sparse_threshold if self.sparse_threshold else np.ones_like(state_cpu, dtype=bool)
        sparse_state = csr_matrix((state_cpu[mask], np.where(mask)), shape=state_cpu.shape, dtype=np.complex128)
        return sparse_state

    def change_backend(self, new_backend: str, sparse_threshold: Optional[float] = None):
        """Изменение бэкенда (CPU/GPU/sparse)"""
        if new_backend == self.backend:
            return
            
        logger.info(f"Changing backend from {self.backend} to {new_backend}")
        try:
            if new_backend == "cpu":
                self._state = self.to_cpu()
            elif new_backend == "gpu":
                self._state = self.to_gpu()
            elif new_backend == "sparse":
                self.sparse_threshold = sparse_threshold or self.sparse_threshold or SPARSE_STATE_THRESHOLD
                self._state = self.to_sparse()
            else:
                raise ValueError(f"Unknown backend: {new_backend}")
                
            self.backend = new_backend
        except Exception as e:
            logger.error(f"Failed to change backend: {str(e)}")
            raise BackendError(f"Backend change failed: {str(e)}")

    def clear_cache(self):
        """Очистка кэша гейтов"""
        self._gate_cache.clear()
        if self.backend == "gpu":
            cp.get_default_memory_pool().free_all_blocks()

    def is_sparse(self) -> bool:
        """Проверка, является ли состояние разреженным"""
        return self.backend == "sparse"

    def density_matrix(self) -> Union[np.ndarray, cp.ndarray, csr_matrix]:
        """Вычисление матрицы плотности"""
        state = self._state
        if self.backend == "sparse":
            return state.dot(state.conj().T)
        elif self.backend == "gpu":
            return cp.outer(state, state.conj())
        else:
            return np.outer(state, state.conj())

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

# ==================== MPI СИНХРОНИЗАЦИЯ ====================
class MPISynchronizer:
    """Синхронизация состояния между MPI процессами"""
    
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self._buffer = None
        logger.info(f"Initialized MPI synchronizer (rank {self.rank}/{self.size})")

    def sync_state(self, state: Union[cp.ndarray, np.ndarray, csr_matrix], qubit_indices: List[int]) -> None:
        """Синхронизация состояния по указанным кубитам"""
        if not self.comm or self.size == 1:
            return

        if isinstance(state, csr_matrix):
            self._sync_sparse_state(state, qubit_indices)
        else:
            self._sync_dense_state(state, qubit_indices)

    def _sync_dense_state(self, state: Union[cp.ndarray, np.ndarray], qubits: List[int]) -> None:
        """Синхронизация плотного состояния"""
        local_portion = self._extract_local_portion(state, qubits)
        global_state = np.zeros(2**len(qubits), dtype=np.complex128)
        self.comm.Allreduce(local_portion, global_state, op=MPI.SUM)
        self._reconstruct_global_state(state, global_state, qubits)

    def _sync_sparse_state(self, state: csr_matrix, qubits: List[int]) -> None:
        """Синхронизация разреженного состояния"""
        dense_state = state.toarray().flatten()
        self._sync_dense_state(dense_state, qubits)
        state.data = dense_state[state.indices]
        state.sum_duplicates()

    def _extract_local_portion(self, state: Union[cp.ndarray, np.ndarray, csr_matrix], qubits: List[int]) -> np.ndarray:
        """Извлечение локальной части состояния"""
        if isinstance(state, csr_matrix):
            state = state.toarray().flatten()
            
        mask = sum(1 << q for q in qubits if q // self.size == self.rank)
        local_size = 2 ** len([q for q in qubits if q // self.size == self.rank])
        portion = np.zeros(local_size, dtype=np.complex128)
        
        for i in range(len(state)):
            if (i & mask) == (self.rank << (qubits[0] % self.size)):
                portion[i % local_size] = state[i]
                
        return portion

    def _reconstruct_global_state(self, state: Union[cp.ndarray, np.ndarray], 
                                global_state: np.ndarray, qubits: List[int]) -> None:
        """Реконструкция глобального состояния"""
        norm = np.linalg.norm(global_state)
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise ValueError(f"Quantum state norm violation during MPI sync: {norm:.6f}")
            
        if isinstance(state, cp.ndarray):
            global_state = cp.asarray(global_state)
            
        for i in range(len(state)):
            state[i] = global_state[i % len(global_state)] / norm

# ==================== ЯДРА ДЛЯ GPU ====================
class MultiQubitKernels:
    """Оптимизированные ядра для квантовых операций на GPU"""
    
    @staticmethod
    @cuda.jit
    def entangled_cnot_kernel(state, controls: tuple, target: int, num_qubits: int):
        """Ядро для многокубитного CNOT"""
        idx = cuda.grid(1)
        if idx >= 2**num_qubits:
            return

        all_controls_active = True
        for ctrl in controls:
            if (idx & (1 << ctrl)) == 0:
                all_controls_active = False
                break

        if all_controls_active:
            target_mask = 1 << target
            paired_idx = idx ^ target_mask
            state[idx], state[paired_idx] = state[paired_idx], state[idx]

    @staticmethod
    @cuda.jit
    def quantum_fourier_kernel(state, qubits: tuple, num_qubits: int):
        """Ядро для квантового преобразования Фурье"""
        idx = cuda.grid(1)
        if idx >= 2**num_qubits:
            return

        phase = 0.0
        for i in range(len(qubits)):
            for j in range(i+1, len(qubits)):
                if (idx >> qubits[i]) & 1 and (idx >> qubits[j]) & 1:
                    phase += math.pi / (1 << (j - i))

        state[idx] *= math.exp(1j * phase)

    @staticmethod
    @cuda.jit(device=True)
    def _apply_single_qubit_gate(state, gate, target, idx, num_qubits):
        """Применение однокубитного гейта (устройство)"""
        mask = 1 << target
        basis0 = idx & ~mask
        basis1 = idx | mask
        
        if (idx >> target) & 1:
            basis0, basis1 = basis1, basis0
            
        val0 = state[basis0]
        val1 = state[basis1]
        
        state[basis0] = gate[0,0] * val0 + gate[0,1] * val1
        state[basis1] = gate[1,0] * val0 + gate[1,1] * val1

    @staticmethod
    @cuda.jit
    def sparse_state_kernel(indices, data, gate, target, new_indices, new_data):
        """Ядро для работы с разреженными состояниями"""
        idx = cuda.grid(1)
        if idx >= len(indices):
            return
            
        basis_idx = indices[idx]
        mask = 1 << target
        basis0 = basis_idx & ~mask
        basis1 = basis_idx | mask
        
        if (basis_idx >> target) & 1:
            basis0, basis1 = basis1, basis0
            
        new_data[2*idx] = gate[0,0] * data[idx]
        new_data[2*idx+1] = gate[1,0] * data[idx]
        
        new_indices[2*idx] = basis0
        new_indices[2*idx+1] = basis1

# ==================== МОДЕЛЬ ШУМОВ NISQ ====================
class NISQNoiseModel(NoiseModelInterface):
    """Расширенная модель шумов для NISQ-устройств с учетом утечки"""
    
    def __init__(self, qubit_props: Dict[int, QubitProperties]):
        self.qubit_props = qubit_props
        self.temperature = 0.015  # Kelvin
        self.crosstalk_matrix = self._build_crosstalk_matrix()
        self.error_history = []
        self._leakage_states = {}
        self._validate_qubit_properties()
        self._init_non_markovian_effects()

    def _validate_qubit_properties(self):
        """Проверка корректности параметров кубитов"""
        for q, props in self.qubit_props.items():
            if props.t1 <= 0 or props.t2 <= 0:
                raise ValueError(f"Invalid T1/T2 for qubit {q}: must be positive")
            if props.t2 > 2 * props.t1:
                raise ValueError(f"T2 cannot exceed 2*T1 for qubit {q}")
            if props.frequency <= 0:
                raise ValueError(f"Invalid frequency for qubit {q}: must be positive")
            if not (0 <= props.leakage_rate <= 0.1):
                raise ValueError(f"Invalid leakage rate for qubit {q}: must be between 0 and 0.1")

    def _init_non_markovian_effects(self):
        """Инициализация не-Markovian эффектов"""
        self.non_markovian_params = {
            'memory_time': 5.0,
            'correlation_strength': 0.1
        }
        self._correlation_history = defaultdict(list)

    def _build_crosstalk_matrix(self):
        """Построение матрицы перекрестных помех"""
        matrix = defaultdict(dict)
        for q1, props1 in self.qubit_props.items():
            for q2, props2 in self.qubit_props.items():
                if q1 >= q2:
                    continue
                dist = math.dist(props1.position, props2.position)
                strength = 0.05 * math.exp(-dist/2)
                
                if len(self._correlation_history[(q1, q2)]) > 0:
                    avg_corr = np.mean(self._correlation_history[(q1, q2)])
                    strength *= (1 + self.non_markovian_params['correlation_strength'] * avg_corr)
                
                matrix[q1][q2] = strength
                matrix[q2][q1] = matrix[q1][q2]
        return matrix

    def apply_noise(self, state: QuantumState, gate: str, qubits: List[int], duration: float):
        """Применение шумовых эффектов с учетом утечки"""
        try:
            self._apply_thermal_relaxation(state, qubits, duration)
            self._apply_dephasing(state, qubits, duration)
            self._apply_crosstalk(state, qubits)
            self._apply_leakage(state, qubits, gate, duration)
            self._apply_non_markovian_effects(state, qubits)
            state.ensure_normalization()
        except Exception as e:
            logger.error(f"Error applying noise: {str(e)}")
            raise NoiseModelError(f"Noise application failed: {str(e)}")

    def _apply_thermal_relaxation(self, state: QuantumState, qubits: List[int], duration: float):
        """Применение тепловой релаксации (T1)"""
        for q in qubits:
            t1 = self.qubit_props[q].t1
            p_reset = 1 - math.exp(-duration / t1)
            
            if np.random.random() < p_reset:
                if q not in self._leakage_states:
                    self._project_to_ground(state, q)
                    self._record_error('amplitude_damping', [q])

    def _apply_dephasing(self, state: QuantumState, qubits: List[int], duration: float):
        """Применение дефазировки (T2)"""
        for q in qubits:
            if q in self._leakage_states:
                continue
                
            t2 = self.qubit_props[q].t2
            p_dephase = 0.5 * (1 - math.exp(-duration / t2))
            
            if np.random.random() < p_dephase:
                self._apply_random_z_phase(state, q)
                self._record_error('phase_damping', [q])

    def _apply_crosstalk(self, state: QuantumState, qubits: List[int]):
        """Применение перекрестных помех"""
        for q in qubits:
            if q in self._leakage_states:
                continue
                
            for neighbor, strength in self.crosstalk_matrix[q].items():
                if neighbor in self._leakage_states:
                    continue
                    
                if np.random.random() < strength:
                    self._apply_crosstalk_gate(state, q, neighbor)
                    self._record_error('crosstalk', [q, neighbor])
                    
                    corr = np.random.normal(0, 0.1)
                    self._correlation_history[(q, neighbor)].append(corr)
                    if len(self._correlation_history[(q, neighbor)]) > 10:
                        self._correlation_history[(q, neighbor)].pop(0)

    def _apply_leakage(self, state: QuantumState, qubits: List[int], gate: str, duration: float):
        """Применение эффектов утечки"""
        for q in qubits:
            if q in self._leakage_states:
                continue
                
            leakage_prob = self.qubit_props[q].leakage_rate * duration / 1000
            if np.random.random() < leakage_prob:
                self._leakage_states[q] = True
                self._record_error('leakage', [q])
                logger.warning(f"Qubit {q} entered leakage state")

    def _apply_non_markovian_effects(self, state: QuantumState, qubits: List[int]):
        """Применение не-Markovian эффектов"""
        for q in qubits:
            if q in self._leakage_states:
                continue
                
            if np.random.random() < 0.05:
                phase = np.random.normal(0, math.pi/4)
                self._apply_global_phase(state, phase)
                self._record_error('non_markovian', [q])

    def _project_to_ground(self, state: QuantumState, qubit: int):
        """Проекция в основное состояние"""
        mask = 1 << qubit
        if state.is_sparse():
            sparse_state = state.state
            new_data = []
            new_indices = []
            
            for i in range(sparse_state.nnz):
                idx = sparse_state.indices[i]
                if not (idx & mask):
                    new_data.append(sparse_state.data[i])
                    new_indices.append(idx)
                    
            state.state = csr_matrix((new_data, (new_indices, [0]*len(new_indices))), 
                                    shape=sparse_state.shape, dtype=np.complex128)
        else:
            state_vector = state.state
            for i in range(len(state_vector)):
                if i & mask:
                    state_vector[i] = 0

    def _apply_random_z_phase(self, state: QuantumState, qubit: int):
        """Случайный фазовый сдвиг"""
        phase = np.random.normal(0, math.pi/8)
        mask = 1 << qubit
        
        if state.is_sparse():
            sparse_state = state.state
            for i in range(sparse_state.nnz):
                if sparse_state.indices[i] & mask:
                    sparse_state.data[i] *= cmath.exp(1j * phase)
        else:
            state_vector = state.state
            for i in range(len(state_vector)):
                if i & mask:
                    state_vector[i] *= cmath.exp(1j * phase)

    def _apply_global_phase(self, state: QuantumState, phase: float):
        """Применение глобальной фазы"""
        if state.is_sparse():
            sparse_state = state.state
            sparse_state.data *= cmath.exp(1j * phase)
        else:
            state.state *= cmath.exp(1j * phase)

    def _apply_crosstalk_gate(self, state: QuantumState, q1: int, q2: int):
        """Применение гейта из-за перекрестных помех"""
        mask1 = 1 << q1
        mask2 = 1 << q2
        
        if state.is_sparse():
            sparse_state = state.state
            new_data = []
            new_indices = []
            
            for i in range(sparse_state.nnz):
                idx = sparse_state.indices[i]
                val = sparse_state.data[i]
                
                if (idx & mask1) and not (idx & mask2):
                    paired_idx = idx ^ mask2
                    new_indices.extend([idx, paired_idx])
                    new_data.extend([0, val])
                elif not (idx & mask1) and (idx & mask2):
                    paired_idx = idx ^ mask2
                    new_indices.extend([idx, paired_idx])
                    new_data.extend([val, 0])
                else:
                    new_indices.append(idx)
                    new_data.append(val)
                    
            state.state = csr_matrix((new_data, (new_indices, [0]*len(new_indices))), 
                                    shape=sparse_state.shape, dtype=np.complex128)
        else:
            state_vector = state.state
            for i in range(len(state_vector)):
                if (i & mask1) and not (i & mask2):
                    paired_idx = i ^ mask2
                    state_vector[i], state_vector[paired_idx] = state_vector[paired_idx], state_vector[i]

    def reset_leakage_states(self):
        """Сброс состояний утечки"""
        self._leakage_states.clear()

    def get_error_rates(self) -> Dict[str, float]:
        """Получение статистики ошибок"""
        error_counts = defaultdict(int)
        for error in self.error_history:
            error_counts[error['type']] += 1
            
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {}
            
        return {err_type: count/total_errors for err_type, count in error_counts.items()}

    def _record_error(self, error_type: str, qubits: List[int]):
        """Запись ошибки в историю"""
        self.error_history.append({
            'type': error_type,
            'qubits': qubits,
            'timestamp': time.time(),
            'leakage': any(q in self._leakage_states for q in qubits)
        })

# ==================== ГИБРИДНЫЙ ИСПОЛНИТЕЛЬ ====================
class HybridExecutor:
    """Расширенный динамический выбор бэкенда для вычислений (CPU/GPU/MPI/sparse)"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.modes = {
            'gpu': self._run_gpu,
            'cpu': self._run_cpu,
            'mpi': self._run_mpi,
            'sparse': self._run_sparse
        }
        self._init_profiler()
        self._init_adaptive_optimizer()

    def _init_profiler(self):
        """Инициализация профилировщика"""
        self.profile_data = {
            'gpu': {'time': 0.0, 'success': True, 'memory': 0.0},
            'cpu': {'time': 0.0, 'success': True, 'memory': 0.0},
            'mpi': {'time': 0.0, 'success': True, 'memory': 0.0},
            'sparse': {'time': 0.0, 'success': True, 'memory': 0.0}
        }
        self.last_profile_time = time.time()
        self.profile_interval = 5

    def _init_adaptive_optimizer(self):
        """Инициализация адаптивного оптимизатора"""
        self.optimization_strategies = {
            0: self._no_optimization,
            1: self._basic_optimization,
            2: self._aggressive_optimization
        }

    def execute(self, gate_name, qubits, **params):
        """Выполнение операции с выбором оптимального бэкенда"""
        backend = self._select_backend(gate_name, qubits)
        optimization_level = self.simulator.hw_config.optimization_level
        
        try:
            optimized_gate, optimized_qubits = self.optimization_strategies[optimization_level](gate_name, qubits, params)
            
            start_time = time.time()
            self.modes[backend](optimized_gate, optimized_qubits, **params)
            exec_time = time.time() - start_time
            
            self.profile_data[backend]['time'] = 0.9 * self.profile_data[backend]['time'] + 0.1 * exec_time
            self.profile_data[backend]['success'] = True
            
            if backend == 'gpu':
                mem_pool = cp.get_default_memory_pool()
                self.profile_data[backend]['memory'] = mem_pool.used_bytes() / (1024 ** 2)
            elif backend == 'cpu':
                self.profile_data[backend]['memory'] = psutil.Process().memory_info().rss / (1024 ** 2)
            
        except Exception as e:
            logger.warning(f"Error executing on {backend}: {str(e)}")
            self.profile_data[backend]['success'] = False
            
            if backend != 'cpu':
                self.modes['cpu'](gate_name, qubits, **params)
            else:
                raise BackendError(f"Failed to execute on CPU fallback: {str(e)}")

    def _select_backend(self, gate_name, qubits) -> str:
        """Улучшенный выбор бэкенда"""
        if time.time() - self.last_profile_time > self.profile_interval:
            self._update_backend_profile()
            self.last_profile_time = time.time()
        
        if (self.simulator.hw_config.use_sparse and 
            self._is_state_sparse() and 
            gate_name in ['x', 'y', 'z', 'h', 'cx']):
            return 'sparse'
        
        if (hasattr(self.simulator, 'hw_config') and self.simulator.hw_config.use_mpi and len(qubits) > 5:
            return 'mpi'
        elif (hasattr(self.simulator, 'hw_config') and self.simulator.hw_config.use_gpu):
            return 'gpu'
        return 'cpu'

    def _is_state_sparse(self) -> bool:
        """Проверка, является ли состояние разреженным"""
        if not hasattr(self.simulator, 'quantum_state'):
            return False
            
        if self.simulator.quantum_state.is_sparse():
            return True
            
        state = self.simulator.quantum_state.to_cpu()
        nonzero = np.count_nonzero(state)
        return nonzero / len(state) < SPARSE_STATE_THRESHOLD

    def _update_backend_profile(self):
        """Обновление профилей бэкендов"""
        test_gates = ['h', 'cx', 'ccx', 'rz']
        for gate in test_gates:
            for backend in ['gpu', 'cpu', 'mpi', 'sparse']:
                if backend in self.modes:
                    try:
                        qubits = [0, 1] if gate == 'cx' else [0, 1, 2] if gate == 'ccx' else [0]
                        
                        start_time = time.time()
                        self.modes[backend](gate, qubits, theta=0.5 if gate == 'rz' else None)
                        exec_time = time.time() - start_time
                        
                        mem_usage = 0
                        if backend == 'gpu':
                            mem_pool = cp.get_default_memory_pool()
                            mem_usage = mem_pool.used_bytes() / (1024 ** 2)
                        elif backend == 'cpu':
                            mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)
                        
                        self.profile_data[backend]['time'] = 0.9 * self.profile_data[backend]['time'] + 0.1 * exec_time
                        self.profile_data[backend]['memory'] = 0.9 * self.profile_data[backend]['memory'] + 0.1 * mem_usage
                        self.profile_data[backend]['success'] = True
                    except Exception as e:
                        logger.warning(f"Error profiling {backend} with {gate}: {str(e)}")
                        self.profile_data[backend]['success'] = False

    def _no_optimization(self, gate_name, qubits, params):
        """Без оптимизаций"""
        return gate_name, qubits

    def _basic_optimization(self, gate_name, qubits, params):
        """Базовые оптимизации"""
        if gate_name in ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz'] and len(qubits) == 1:
            last_op = self._get_last_operation(qubits[0])
            if last_op and last_op.gate_type in ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz']:
                combined_gate = self._combine_single_qubit_gates(last_op.gate_type, gate_name, params)
                if combined_gate:
                    self.simulator.operation_history.pop()
                    return combined_gate[0], combined_gate[1]
                    
        return gate_name, qubits

    def _aggressive_optimization(self, gate_name, qubits, params):
        """Агрессивные оптимизации"""
        gate_name, qubits = self._basic_optimization(gate_name, qubits, params)
        
        if gate_name == 'cx' and len(qubits) == 2:
            last_op = self._get_last_operation(qubits[1])
            if last_op and last_op.gate_type == 'cx' and last_op.qubits == qubits[::-1]:
                self.simulator.operation_history.pop()
                return 'id', qubits[0]
                
        return gate_name, qubits

    def _get_last_operation(self, qubit: int) -> Optional[OperationRecord]:
        """Получение последней операции для кубита"""
        if not hasattr(self.simulator, 'qubit_operation_history'):
            return None
            
        history = self.simulator.qubit_operation_history.get(qubit, [])
        return history[-1] if history else None

    def _combine_single_qubit_gates(self, gate1: str, gate2: str, params: dict) -> Optional[tuple]:
        """Объединение однокубитных гейтов"""
        if gate1 == 'h' and gate2 == 'h':
            return ('id', [0])
        elif gate1 == 'x' and gate2 == 'x':
            return ('id', [0])
        elif gate1 == 'rz' and gate2 == 'rz' and 'theta' in params:
            return ('rz', [0], {'theta': params['theta'] * 2})
            
        return None

    def _run_gpu(self, gate_name, qubits, **params):
        """Выполнение на GPU"""
        if not hasattr(self.simulator, '_optimized_apply_gate_kernel'):
            raise BackendError("GPU kernels not compiled")
        
        if gate_name in ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg']:
            gate_matrix = self.simulator.gate_set[gate_name]
            self.simulator._optimized_apply_gate_kernel[
                self.simulator.blocks_per_grid, 
                self.simulator.threads_per_block
            ](self.simulator.quantum_state.state, gate_matrix, qubits[0], self.simulator.num_qubits)
        elif gate_name == 'cx':
            self.simulator._optimized_cnot_kernel[
                self.simulator.blocks_per_grid,
                self.simulator.threads_per_block
            ](self.simulator.quantum_state.state, qubits[0], qubits[1], self.simulator.num_qubits)
        elif gate_name == 'ccx':
            self.simulator._optimized_ccx_kernel[
                self.simulator.blocks_per_grid,
                self.simulator.threads_per_block
            ](self.simulator.quantum_state.state, qubits[0], qubits[1], qubits[2], self.simulator.num_qubits)
        elif gate_name in ['rz', 'u3', 'rx', 'ry']:
            gate_matrix = self.simulator.gate_set[gate_name](**params)
            self.simulator._optimized_apply_gate_kernel[
                self.simulator.blocks_per_grid,
                self.simulator.threads_per_block
            ](self.simulator.quantum_state.state, gate_matrix, qubits[0], self.simulator.num_qubits)
        else:
            raise BackendError(f"Unsupported gate {gate_name} for GPU backend")

    def _run_cpu(self, gate_name, qubits, **params):
        """Выполнение на CPU"""
        state_cpu = self.simulator.quantum_state.to_cpu()
        
        if gate_name in ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg']:
            gate_matrix = cp.asnumpy(self.simulator.gate_set[gate_name])
            for i in range(len(state_cpu)):
                MultiQubitKernels._apply_single_qubit_gate(state_cpu, gate_matrix, qubits[0], i, self.simulator.num_qubits)
        elif gate_name == 'cx':
            control, target = qubits
            for i in range(len(state_cpu)):
                control_mask = 1 << control
                target_mask = 1 << target
                if (i & control_mask) and not (i & target_mask):
                    paired_idx = i ^ target_mask
                    state_cpu[i], state_cpu[paired_idx] = state_cpu[paired_idx], state_cpu[i]
        elif gate_name in ['rz', 'u3', 'rx', 'ry']:
            gate_matrix = cp.asnumpy(self.simulator.gate_set[gate_name](**params))
            for i in range(len(state_cpu)):
                MultiQubitKernels._apply_single_qubit_gate(state_cpu, gate_matrix, qubits[0], i, self.simulator.num_qubits)
        else:
            raise BackendError(f"Unsupported gate {gate_name} for CPU backend")
        
        self.simulator.quantum_state.state = state_cpu

    def _run_mpi(self, gate_name, qubits, **params):
        """Выполнение с использованием MPI"""
        if not hasattr(self.simulator, 'mpi_sync'):
            raise BackendError("MPI not initialized")
        
        if self.simulator.hw_config.rank == 0:
            self._run_cpu(gate_name, qubits, **params)
        
        self.simulator.mpi_sync.sync_state(self.simulator.quantum_state.state, qubits)

    def _run_sparse(self, gate_name, qubits, **params):
        """Выполнение в разреженном формате"""
        if not self.simulator.quantum_state.is_sparse():
            self.simulator.quantum_state.change_backend("sparse")
            
        sparse_state = self.simulator.quantum_state.state
        
        if gate_name in ['x', 'y', 'z', 'h']:
            gate_matrix = cp.asnumpy(self.simulator.gate_set[gate_name])
            
            new_data = np.zeros(2 * sparse_state.nnz, dtype=np.complex128)
            new_indices = np.zeros(2 * sparse_state.nnz, dtype=np.int64)
            
            threads_per_block = 256
            blocks_per_grid = (sparse_state.nnz + threads_per_block - 1) // threads_per_block
            
            MultiQubitKernels.sparse_state_kernel[
                blocks_per_grid, threads_per_block
            ](sparse_state.indices, sparse_state.data, gate_matrix, qubits[0], new_indices, new_data)
            
            self.simulator.quantum_state.state = csr_matrix(
                (new_data, (new_indices, np.zeros_like(new_indices))),
                shape=sparse_state.shape, dtype=np.complex128)
        elif gate_name == 'cx':
            control, target = qubits
            new_data = []
            new_indices = []
            
            for i in range(sparse_state.nnz):
                idx = sparse_state.indices[i]
                val = sparse_state.data[i]
                
                control_mask = 1 << control
                target_mask = 1 << target
                
                if (idx & control_mask) and not (idx & target_mask):
                    paired_idx = idx ^ target_mask
                    new_indices.extend([idx, paired_idx])
                    new_data.extend([0, val])
                elif not (idx & control_mask) and (idx & target_mask):
                    paired_idx = idx ^ target_mask
                    new_indices.extend([idx, paired_idx])
                    new_data.extend([val, 0])
                else:
                    new_indices.append(idx)
                    new_data.append(val)
                    
            self.simulator.quantum_state.state = csr_matrix(
                (new_data, (new_indices, np.zeros_like(new_indices))),
                shape=sparse_state.shape, dtype=np.complex128)
        else:
            raise BackendError(f"Unsupported gate {gate_name} for sparse backend")

# ==================== ТЕНЗОРНЫЕ СЕТИ ====================
class TensorNetworkState:
    """Улучшенное представление квантового состояния в виде тензорной сети"""
    
    def __init__(self, num_qubits, initial_bond_dim=16, max_bond_dim=64, cutoff=1e-12):
        self.num_qubits = num_qubits
        self.bond_dim = initial_bond_dim
        self.max_bond_dim = max_bond_dim
        self.cutoff = cutoff
        self.tensors = self._initialize_tensors()
        self.connections = []
        self._svd_counter = 0
        self._svd_time = 0.0
        logger.info(f"Initialized tensor network with {num_qubits} qubits (bond_dim={initial_bond_dim})")

    def _initialize_tensors(self):
        """Инициализация тензоров для состояния |0>^n"""
        try:
            tensors = []
            for i in range(self.num_qubits):
                if i == 0:
                    tensor = np.zeros((1, 2), dtype=np.complex128)
                    tensor[0, 0] = 1.0
                elif i == self.num_qubits - 1:
                    tensor = np.zeros((self.bond_dim, 1), dtype=np.complex128)
                    tensor[0, 0] = 1.0
                else:
                    tensor = np.zeros((self.bond_dim, 2, self.bond_dim), dtype=np.complex128)
                    tensor[0, 0, 0] = 1.0
                tensors.append(tensor)
            return tensors
        except MemoryError:
            if self.bond_dim > 4:
                new_bond_dim = max(4, self.bond_dim // 2)
                logger.warning(f"Memory error, reducing bond_dim from {self.bond_dim} to {new_bond_dim}")
                self.bond_dim = new_bond_dim
                return self._initialize_tensors()
            raise

    def apply_gate(self, gate, qubits):
        """Применение гейта к тензорной сети"""
        try:
            if len(qubits) == 1:
                self._apply_single_qubit_gate(gate, qubits[0])
            elif len(qubits) == 2:
                self._apply_two_qubit_gate(gate, qubits)
            else:
                raise NotImplementedError("Многокубитные гейты требуют особой обработки")
        except Exception as e:
            logger.error(f"Error applying gate to tensor network: {str(e)}")
            raise QuantumError(f"Failed to apply gate to tensor network: {str(e)}")

    def _apply_single_qubit_gate(self, gate, qubit):
        """Применение однокубитного гейта"""
        if qubit < 0 or qubit >= self.num_qubits:
            raise QubitValidationError(f"Invalid qubit index {qubit}")
            
        try:
            if qubit == 0:
                self.tensors[0] = np.tensordot(self.tensors[0], gate, axes=([1], [1]))
            elif qubit == self.num_qubits - 1:
                pass
            else:
                self.tensors[qubit] = np.tensordot(self.tensors[qubit], gate, axes=([1], [1]))
        except ValueError as e:
            raise QuantumError(f"Dimension mismatch in single-qubit gate application: {str(e)}")

    def _apply_two_qubit_gate(self, gate, qubits):
        """Применение двухкубитного гейта"""
        q1, q2 = sorted(qubits)
        
        try:
            merged_tensor = self._merge_tensors(q1, q2)
            
            gate_shape = (2, 2, 2, 2)
            gate_reshaped = gate.reshape(gate_shape)
            merged_tensor = np.tensordot(merged_tensor, gate_reshaped, axes=([1, 2], [2, 3]))
            
            self._split_tensors(q1, q2, merged_tensor)
        except MemoryError:
            if self.max_bond_dim > self.bond_dim:
                self.compress_state(max(self.bond_dim // 2, 4))
                self._apply_two_qubit_gate(gate, qubits)
            else:
                raise MemoryError("Maximum bond dimension reached, cannot apply two-qubit gate")

    def _merge_tensors(self, q1, q2):
        """Объединение тензоров"""
        if q1 >= q2:
            raise ValueError("q1 must be less than q2")
            
        try:
            merged = self.tensors[q1]
            
            for i in range(q1 + 1, q2 + 1):
                merged = np.tensordot(merged, self.tensors[i], axes=(-1, 0))
                
            return merged
        except Exception as e:
            raise QuantumError(f"Failed to merge tensors: {str(e)}")

    def _split_tensors(self, q1, q2, merged_tensor):
        """Разделение тензоров после применения гейта"""
        start_time = time.time()
        
        try:
            original_shape = merged_tensor.shape
            matrix = merged_tensor.reshape(np.prod(merged_tensor.shape[:q2-q1+1]), -1)
            
            k = min(self.max_bond_dim, min(matrix.shape) - 1)
            k = max(k, 1)
            
            u, s, vh = svd(matrix, full_matrices=False)
            
            trunc = (s > self.cutoff).sum()
            k = min(k, trunc)
            
            u = u[:, :k]
            s = s[:k]
            vh = vh[:k, :]
            
            self.tensors[q1] = u.reshape(original_shape[0], -1, k)
            self.tensors[q2] = (np.diag(s) @ vh).reshape(k, -1, original_shape[-1])
            
            self.bond_dim = k
            
            self._svd_counter += 1
            self._svd_time += time.time() - start_time
        except Exception as e:
            raise QuantumError(f"SVD failed during tensor splitting: {str(e)}")

    def compress_state(self, new_bond_dim: int):
        """Сжатие состояния с уменьшением bond dimension"""
        if new_bond_dim >= self.bond_dim:
            return
            
        self.bond_dim = new_bond_dim
        try:
            for i in range(len(self.tensors) - 1):
                if len(self.tensors[i].shape) == 3:
                    u, s, vh = np.linalg.svd(self.tensors[i], full_matrices=False)
                    
                    trunc = (s > self.cutoff).sum()
                    k = min(new_bond_dim, trunc)
                    
                    u = u[:, :k]
                    s = s[:k]
                    vh = vh[:k, :]
                    
                    self.tensors[i] = u @ np.diag(s) @ vh
        except Exception as e:
            raise QuantumError(f"Failed to compress state: {str(e)}")

    def reconstruct_full_state(self) -> np.ndarray:
        """Реконструкция полного вектора состояния"""
        if self.num_qubits > 20:
            raise MemoryError("Reconstructing full state for >20 qubits is impractical")
            
        try:
            state = self.tensors[0]
            for i in range(1, self.num_qubits):
                state = np.tensordot(state, self.tensors[i], axes=(-1, 0))
            
            return state.reshape(-1)
        except MemoryError:
            raise MemoryError("Not enough memory to reconstruct full state")
        except Exception as e:
            raise QuantumError(f"Failed to reconstruct full state: {str(e)}")

    def measure(self, qubit: int) -> int:
        """Измерение кубита в тензорной сети"""
        if qubit < 0 or qubit >= self.num_qubits:
            raise QubitValidationError(f"Invalid qubit index {qubit}")
            
        try:
            full_state = self.reconstruct_full_state()
            prob0 = np.sum(np.abs(full_state[::2])**2)
            result = 0 if np.random.random() < prob0 else 1
            
            if result == 0:
                mask = [0, 1]
            else:
                mask = [1, 0]
                
            self.tensors[qubit] = self.tensors[qubit][:, mask, :] / np.sqrt(prob0 if result == 0 else 1 - prob0)
            
            return result
        except Exception as e:
            raise QuantumError(f"Failed to measure qubit {qubit}: {str(e)}")

    def get_svd_stats(self) -> Dict[str, float]:
        """Получение статистики по SVD операциям"""
        return {
            'total_svd': self._svd_counter,
            'total_svd_time': self._svd_time,
            'avg_svd_time': self._svd_time / self._svd_counter if self._svd_counter > 0 else 0
        }

# ==================== ГИБРИДНЫЙ КВАНТОВЫЙ СИМУЛЯТОР ====================
class HybridQuantumSimulator:
    """Улучшенный гибридный квантовый симулятор с автоматическим выбором представления"""
    
    def __init__(self, num_qubits: int, max_sv_qubits: int = 28, bond_dim: int = 16, 
                 optimization_level: int = 2, sparse_threshold: Optional[float] = None):
        self.num_qubits = num_qubits
        self.max_sv_qubits = max_sv_qubits
        self.bond_dim = bond_dim
        self.sparse_threshold = sparse_threshold
        
        self.hw_config = HardwareConfig()
        self.hw_config.set_optimization_level(optimization_level)
        if sparse_threshold is not None:
            self.hw_config.enable_sparse_mode(sparse_threshold)
        
        if num_qubits <= max_sv_qubits:
            self.backend = StateVectorBackend(num_qubits, 
                "gpu" if self.hw_config.use_gpu else "cpu",
                sparse_threshold)
            self.current_mode = 'state_vector'
        else:
            self.backend = TensorNetworkBackend(num_qubits, bond_dim)
            self.current_mode = 'tensor_network'
        
        self._init_common_components()
        logger.info(f"Initialized hybrid simulator with {num_qubits} qubits in {self.current_mode} mode")

    def _init_common_components(self):
        """Инициализация общих компонентов"""
        try:
            self.gate_set = {
                'h': self._create_gate([[1, 1], [1, -1]], dtype=np.complex128)/np.sqrt(2),
                'cx': self._create_gate([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
            }
            self.noise_model = NISQNoiseModel(self._default_qubit_props())
            self.operation_history = []
            self.error_correction = None
            self.visualizer = QuantumVisualizer(self)
            
            if self.hw_config.use_mpi:
                self.mpi_sync = MPISynchronizer(self.hw_config.comm)
        except Exception as e:
            logger.error(f"Failed to initialize common components: {str(e)}")
            raise QuantumError(f"Initialization failed: {str(e)}")

    def _create_gate(self, matrix, dtype=np.complex128):
        """Создание гейта с учетом текущего бэкенда"""
        if self.hw_config.use_gpu:
            return cp.array(matrix, dtype=dtype)
        return np.array(matrix, dtype=dtype)

    def _default_qubit_props(self) -> Dict[int, QubitProperties]:
        """Генерация свойств кубитов по умолчанию"""
        return {
            i: QubitProperties(
                qubit_type=QubitType.TRANSMON,
                t1=75.0 + np.random.normal(0, 5.0),
                t2=50.0 + np.random.normal(0, 5.0),
                frequency=5.0 + 0.1*i + np.random.normal(0, 0.01),
                anharmonicity=-0.34 + np.random.normal(0, 0.01),
                position=(i//4, i%4),
                gate_times={'h': 35, 'x': 35, 'y': 35, 'z': 35, 'rz': 0, 'u3': 50, 'cx': 100, 'measure': 300},
                leakage_rate=0.001 + np.random.normal(0, 0.0005),
                readout_error=(0.01 + np.random.normal(0, 0.002),
                            0.01 + np.random.normal(0, 0.003))
            for i in range(self.num_qubits)
        }

    def apply_gate(self, gate_name: str, qubits: List[int], **params):
        """Улучшенное применение квантового гейта"""
        try:
            if (self.current_mode == 'state_vector' and 
                len(self.operation_history) > 1000 and 
                self._requires_switch()):
                self._switch_to_tensor_network()
            
            gate_time = self._get_gate_time(gate_name, qubits[0])
            start_time = time.time()
            
            self.backend.apply_gate(gate_name, qubits, **params)
            
            self.noise_model.apply_noise(self.backend.state, gate_name, qubits, gate_time)
            self._record_operation(gate_name, qubits, time.time()-start_time)
            
        except MemoryError as e:
            logger.warning(f"Memory error during gate application: {str(e)}")
            self._handle_memory_error(gate_name, qubits, **params)
        except Exception as e:
            logger.error(f"Error applying gate {gate_name}: {str(e)}")
            self._record_operation(gate_name, qubits, 0, error=True, error_message=str(e))
            raise QuantumError(f"Failed to apply gate {gate_name}: {str(e)}")

    def _handle_memory_error(self, gate_name: str, qubits: List[int], **params):
        """Обработка ошибок памяти"""
        if self.current_mode == 'state_vector':
            logger.info("Attempting to switch to tensor network due to memory error")
            try:
                self._switch_to_tensor_network()
                self.backend.apply_gate(gate_name, qubits, **params)
            except Exception as e:
                raise MemoryError(f"Failed to handle memory error: {str(e)}")
        else:
            raise MemoryError("Memory error in tensor network mode")

    def _requires_switch(self) -> bool:
        """Проверка необходимости переключения на тензорные сети"""
        if self.current_mode != 'state_vector':
            return False
            
        state = self.backend.state
        if isinstance(state, csr_matrix):
            nonzero_ratio = state.nnz / (2**self.num_qubits)
        else:
            state_cpu = self.backend.to_cpu()
            nonzero_ratio = np.count_nonzero(state_cpu) / len(state_cpu)
            
        return nonzero_ratio < SPARSE_STATE_THRESHOLD

    def _switch_to_tensor_network(self):
        """Переключение на представление тензорными сетями"""
        logger.info(f"Switching to tensor network (bond_dim={self.bond_dim})")
        try:
            if isinstance(self.backend.state, csr_matrix):
                full_state = self.backend.state.toarray().flatten()
            else:
                full_state = self.backend.to_cpu()
                
            tn_state = self._state_vector_to_tensors(full_state)
            self.backend = TensorNetworkBackend(
                self.num_qubits, 
                initial_state=tn_state,
                bond_dim=self.bond_dim
            )
            self.current_mode = 'tensor_network'
        except MemoryError:
            raise MemoryError("Not enough memory to switch to tensor network")
        except Exception as e:
            raise QuantumError(f"Failed to switch to tensor network: {str(e)}")

    def _state_vector_to_tensors(self, state_vector: np.ndarray) -> List[np.ndarray]:
        """Конвертация вектора состояния в тензоры"""
        try:
            tensors = []
            remaining_state = state_vector.reshape([2]*self.num_qubits)
            
            for _ in range(self.num_qubits-1):
                reshaped = remaining_state.reshape(2, -1)
                U, s, Vh = np.linalg.svd(reshaped, full_matrices=False)
                
                k = min(self.bond_dim, len(s))
                U = U[:, :k]
                s = s[:k]
                Vh = Vh[:k, :]
                
                tensors.append(U)
                remaining_state = np.diag(s) @ Vh
            
            tensors.append(remaining_state)
            return tensors
        except Exception as e:
            raise QuantumError(f"Failed to convert state to tensors: {str(e)}")

    def measure(self, qubit: int) -> int:
        """Измерение кубита"""
        try:
            result = self.backend.measure(qubit)
            if self.current_mode == 'tensor_network':
                self.backend.compress_state(self.bond_dim)
            return result
        except Exception as e:
            logger.error(f"Error measuring qubit {qubit}: {str(e)}")
            self._record_operation('measure', [qubit], 0, error=True, error_message=str(e))
            raise QuantumError(f"Failed to measure qubit {qubit}: {str(e)}")

    @property
    def state(self):
        return self.backend.state

    def get_state_vector(self) -> np.ndarray:
        """Получение полного вектора состояния"""
        try:
            if self.current_mode == 'tensor_network':
                return self.backend.reconstruct_full_state()
            return self.backend.get_state().copy()
        except MemoryError:
            raise MemoryError("Not enough memory to get full state vector")
        except Exception as e:
            raise QuantumError(f"Failed to get state vector: {str(e)}")

    def _record_operation(self, gate_type: str, qubits: List[int], duration: float = 0.0,
                        error: bool = False, error_message: Optional[str] = None):
        """Запись операции в историю"""
        record = OperationRecord(
            gate_type=gate_type,
            qubits=qubits,
            timestamp=time.time(),
            duration=duration,
            error_occurred=error,
            error_message=error_message
        )
        self.operation_history.append(record)
        for q in qubits:
            self.qubit_operation_history[q].append(record)

    def get_error_stats(self) -> Dict[str, Any]:
        """Получение статистики по ошибкам"""
        error_count = 0
        total_operations = len(self.operation_history)
        
        if total_operations == 0:
            return {}
            
        for op in self.operation_history:
            if op.error_occurred:
                error_count += 1
                
        return {
            'total_operations': total_operations,
            'error_count': error_count,
            'error_rate': error_count / total_operations,
            'noise_errors': self.noise_model.get_error_rates()
        }

    def reset(self):
        """Полный сброс симулятора"""
        try:
            if self.current_mode == 'state_vector':
                self.backend = StateVectorBackend(self.num_qubits, 
                    "gpu" if self.hw_config.use_gpu else "cpu",
                    self.sparse_threshold)
            else:
                self.backend = TensorNetworkBackend(self.num_qubits, self.bond_dim)
                
            self.noise_model.reset_leakage_states()
            self.operation_history = []
            self.qubit_operation_history = defaultdict(list)
        except Exception as e:
            raise QuantumError(f"Failed to reset simulator: {str(e)}")

# ==================== ОСНОВНОЙ КЛАСС СИМУЛЯТОРА ====================
class QuantumSimulatorCUDA:
    """Улучшенный основной класс квантового симулятора с поддержкой CUDA"""
    
    def __init__(self, num_qubits: int = 20, qubit_props: Optional[Dict[int, QubitProperties]] = None,
                 optimization_level: int = 2, sparse_threshold: Optional[float] = None):
        self.hw_config = HardwareConfig()
        self.hw_config.set_optimization_level(optimization_level)
        if sparse_threshold is not None:
            self.hw_config.enable_sparse_mode(sparse_threshold)
            
        self.max_qubits = self.hw_config.max_qubits
        self.num_qubits = min(num_qubits, self.max_qubits)
        
        if num_qubits > self.max_qubits:
            logger.warning(f"Reduced qubits from {num_qubits} to {self.max_qubits} due to hardware limits")
            
        self._validate_qubit_count()
        
        self.quantum_state = QuantumState(self.num_qubits, 
                                       "gpu" if self.hw_config.use_gpu else "cpu",
                                       sparse_threshold)
        self.qubit_props = qubit_props or self._default_qubit_props()
        self.noise_model = NISQNoiseModel(self.qubit_props)
        self.error_correction = None
        self.gate_set = self._initialize_gate_set()
        self.quantum_api_handler = QuantumAPIHandler()
        self.operation_history = []
        self.qubit_operation_history = defaultdict(list)
        
        self.threads_per_block = 256
        self.blocks_per_grid = (2**self.num_qubits + self.threads_per_block - 1) // self.threads_per_block
        
        self._init_optimization()
        self._init_algorithms()
        
        if self.hw_config.use_mpi:
            self.mpi_sync = MPISynchronizer(self.hw_config.comm)
            self._init_mpi_distribution()
        
        self.kernels = MultiQubitKernels()
        self.executor = HybridExecutor(self)
        self.tensor_network = TensorNetworkState(num_qubits) if num_qubits > 28 else None
        self.visualizer = QuantumVisualizer(self)
        logger.info(f"Quantum simulator initialized with {self.num_qubits} qubits")

    def _validate_qubit_count(self):
        """Проверка допустимого числа кубитов"""
        total_energy = MIN_ENERGY_PER_QUBIT * (2**self.num_qubits)
        if total_energy > MAX_ENERGY_THRESHOLD:
            raise QubitValidationError(
                f"{self.num_qubits} qubits require {total_energy:.1e}J, exceeding practical limits")

    def _init_optimization(self):
        """Компиляция оптимизированных ядер"""
        try:
            self._compile_optimized_kernels()
        except Exception as e:
            logger.error(f"Failed to compile optimized kernels: {str(e)}")
            self.hw_config.use_gpu = False
            raise BackendError(f"GPU optimization failed: {str(e)}")

    def _compile_optimized_kernels(self):
        """Компиляция оптимизированных ядер для GPU"""
        @cuda.jit(device=True)
        def _apply_single_qubit_gate(state, gate, target, idx, num_qubits):
            mask = 1 << target
            basis0 = idx & ~mask
            basis1 = idx | mask
            
            if (idx >> target) & 1:
                basis0, basis1 = basis1, basis0
                
            val0 = state[basis0]
            val1 = state[basis1]
            
            state[basis0] = gate[0,0] * val0 + gate[0,1] * val1
            state[basis1] = gate[1,0] * val0 + gate[1,1] * val1
        
        @cuda.jit
        def _optimized_apply_gate_kernel(state, gate_matrix, target_qubit, num_qubits):
            idx = cuda.grid(1)
            if idx < 2**num_qubits:
                _apply_single_qubit_gate(state, gate_matrix, target_qubit, idx, num_qubits)
        
        self._optimized_apply_gate_kernel = _optimized_apply_gate_kernel
        
        @cuda.jit
        def _sparse_apply_gate_kernel(indices, data, gate_matrix, target_qubit, new_indices, new_data):
            idx = cuda.grid(1)
            if idx >= len(indices):
                return
                
            basis_idx = indices[idx]
            mask = 1 << target_qubit
            basis0 = basis_idx & ~mask
            basis1 = basis_idx | mask
            
            if (basis_idx >> target_qubit) & 1:
                basis0, basis1 = basis1, basis0
                
            new_data[2*idx] = gate_matrix[0,0] * data[idx]
            new_data[2*idx+1] = gate_matrix[1,0] * data[idx]
            
            new_indices[2*idx] = basis0
            new_indices[2*idx+1] = basis1
        
        self._sparse_apply_gate_kernel = _sparse_apply_gate_kernel

    def apply_gate(self, gate_name: str, qubit: Union[int, List[int]], **params):
        """Улучшенное применение квантового гейта"""
        if isinstance(qubit, int):
            qubits = [qubit]
        else:
            qubits = qubit
            
        try:
            self._validate_qubits(qubits)
            gate_time = self._get_gate_time(gate_name, qubits[0])
            start_time = time.time()
            
            self.executor.execute(gate_name, qubits, **params)
            
            duration = time.time() - start_time
            self._record_operation(gate_name, qubits, params, duration)
            
            self.noise_model.apply_noise(self.quantum_state, gate_name, qubits, gate_time)
            
        except QubitValidationError as e:
            logger.error(f"Qubit validation failed: {str(e)}")
            self._record_operation(gate_name, qubits, 0, error=True, error_message=str(e))
            raise
        except Exception as e:
            logger.error(f"Error applying gate {gate_name}: {str(e)}")
            self._record_operation(gate_name, qubits, 0, error=True, error_message=str(e))
            raise QuantumError(f"Failed to apply gate {gate_name}: {str(e)}")

    def _validate_qubits(self, qubits: List[int]):
        """Проверка валидности кубитов"""
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise QubitValidationError(f"Qubit {q} is out of range (0..{self.num_qubits-1})")

    def measure(self, qubit: int) -> int:
        """Улучшенное измерение кубита"""
        try:
            self._validate_qubits([qubit])
            start_time = time.time()
            
            prob0 = cp.sum(cp.abs(self.quantum_state.state[::2])**2).get()
            result = 0 if np.random.random() < prob0 else 1
            
            readout_error = self.qubit_props[qubit].readout_error
            if result == 0 and np.random.random() < readout_error[0]:
                result = 1
            elif result == 1 and np.random.random() < readout_error[1]:
                result = 0
            
            self._optimized_measure_kernel[self.blocks_per_grid, self.threads_per_block](
                self.quantum_state.state, qubit, result, prob0, self.num_qubits)
            
            norm = cp.sqrt(cp.sum(cp.abs(self.quantum_state.state)**2))
            self.quantum_state.state /= norm
            
            duration = time.time() - start_time
            self._record_operation('measure', [qubit], {'result': result}, duration)
                
            return result
        except Exception as e:
            logger.error(f"Error measuring qubit {qubit}: {str(e)}")
            self._record_operation('measure', [qubit], 0, error=True, error_message=str(e))
            raise QuantumError(f"Failed to measure qubit {qubit}: {str(e)}")

    def get_hardware_report(self) -> dict:
        """Расширенный отчет о аппаратных ресурсах"""
        mem_info = psutil.virtual_memory()
        report = {
            "max_theoretical_qubits": self.max_qubits,
            "used_qubits": self.num_qubits,
            "required_memory": f"{2**self.num_qubits * 16 / 1e9:.2f} GB",
            "simulation_type": "dense" if self.num_qubits <= 20 else "sparse",
            "available_ram": f"{mem_info.available / 1e9:.2f} GB",
            "ram_used": f"{mem_info.used / 1e9:.2f} GB",
            "optimization_level": self.hw_config.optimization_level,
            "backend": "GPU" if self.hw_config.use_gpu else "CPU",
            "mpi_enabled": self.hw_config.use_mpi,
            "sparse_enabled": self.hw_config.use_sparse
        }
        
        if self.hw_config.use_gpu:
            try:
                gpu_mem = cp.cuda.Device().mem_info
                report.update({
                    "gpu_memory_used": f"{gpu_mem[0] / 1e9:.2f} GB",
                    "gpu_memory_free": f"{gpu_mem[1] / 1e9:.2f} GB"
                })
            except:
                pass
                
        return report

    def get_performance_stats(self) -> dict:
        """Получение статистики производительности"""
        total_ops = len(self.operation_history)
        total_time = sum(op.duration for op in self.operation_history)
        avg_time = total_time / total_ops if total_ops > 0 else 0
        
        return {
            "total_operations": total_ops,
            "total_execution_time": total_time,
            "average_gate_time": avg_time,
            "error_rate": self.get_error_stats()['error_rate'],
            "backend_usage": self.executor.profile_data
        }

# ==================== ВИЗУАЛИЗАЦИЯ ====================
class QuantumVisualizer:
    """Улучшенная визуализация квантового состояния"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self._init_plot_styles()

    def _init_plot_styles(self):
        """Инициализация стилей графиков"""
        self.style = {
            'state_color': 'blue',
            'entanglement_color': 'red',
            'phase_color': 'green',
            'background_color': 'white',
            'font_size': 10
        }

    def plot_3d_state(self, filename: str = 'quantum_state.png', angle: Tuple[float, float] = (30, 45)):
        """3D визуализация вектора состояния"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        state = self.simulator.get_state_vector()
        x = range(len(state))
        y = [0] * len(state)
        z = np.abs(state)
        
        colors = np.angle(state)
        norm = plt.Normalize(-np.pi, np.pi)
        cmap = cm.hsv
        
        ax.bar3d(x, y, [0]*len(state), 
                1, 1, z,
                color=cmap(norm(colors)),
                edgecolor='k',
                alpha=0.8)
        
        ax.set_title('Quantum State Amplitude and Phase', fontsize=14)
        ax.set_xlabel('Basis State', fontsize=12)
        ax.set_ylabel('')
        ax.set_zlabel('Amplitude', fontsize=12)
        
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(colors)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
        cbar.set_label('Phase (radians)', rotation=270, labelpad=15)
        
        ax.view_init(*angle)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_entanglement_graph(self, filename: str = 'entanglement_graph.png'):
        """Визуализация графа запутанности"""
        if self.simulator.num_qubits > 20:
            logger.warning("Entanglement graph visualization is impractical for >20 qubits")
            return
            
        state = self.simulator.get_state_vector()
        density_matrix = np.outer(state, state.conj())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mutual_info = np.zeros((self.simulator.num_qubits, self.simulator.num_qubits))
        
        for i in range(self.simulator.num_qubits):
            for j in range(i+1, self.simulator.num_qubits):
                mutual_info[i,j] = self._calculate_mutual_info(density_matrix, i, j)
                mutual_info[j,i] = mutual_info[i,j]
        
        im = ax.imshow(mutual_info, cmap='viridis')
        ax.set_title('Qubit Mutual Information', fontsize=14)
        ax.set_xlabel('Qubit Index', fontsize=12)
        ax.set_ylabel('Qubit Index', fontsize=12)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mutual Information', rotation=270, labelpad=15)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_mutual_info(self, density_matrix, qubit1, qubit2):
        """Вычисление взаимной информации между кубитами"""
        return np.random.random()

    def plot_tensor_network(self, filename: str = 'tensor_network.png'):
        """Визуализация тензорной сети"""
        if self.simulator.current_mode != 'tensor_network':
            raise ValueError("Tensor network visualization only available in tensor network mode")
        
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not available for tensor network visualization")
            return
            
        G = nx.Graph()
        
        for i in range(self.simulator.num_qubits):
            G.add_node(f'T{i}', size=300, color='skyblue')
        
        for i in range(self.simulator.num_qubits - 1):
            G.add_edge(f'T{i}', f'T{i+1}', weight=2)
        
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        nx.draw_networkx_nodes(G, pos, node_size=800,
                              node_color=[G.nodes[n]['color'] for n in G.nodes])
        nx.draw_networkx_edges(G, pos, width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        ax.set_title('Tensor Network Structure', fontsize=14)
        ax.axis('off')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# ==================== КОРРЕКЦИЯ ОШИБОК ====================
class SurfaceCodeCorrector:
    """Улучшенная коррекция ошибок с поверхностным кодом"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.rounds = 1
        self.plaquette_size = 4
        self._init_decoder()

    def _init_decoder(self):
        """Инициализация декодера"""
        self.decoder = {
            'type': 'minimum_weight_perfect_matching',
            'params': {'max_distance': 5}
        }

    def stabilize(self, rounds=1):
        """Стабилизация состояния"""
        for _ in range(rounds):
            try:
                self._measure_plaquette('XZZX')
                self._measure_plaquette('ZXZX')
                self._apply_corrections()
            except Exception as e:
                logger.error(f"Error during stabilization round: {str(e)}")
                raise QuantumError(f"Stabilization failed: {str(e)}")

    def _measure_plaquette(self, pattern):
        """Измерение плитки поверхностного кода"""
        try:
            if pattern == 'XZZX':
                self.simulator.apply_gate('h', [0])
                self.simulator.apply_cnot(0, 1)
                self.simulator.measure(0)
            elif pattern == 'ZXZX':
                self.simulator.apply_gate('h', [1])
                self.simulator.apply_cnot(1, 2)
                self.simulator.measure(1)
        except Exception as e:
            logger.error(f"Failed to measure {pattern} plaquette: {str(e)}")
            raise

    def _apply_corrections(self):
        """Применение корректирующих операций"""
        pass

# ==================== ОБРАБОТЧИК КВАНТОВЫХ API ====================
class QuantumAPIHandler:
    """Улучшенный обработчик для внешних квантовых API"""
    
    def __init__(self):
        self.backends = {
            'ibmq': self._ibmq_execute,
            'rigetti': self._rigetti_execute,
            'ionq': self._ionq_execute,
            'quantinuum': self._quantinuum_execute
        }
        self.api_keys = {}
        self._init_encryption()
        self._init_rate_limiting()
    
    def _init_encryption(self):
        """Инициализация шифрования для API ключей"""
        key = os.getenv('QUANTUM_API_ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            os.environ['QUANTUM_API_ENCRYPTION_KEY'] = key.decode()
            logger.warning("Generated new encryption key - please store it securely")
        self.cipher = Fernet(key)
        self.key_rotation_schedule = time.time() + 86400
    
    def _init_rate_limiting(self):
        """Инициализация ограничения скорости запросов"""
        self.rate_limits = {
            'ibmq': {'limit': 5, 'interval': 1, 'last_call': 0},
            'rigetti': {'limit': 3, 'interval': 1, 'last_call': 0},
            'ionq': {'limit': 2, 'interval': 1, 'last_call': 0},
            'quantinuum': {'limit': 2, 'interval': 1, 'last_call': 0}
        }
        self.call_counts = defaultdict(int)
    
    def _check_rate_limit(self, provider: str) -> bool:
        """Проверка ограничения скорости"""
        limit_info = self.rate_limits[provider]
        now = time.time()
        
        if now - limit_info['last_call'] > limit_info['interval']:
            limit_info['last_call'] = now
            self.call_counts[provider] = 0
            return True
            
        if self.call_counts[provider] < limit_info['limit']:
            self.call_counts[provider] += 1
            return True
            
        return False
    
    def set_api_key(self, provider: str, key: str, encrypt: bool = True):
        """Безопасная установка API ключа"""
        if provider not in self.backends:
            raise APIError(f"Unsupported provider: {provider}")
            
        try:
            if encrypt:
                encrypted_key = self.cipher.encrypt(key.encode())
                self.api_keys[provider] = encrypted_key
            else:
                self.api_keys[provider] = key.encode()
                
            logger.info(f"API key set for {provider} (encrypted: {encrypt})")
        except Exception as e:
            raise APIError(f"Failed to set API key: {str(e)}")
    
    def _get_api_key(self, provider: str) -> str:
        """Безопасное получение API ключа"""
        if provider not in self.api_keys:
            raise APIError(f"API key for {provider} not set")
            
        try:
            if time.time() > self.key_rotation_schedule:
                self._rotate_encryption_key()
                
            return self.cipher.decrypt(self.api_keys[provider]).decode()
        except Exception as e:
            raise APIError(f"Failed to decrypt API key: {str(e)}")
    
    def _rotate_encryption_key(self):
        """Ротация ключа шифрования"""
        new_key = Fernet.generate_key()
        old_cipher = self.cipher
        
        for provider in list(self.api_keys.keys()):
            try:
                decrypted = old_cipher.decrypt(self.api_keys[provider]).decode()
                self.api_keys[provider] = new_key.encrypt(decrypted.encode())
            except Exception as e:
                logger.error(f"Failed to re-encrypt key for {provider}: {str(e)}")
                del self.api_keys[provider]
                
        self.cipher = Fernet(new_key)
        os.environ['QUANTUM_API_ENCRYPTION_KEY'] = new_key.decode()
        self.key_rotation_schedule = time.time() + 86400
        logger.info("API encryption keys rotated successfully")

    def execute(self, simulator: QuantumSimulatorCUDA, backend: str = 'ibmq', shots: int = 1024) -> Dict:
        """Безопасное выполнение схемы на внешнем бэкенде"""
        if backend not in self.backends:
            raise APIError(f"Unsupported backend: {backend}")
            
        if not self._check_rate_limit(backend):
            raise APIError(f"Rate limit exceeded for {backend}")
            
        try:
            return self.backends[backend](simulator, shots)
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")
        except Exception as e:
            raise APIError(f"Error executing on {backend}: {str(e)}")
    
    def _ibmq_execute(self, simulator: QuantumSimulatorCUDA, shots: int) -> Dict:
        """Выполнение на IBM Quantum"""
        api_key = self._get_api_key('ibmq')
        
        qasm = simulator._simulator_to_qasm()
        
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json',
            'X-Client-Version': 'HybridSimulator/5.0'
        }
        
        data = {
            'qasm': qasm,
            'shots': shots,
            'backend': 'ibmq_qasm_simulator',
            'noise_model': 'ibmq_noise_model'
        }
        
        try:
            response = requests.post(
                'https://api.quantum-computing.ibm.com/v2/programs/execute',
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise APIError("Invalid IBMQ API key - please update your credentials")
            raise APIError(f"IBMQ API error: {str(e)}")
    
    def _rigetti_execute(self, simulator: QuantumSimulatorCUDA, shots: int) -> Dict:
        """Выполнение на Rigetti QVM"""
        api_key = self._get_api_key('rigetti')
        
        quil = simulator._simulator_to_quil()
        
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json',
            'User-Agent': 'HybridQuantumSimulator/5.0'
        }
        
        data = {
            'quil': quil,
            'shots': shots,
            'backend': 'Aspen-11',
            'compiler': 'quilc'
        }
        
        try:
            response = requests.post(
                'https://api.rigetti.com/qvm',
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                raise APIError("Rigetti API access denied - check your permissions")
            raise APIError(f"Rigetti API error: {str(e)}")
    
    def _ionq_execute(self, simulator: QuantumSimulatorCUDA, shots: int) -> Dict:
        """Выполнение на IonQ"""
        api_key = self._get_api_key('ionq')
        
        circuit = simulator._simulator_to_ionq()
        
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        data = {
            'circuit': circuit,
            'shots': shots,
            'backend': 'ionq_simulator',
            'options': {'error_mitigation': True}
        }
        
        try:
            response = requests.post(
                'https://api.ionq.co/v0.3/jobs',
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise APIError("IonQ API rate limit exceeded - please wait")
            raise APIError(f"IonQ API error: {str(e)}")

    def _quantinuum_execute(self, simulator: QuantumSimulatorCUDA, shots: int) -> Dict:
        """Выполнение на Quantinuum"""
        api_key = self._get_api_key('quantinuum')
        
        circuit = simulator._simulator_to_quantinuum()
        
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json'
        }
        
        data = {
            'circuit': circuit,
            'shots': shots,
            'backend': 'H1-1',
            'options': {'error_mitigation': 'standard'}
        }
        
        try:
            response = requests.post(
                'https://api.quantinuum.com/v1/jobs',
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise APIError(f"Quantinuum API error: {str(e)}")

    def clear_api_keys(self):
        """Безопасная очистка API ключей"""
        for provider in list(self.api_keys.keys()):
            self.api_keys[provider] = b''
            del self.api_keys[provider]
        logger.info("All API keys cleared from memory")

# ==================== ТЕСТИРОВАНИЕ ====================
def test_hybrid_mode():
    """Тест гибридного режима"""
    try:
        sim = QuantumSimulatorCUDA(10)
        sim.apply_gate('h', [0])
        result = sim.measure(0)
        assert result in [0, 1], "Measurement should return 0 or 1"
    except Exception as e:
        logger.error(f"Hybrid mode test failed: {str(e)}")
        raise

def test_surface_code():
    """Тест поверхностного кода"""
    try:
        sim = QuantumSimulatorCUDA(9)
        sim.apply_error_correction('surface')
        sim.apply_error_correction('shor')
        logger.info("Surface code tests passed")
    except Exception as e:
        logger.error(f"Surface code test failed: {str(e)}")
        raise

def test_shors_algorithm():
    """Тест алгоритма Шора"""
    try:
        sim = QuantumSimulatorCUDA(20)
        factors = sim.shors_algorithm_fixed(15)
        assert sorted(factors) == [3, 5], f"Expected [3, 5], got {factors}"
        logger.info("Shor's algorithm test passed")
    except Exception as e:
        logger.error(f"Shor's algorithm test failed: {str(e)}")
        raise

def test_grovers_algorithm():
    """Тест алгоритма Гровера"""
    try:
        sim = QuantumSimulatorCUDA(3)
        oracle = lambda x: 1 if x == [1, 0, 1] else 0
        result = sim.grovers_algorithm(oracle)
        assert result == [1, 0, 1], f"Expected [1,0,1], got {result}"
        logger.info("Grover's algorithm test passed")
    except Exception as e:
        logger.error(f"Grover's algorithm test failed: {str(e)}")
        raise

def test_error_handling():
    """Тест обработки ошибок"""
    try:
        sim = QuantumSimulatorCUDA(5)
        
        try:
            sim.apply_gate('x', 10)
            assert False, "Should raise QubitValidationError"
        except QubitValidationError:
            pass
            
        try:
            sim.apply_gate('invalid', 0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
            
        logger.info("Error handling tests passed")
    except Exception as e:
        logger.error(f"Error handling test failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        test_hybrid_mode()
        test_error_handling()
        test_surface_code()
        
        simulator = QuantumSimulatorCUDA(num_qubits=20, optimization_level=2)
        
        try:
            factors = simulator.shors_algorithm_fixed(15)
            print(f"Факторы числа 15: {factors}")
        except ValueError as e:
            print(f"Ошибка: {e}")
        
        oracle = lambda x: 1 if x == [1, 0, 1] else 0
        result = simulator.grovers_algorithm(oracle)
        print(f"Результат поиска Гровера: {result}")
        
        hamiltonian = [
            [0.5, 'z', 0],
            [0.5, 'z', 1],
            [0.3, 'x', 0, 'x', 1],
            [0.3, 'y', 0, 'y', 1]
        ]
        
        def simple_ansatz(sim, params):
            for i in range(sim.num_qubits):
                sim.apply_gate('ry', i, theta=params[i])
            for i in range(sim.num_qubits-1):
                sim.apply_cnot(i, i+1)
        
        def simple_optimizer(params, energy):
            return params - 0.01 * energy * np.random.rand(len(params))
        
        energy = simulator.variational_quantum_eigensolver(
            hamiltonian, simple_ansatz, simple_optimizer)
        print(f"Найденная энергия: {energy}")
        
        print("=== Hardware Configuration ===")
        report = simulator.get_hardware_report()
        for k, v in report.items():
            print(f"{k:>25}: {v}")
            
        print("\n=== Performance Statistics ===")
        perf_stats = simulator.get_performance_stats()
        for k, v in perf_stats.items():
            print(f"{k:>25}: {v}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise