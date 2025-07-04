# quantum_extensions.py
import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Optional, Callable
from quantum_core import QuantumSimulatorCUDA, QuantumError
from dataclasses import dataclass
import itertools
from enum import Enum

# ==================== ДЕКОДЕР ПОВЕРХНОСТНОГО КОДА ====================
class SurfaceCodeDecoder:
    """Реализация декодера минимального веса для поверхностного кода"""
    
    def __init__(self, distance: int = 3):
        self.distance = distance
        self.graph = self._build_decoding_graph()
        self.error_cache = {}
        
    def _build_decoding_graph(self):
        """Построение графа для декодирования"""
        graph = nx.Graph()
        size = 2 * self.distance - 1
        
        # Добавляем узлы для стабилизаторов X и Z
        for x in range(size):
            for y in range(size):
                if (x + y) % 2 == 0:
                    graph.add_node((x, y), type='stabilizer', 
                                 plaquette='X' if x % 2 == 0 else 'Z')
        
        # Добавляем ребра между соседними стабилизаторами
        for x in range(size):
            for y in range(size):
                if (x + y) % 2 == 0:
                    for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            graph.add_edge((x, y), (nx, ny), weight=1)
        
        return graph
    
    def decode(self, syndrome: Dict[Tuple[int, int], int]) -> List[Tuple[int, int, str]]:
        """
        Декодирование синдрома и возврат списка исправляющих операций
        Args:
            syndrome: Словарь {(x, y): error} где error=0/1
        Returns:
            Список коррекций [(x, y, type)] где type='X'/'Z'
        """
        non_zero = [pos for pos, err in syndrome.items() if err == 1]
        
        if not non_zero:
            return []
            
        syndrome_graph = nx.Graph()
        for i, pos1 in enumerate(non_zero):
            for j, pos2 in enumerate(non_zero[i+1:], i+1):
                try:
                    path = nx.shortest_path(self.graph, pos1, pos2, weight='weight')
                    weight = len(path) - 1
                    syndrome_graph.add_edge(i, j, weight=weight)
                except nx.NetworkXNoPath:
                    continue
        
        matching = nx.algorithms.matching.min_weight_matching(syndrome_graph, maxcardinality=True)
        
        corrections = []
        for i, j in matching:
            pos1, pos2 = non_zero[i], non_zero[j]
            path = nx.shortest_path(self.graph, pos1, pos2)
            
            for x, y in path[1:-1]:
                if (x + y) % 2 == 1:
                    plaquette_type = self.graph.nodes[path[0]]['plaquette']
                    corrections.append((x, y, plaquette_type))
        
        return corrections

# ==================== КОРРЕКЦИЯ ОШИБОК ====================
class SurfaceCodeCorrector:
    """Полная реализация коррекции ошибок с поверхностным кодом"""
    
    def __init__(self, simulator, distance: int = 3):
        self.simulator = simulator
        self.distance = distance
        self.decoder = SurfaceCodeDecoder(distance)
        self.syndrome_history = []
        self.rounds = 0
        self.qubit_map = self._initialize_qubit_map()
        self.stabilizers = self._initialize_stabilizers()
        
    def _initialize_qubit_map(self) -> Dict[Tuple[int, int], int]:
        qubit_map = {}
        size = 2 * self.distance - 1
        qubit_idx = 0
        
        for x in range(size):
            for y in range(size):
                if (x + y) % 2 == 1:
                    qubit_map[(x, y)] = qubit_idx
                    qubit_idx += 1
                    
        return qubit_map
    
    def _initialize_stabilizers(self) -> Dict[Tuple[int, int], List[int]]:
        stabilizers = {}
        size = 2 * self.distance - 1
        
        for x in range(size):
            for y in range(size):
                if (x + y) % 2 == 0:
                    neighbors = []
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size and (nx + ny) % 2 == 1:
                            neighbors.append(self.qubit_map[(nx, ny)])
                    stabilizers[(x, y)] = neighbors
                    
        return stabilizers
    
    def stabilize(self, rounds: int = 1):
        for _ in range(rounds):
            syndrome = self._measure_syndrome()
            corrections = self.decoder.decode(syndrome)
            self._apply_corrections(corrections)
            self.rounds += 1
    
    def _measure_syndrome(self) -> Dict[Tuple[int, int], int]:
        syndrome = {}
        
        for pos, qubits in self.stabilizers.items():
            x, y = pos
            if x % 2 == 0:
                syndrome[pos] = self._measure_x_stabilizer(qubits)
            else:
                syndrome[pos] = self._measure_z_stabilizer(qubits)
        
        self.syndrome_history.append(syndrome)
        return syndrome
    
    def _measure_x_stabilizer(self, qubits: List[int]) -> int:
        saved_states = [self.simulator.measure(q) for q in qubits]
        ancilla = self.simulator.num_qubits
        self.simulator.quantum_state = QuantumState(self.simulator.num_qubits + 1)
        
        self.simulator.apply_gate('h', ancilla)
        for q in qubits:
            self.simulator.apply_cnot(ancilla, q)
        self.simulator.apply_gate('h', ancilla)
        
        result = self.simulator.measure(ancilla)
        
        for q, state in zip(qubits, saved_states):
            if state == 1:
                self.simulator.apply_gate('x', q)
        
        return result
    
    def _measure_z_stabilizer(self, qubits: List[int]) -> int:
        saved_states = [self.simulator.measure(q) for q in qubits]
        ancilla = self.simulator.num_qubits
        self.simulator.quantum_state = QuantumState(self.simulator.num_qubits + 1)
        
        for q in qubits:
            self.simulator.apply_gate('h', q)
            self.simulator.apply_cnot(q, ancilla)
            self.simulator.apply_gate('h', q)
        
        result = self.simulator.measure(ancilla)
        
        for q, state in zip(qubits, saved_states):
            if state == 1:
                self.simulator.apply_gate('x', q)
        
        return result
    
    def _apply_corrections(self, corrections: List[Tuple[int, int, str]]):
        for x, y, op_type in corrections:
            if (x, y) in self.qubit_map:
                q = self.qubit_map[(x, y)]
                if op_type == 'X':
                    self.simulator.apply_gate('x', q)
                elif op_type == 'Z':
                    self.simulator.apply_gate('z', q)
    
    def get_logical_error_rate(self) -> float:
        if len(self.syndrome_history) < 2:
            return 0.0
            
        changed = 0
        total = 0
        
        for i in range(1, len(self.syndrome_history)):
            prev = self.syndrome_history[i-1]
            curr = self.syndrome_history[i]
            
            for pos in prev:
                if prev[pos] != curr[pos]:
                    changed += 1
                total += 1
                
        return changed / total if total > 0 else 0.0

class ShorCodeCorrector:
    """Корректор ошибок с использованием кода Шора (9 кубитов)"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        if simulator.num_qubits < 9:
            raise QuantumError("Shor code requires at least 9 qubits")
        
        self.encoded_qubit = 0
        self.ancilla_qubits = list(range(9, 17))
        
    def encode_logical_qubit(self, state: complex):
        if np.abs(state[0])**2 + np.abs(state[1])**2 > 1.0001:
            raise ValueError("Нормировка состояния должна быть <= 1")
        
        for block in [0, 3, 6]:
            self.simulator.quantum_state.state[0] = state[0]
            self.simulator.quantum_state.state[2**block] = state[1]
            self.simulator.apply_cnot(block, block+1)
            self.simulator.apply_cnot(block, block+2)

    def stabilize(self, rounds: int = 1):
        for _ in range(rounds):
            self._measure_x_stabilizers()
            self._measure_z_stabilizers()
            
    def _measure_x_stabilizers(self):
        for i in range(0, 9, 3):
            ancilla = self.ancilla_qubits[i//3]
            self.simulator.apply_gate('h', ancilla)
            self.simulator.apply_cnot(ancilla, i)
            self.simulator.apply_cnot(ancilla, i+1)
            self.simulator.apply_gate('h', ancilla)
            syndrome_bit = self.simulator.measure(ancilla)
            if syndrome_bit:
                self._correct_x_error(i)
    
    def _measure_z_stabilizers(self):
        for i in range(3):
            ancilla = self.ancilla_qubits[3 + i]
            self.simulator.apply_gate('h', ancilla)
            self.simulator.apply_cz(ancilla, i)
            self.simulator.apply_cz(ancilla, i+3)
            self.simulator.apply_gate('h', ancilla)
            syndrome_bit = self.simulator.measure(ancilla)
            if syndrome_bit:
                self._correct_z_error(i)
    
    def _correct_x_error(self, block: int):
        self.simulator.apply_gate('x', block)
    
    def _correct_z_error(self, block: int):
        self.simulator.apply_gate('z', block)
    
    def decode_logical_qubit(self) -> int:
        self.stabilize(rounds=3)
        
        for block in [0, 3, 6]:
            self._measure_and_correct_block(block)
        
        return self.simulator.measure(0)
    
    def _measure_and_correct_block(self, block):
        syndrome = 0
        for _ in range(3):
            syndrome = (syndrome << 1) | self._measure_block_syndrome(block)
        
        if syndrome & 0b011:
            self.simulator.apply_gate('x', block)
        elif syndrome & 0b101:
            self.simulator.apply_gate('x', block+1)
        elif syndrome & 0b110:
            self.simulator.apply_gate('x', block+2)

# ==================== LATTICE QCD ====================
class LatticeQCD:
    """Реализация квантовой хромодинамики на решетке"""
    
    def __init__(self, simulator, lattice_size=(4,4,4,4), quark_flavors=2, gauge_group='SU(3)'):
        self.simulator = simulator
        self.lattice_size = lattice_size
        self.quark_flavors = quark_flavors
        self.gauge_group = gauge_group
        self.n_dim = len(lattice_size)
        self.total_sites = np.prod(lattice_size)
        self.beta = 5.0
        self.kappa = 0.15
        self.mass = 0.1
        self._init_lattice()
        self._init_hamiltonian()
    
    def _init_lattice(self):
        required_qubits = self._calculate_required_qubits()
        if required_qubits > self.simulator.num_qubits:
            raise MemoryError(f"Недостаточно кубитов. Требуется: {required_qubits}")
            
        self.gauge_links = self._init_gauge_links()
        self.quark_fields = self._init_quark_fields()
    
    def _calculate_required_qubits(self) -> int:
        link_qubits = 3 if self.gauge_group == 'SU(2)' else 8
        quark_qubits_per_site = 4 * 3 * 2
        total_links = self.n_dim * self.total_sites
        total_quarks = self.quark_flavors * self.total_sites
        return total_links * link_qubits + total_quarks * quark_qubits_per_site
    
    def _init_gauge_links(self) -> Dict[Tuple, List[int]]:
        links = {}
        link_qubits = 3 if self.gauge_group == 'SU(2)' else 8
        current_qubit = 0
        
        for site in np.ndindex(*self.lattice_size):
            for mu in range(self.n_dim):
                links[(site, mu)] = list(range(current_qubit, current_qubit + link_qubits))
                current_qubit += link_qubits
                
        return links
    
    def _init_quark_fields(self) -> Dict[Tuple, List[int]]:
        fields = {}
        components_per_flavor = 4 * 3 * 2
        current_qubit = len(self.gauge_links) * (3 if self.gauge_group == 'SU(2)' else 8)
        
        for site in np.ndindex(*self.lattice_size):
            fields[site] = {}
            for flavor in range(self.quark_flavors):
                fields[site][flavor] = list(range(current_qubit, current_qubit + components_per_flavor))
                current_qubit += components_per_flavor
                
        return fields
    
    def _init_hamiltonian(self):
        self.hamiltonian_terms = []
        self._add_gauge_terms()
        self._add_fermion_terms()
        self._add_interaction_terms()
    
    def _add_gauge_terms(self):
        for site in np.ndindex(*self.lattice_size):
            for mu in range(self.n_dim):
                for nu in range(mu+1, self.n_dim):
                    term = {
                        'type': 'gauge',
                        'strength': self.beta,
                        'links': [
                            (site, mu),
                            (self._shift_site(site, mu), nu),
                            (self._shift_site(site, nu), mu),
                            (site, nu)
                        ]
                    }
                    self.hamiltonian_terms.append(term)
    
    def _add_fermion_terms(self):
        for site in np.ndindex(*self.lattice_size):
            for flavor in range(self.quark_flavors):
                term = {
                    'type': 'mass',
                    'strength': self.mass,
                    'site': site,
                    'flavor': flavor
                }
                self.hamiltonian_terms.append(term)
        
        for site in np.ndindex(*self.lattice_size):
            for mu in range(self.n_dim):
                neighbor = self._shift_site(site, mu)
                for flavor in range(self.quark_flavors):
                    term = {
                        'type': 'hopping',
                        'strength': self.kappa,
                        'direction': mu,
                        'sites': (site, neighbor),
                        'flavor': flavor
                    }
                    self.hamiltonian_terms.append(term)
    
    def _add_interaction_terms(self):
        for site in np.ndindex(*self.lattice_size):
            for mu in range(self.n_dim):
                for flavor in range(self.quark_flavors):
                    term = {
                        'type': 'interaction',
                        'strength': 1.0,
                        'site': site,
                        'direction': mu,
                        'flavor': flavor
                    }
                    self.hamiltonian_terms.append(term)
    
    def _shift_site(self, site, direction):
        return tuple((site[i] + (1 if i == direction else 0)) % self.lattice_size[i] 
                    for i in range(len(site)))
    
    def prepare_vacuum_state(self):
        for link_qubits in self.gauge_links.values():
            for q in link_qubits:
                self.simulator.apply_gate('x', q)
        
        for field_qubits in self._iterate_quark_qubits():
            for q in field_qubits:
                self.simulator.apply_gate('h', q)
                self.simulator.apply_gate('rz', q, theta=np.pi/2)
    
    def _iterate_quark_qubits(self):
        for site in self.quark_fields:
            for flavor in self.quark_fields[site]:
                yield self.quark_fields[site][flavor]
    
    def apply_hamiltonian(self, time_step: float):
        for term in self.hamiltonian_terms:
            if term['type'] == 'gauge':
                self._apply_gauge_term(term, time_step)
            elif term['type'] == 'mass':
                self._apply_mass_term(term, time_step)
            elif term['type'] == 'hopping':
                self._apply_hopping_term(term, time_step)
            elif term['type'] == 'interaction':
                self._apply_interaction_term(term, time_step)
    
    def _apply_gauge_term(self, term, time_step):
        link_qubits = []
        for link in term['links']:
            link_qubits.extend(self.gauge_links[link])
        
        strength = term['strength'] * time_step
        self._apply_su_n_gate(link_qubits, strength, self.gauge_group)
    
    def _apply_mass_term(self, term, time_step):
        qubits = self.quark_fields[term['site']][term['flavor']]
        strength = term['strength'] * time_step
        
        for q in qubits[::2]:
            self.simulator.apply_gate('rz', q, theta=-strength)
    
    def _apply_hopping_term(self, term, time_step):
        site1, site2 = term['sites']
        flavor = term['flavor']
        strength = term['strength'] * time_step
        
        qubits1 = self.quark_fields[site1][flavor]
        qubits2 = self.quark_fields[site2][flavor]
        
        for q1, q2 in zip(qubits1, qubits2):
            self.simulator.apply_gate('cx', q1, q2)
            self.simulator.apply_gate('rz', q2, theta=-strength)
            self.simulator.apply_gate('cx', q1, q2)
    
    def _apply_interaction_term(self, term, time_step):
        site = term['site']
        mu = term['direction']
        flavor = term['flavor']
        strength = term['strength'] * time_step
        
        link_qubits = self.gauge_links[(site, mu)]
        quark_qubits = self.quark_fields[site][flavor]
        
        for i in range(0, min(len(link_qubits), len(quark_qubits)), 2):
            self.simulator.apply_gate('cx', link_qubits[i], quark_qubits[i])
            self.simulator.apply_gate('rz', quark_qubits[i], theta=-strength)
            self.simulator.apply_gate('cx', link_qubits[i], quark_qubits[i])
    
    def _apply_su_n_gate(self, qubits, strength, group):
        if group == 'SU(2)':
            if len(qubits) != 3:
                raise ValueError("Для SU(2) требуется 3 кубита на линк")
            
            self.simulator.apply_gate('h', qubits[0])
            self.simulator.apply_gate('cx', qubits[0], qubits[1])
            self.simulator.apply_gate('rz', qubits[1], theta=strength)
            self.simulator.apply_gate('cx', qubits[0], qubits[1])
            self.simulator.apply_gate('h', qubits[0])
            
        elif group == 'SU(3)':
            if len(qubits) != 8:
                raise ValueError("Для SU(3) требуется 8 кубитов на линк")
            
            for i in range(0, 8, 3):
                self.simulator.apply_gate('ry', qubits[i], theta=np.pi/4)
                self.simulator.apply_gate('cx', qubits[i], qubits[i+1])
                self.simulator.apply_gate('rz', qubits[i+1], theta=strength/2)
                self.simulator.apply_gate('cx', qubits[i], qubits[i+2])
                self.simulator.apply_gate('rz', qubits[i+2], theta=strength/2)
    
    def measure_plaquette(self, site, mu, nu) -> float:
        link_qubits = []
        for link in [
            (site, mu),
            (self._shift_site(site, mu), nu),
            (self._shift_site(site, nu), mu),
            (site, nu)
        ]:
            link_qubits.extend(self.gauge_links[link])
        
        results = []
        for _ in range(100):
            result = 0
            for q in link_qubits:
                result += (self.simulator.measure(q) * 2 - 1)
            results.append(result / len(link_qubits))
            
        return np.mean(results)
    
    def measure_chiral_condensate(self) -> float:
        condensate = 0.0
        for site in np.ndindex(*self.lattice_size):
            for flavor in range(self.quark_flavors):
                qubits = self.quark_fields[site][flavor]
                for q in qubits[::2]:
                    condensate += (self.simulator.measure(q) - 0.5) * 2
                    
        return condensate / (self.total_sites * self.quark_flavors)
    
    def thermalize(self, steps: int = 100, delta_t: float = 0.1):
        for step in range(steps):
            self.apply_hamiltonian(delta_t)
            
            if step % 10 == 0:
                self._random_gauge_transformation()
            
            if step % 50 == 0:
                print(f"Thermalization step {step}/{steps}")
    
    def _random_gauge_transformation(self):
        for site in np.ndindex(*self.lattice_size):
            if self.gauge_group == 'SU(2)':
                angle = np.random.uniform(0, 2*np.pi)
                qubits = self._find_connected_links(site)
                for q in qubits:
                    self.simulator.apply_gate('rx', q, theta=angle)
            else:
                angles = np.random.uniform(0, 2*np.pi, size=3)
                qubits = self._find_connected_links(site)
                for i, q in enumerate(qubits[:3]):
                    self.simulator.apply_gate('rz', q, theta=angles[i])
    
    def _find_connected_links(self, site):
        connected = []
        for mu in range(self.n_dim):
            connected.extend(self.gauge_links[(site, mu)])
            neighbor = self._shift_site(site, mu)
            connected.extend(self.gauge_links[(neighbor, mu)])
        return connected

# ==================== КВАНТОВЫЕ АЛГОРИТМЫ ====================
def shors_algorithm_fixed(simulator: QuantumSimulatorCUDA, N: int) -> List[int]:
    """Полная реализация алгоритма Шора для факторизации"""
    if N % 2 == 0:
        return [2, N//2]
    
    a = np.random.randint(2, N-1)
    while math.gcd(a, N) != 1:
        a = np.random.randint(2, N-1)
    
    period = _quantum_period_finding(simulator, a, N)
    
    factors = []
    if period % 2 == 0:
        candidate = math.gcd(a**(period//2) - 1, N)
        if candidate not in [1, N]:
            factors.append(candidate)
            factors.append(N // candidate)
    
    return sorted(factors) if factors else [1, N]

def _quantum_period_finding(simulator: QuantumSimulatorCUDA, a: int, N: int) -> int:
    """Квантовая часть алгоритма Шора: нахождение периода"""
    n = simulator.num_qubits
    t = 2 * n  # Число кубитов для оценки фазы
    
    # Инициализация регистра
    for q in range(t):
        simulator.apply_gate('h', q)
    
    # Добавление модулярного возведения в степень
    simulator.apply_modular_exponentiation(a, N, list(range(t)), list(range(t, t+n)))
    
    # Обратное преобразование Фурье
    for q in reversed(range(t)):
        simulator.apply_gate('h', q)
        for j in range(q):
            simulator.apply_cu1(q, j, angle=-np.pi/(2**(q-j)))
    
    # Измерение и определение периода
    measurements = [simulator.measure(q) for q in range(t)]
    phase = sum(bit * 2**(-i-1) for i, bit in enumerate(reversed(measurements)))
    return int(1/phase)

def grovers_algorithm(simulator: QuantumSimulatorCUDA, oracle: Callable, iterations: int = None) -> List[int]:
    """Реализация алгоритма Гровера"""
    n = simulator.num_qubits
    if iterations is None:
        iterations = int((np.pi/4)*np.sqrt(2**n))
    
    # Инициализация суперпозиции
    for q in range(n):
        simulator.apply_gate('h', q)
    
    # Итерации Гровера
    for _ in range(iterations):
        _apply_oracle(simulator, oracle)
        _apply_diffusion(simulator, n)
    
    # Измерение результата
    return [simulator.measure(q) for q in range(n)]

def _apply_oracle(simulator: QuantumSimulatorCUDA, oracle: Callable):
    """Применение оракула алгоритма Гровера"""
    n = simulator.num_qubits
    ancilla = n
    
    # Расширяем состояние для вспомогательного кубита
    simulator.quantum_state = QuantumState(n + 1)
    
    # Применяем оракул через управляемые операции
    for state in range(2**n):
        if oracle([int(b) for b in f"{state:0{n}b}"]):
            simulator.apply_gate('x', ancilla)
            simulator.apply_mct(list(range(n)), ancilla)
            simulator.apply_gate('x', ancilla)

def _apply_diffusion(simulator: QuantumSimulatorCUDA, n: int):
    """Применение диффузионного оператора"""
    for q in range(n):
        simulator.apply_gate('h', q)
    for q in range(n):
        simulator.apply_gate('x', q)
    simulator.apply_gate('h', n-1)
    simulator.apply_mct(list(range(n-1)), n-1)
    simulator.apply_gate('h', n-1)
    for q in range(n):
        simulator.apply_gate('x', q)
    for q in range(n):
        simulator.apply_gate('h', q)

# ==================== ИНТЕГРАЦИЯ С СИМУЛЯТОРОМ ====================
def add_lattice_qcd(simulator: QuantumSimulatorCUDA, lattice_size=(4,4,4,4), quark_flavors=2, gauge_group='SU(3)'):
    """Добавление модуля Lattice QCD к симулятору"""
    simulator.lattice_qcd = LatticeQCD(simulator, lattice_size, quark_flavors, gauge_group)

def measure_qcd_observables(simulator: QuantumSimulatorCUDA) -> Dict[str, float]:
    """Измерение основных наблюдаемых в QCD"""
    if not hasattr(simulator, 'lattice_qcd'):
        raise ValueError("Lattice QCD module not initialized")
        
    results = {
        'plaquette': _measure_average_plaquette(simulator),
        'chiral_condensate': simulator.lattice_qcd.measure_chiral_condensate(),
        'energy_density': _estimate_energy_density(simulator)
    }
    return results

def _measure_average_plaquette(simulator: QuantumSimulatorCUDA) -> float:
    lattice = simulator.lattice_qcd
    total = 0.0
    count = 0
    
    for site in np.ndindex(*lattice.lattice_size):
        for mu in range(lattice.n_dim):
            for nu in range(mu+1, lattice.n_dim):
                total += lattice.measure_plaquette(site, mu, nu)
                count += 1
                
    return total / count

def _estimate_energy_density(simulator: QuantumSimulatorCUDA) -> float:
    avg_plaquette = _measure_average_plaquette(simulator)
    return (1 - avg_plaquette) * simulator.lattice_qcd.beta