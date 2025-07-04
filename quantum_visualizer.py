# quantum_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Dict, List, Tuple, Optional
import networkx as nx
from quantum_core import QuantumSimulatorCUDA

class QuantumVisualizer:
    """Полнофункциональный модуль визуализации квантовых состояний"""
    
    def __init__(self, simulator: QuantumSimulatorCUDA):
        self.simulator = simulator
        self._init_plot_styles()
        self._init_color_maps()
    
    def _init_plot_styles(self):
        """Инициализация стилей графиков"""
        self.style = {
            'state_color': '#1f77b4',
            'entanglement_color': '#ff7f0e',
            'phase_color': '#2ca02c',
            'background_color': '#f8f8f8',
            'font_size': 12,
            'line_width': 2,
            'marker_size': 50
        }
    
    def _init_color_maps(self):
        """Инициализация цветовых карт"""
        self.cmaps = {
            'phase': cm.hsv,
            'entanglement': cm.viridis,
            'probability': cm.Blues
        }
    
    def plot_3d_state(self, filename: str = 'quantum_state.png', 
                     angle: Tuple[float, float] = (30, 45),
                     show_phase: bool = True):
        """
        3D визуализация квантового состояния с амплитудами и фазами
        Args:
            filename: Имя файла для сохранения
            angle: Углы обзора (elev, azim)
            show_phase: Показывать ли информацию о фазе
        """
        state = self.simulator.get_state_vector()
        n_qubits = self.simulator.num_qubits
        n_states = 2**n_qubits
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Подготовка данных
        x = np.arange(n_states)
        y = np.zeros(n_states)
        z = np.abs(state)
        phases = np.angle(state)
        colors = self.cmaps['phase']((phases + np.pi) / (2 * np.pi))
        
        # Построение графика
        bars = ax.bar3d(x, y, np.zeros(n_states), 
                        1, 1, z,
                        color=colors if show_phase else self.style['state_color'],
                        edgecolor='k',
                        alpha=0.8,
                        linewidth=0.5)
        
        # Настройки графика
        ax.set_title('Quantum State Visualization', fontsize=self.style['font_size']+2)
        ax.set_xlabel('Basis State', fontsize=self.style['font_size'])
        ax.set_zlabel('Amplitude', fontsize=self.style['font_size'])
        ax.set_yticks([])
        ax.grid(True)
        
        # Добавление цветовой шкалы для фазы
        if show_phase:
            mappable = cm.ScalarMappable(cmap=self.cmaps['phase'],
                                        norm=plt.Normalize(-np.pi, np.pi))
            mappable.set_array(phases)
            cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
            cbar.set_label('Phase (radians)', rotation=270, labelpad=20)
        
        ax.view_init(*angle)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_entanglement_graph(self, filename: str = 'entanglement_graph.png',
                              threshold: float = 0.1):
        """
        Визуализация графа запутанности между кубитами
        Args:
            filename: Имя файла для сохранения
            threshold: Порог для отображения связей
        """
        if self.simulator.num_qubits > 20:
            raise ValueError("Визуализация для >20 кубитов требует слишком много ресурсов")
        
        density_matrix = self._get_density_matrix()
        mutual_info = self._calculate_all_mutual_info(density_matrix)
        
        # Создание графа
        G = nx.Graph()
        for i in range(self.simulator.num_qubits):
            G.add_node(i, label=f'Q{i}')
        
        # Добавление ребер с весами
        for i in range(self.simulator.num_qubits):
            for j in range(i+1, self.simulator.num_qubits):
                if mutual_info[i,j] > threshold:
                    G.add_edge(i, j, weight=mutual_info[i,j])
        
        # Визуализация
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Рисуем узлы
        nx.draw_networkx_nodes(G, pos, node_size=500,
                             node_color=self.style['state_color'])
        
        # Рисуем ребра с шириной по взаимной информации
        edges = G.edges(data=True)
        widths = [d['weight']*3 for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, width=widths,
                              edge_color=self.style['entanglement_color'])
        
        # Подписи
        nx.draw_networkx_labels(G, pos, font_size=self.style['font_size'],
                              font_weight='bold')
        
        # Цветовая шкала
        sm = plt.cm.ScalarMappable(cmap=self.cmaps['entanglement'],
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8)
        cbar.set_label('Mutual Information', fontsize=self.style['font_size'])
        
        plt.title('Qubit Entanglement Graph', fontsize=self.style['font_size']+2)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tensor_network(self, filename: str = 'tensor_network.png'):
        """
        Визуализация тензорной сети (если используется)
        Args:
            filename: Имя файла для сохранения
        """
        if not hasattr(self.simulator, 'tensor_network'):
            raise ValueError("Тензорная сеть не инициализирована")
        
        tn = self.simulator.tensor_network
        G = nx.Graph()
        
        # Добавление узлов для тензоров
        for i in range(len(tn.tensors)):
            G.add_node(i, size=300 + 50*tn.tensors[i].size,
                      color=self.style['state_color'])
        
        # Добавление ребер для связей
        for i in range(len(tn.tensors)-1):
            bond_dim = tn.tensors[i].shape[-1]  # Размер связи
            G.add_edge(i, i+1, weight=bond_dim/tn.max_bond_dim)
        
        # Визуализация
        plt.figure(figsize=(12, 6))
        pos = nx.spring_layout(G, seed=42)
        
        # Размер узлов пропорционален размеру тензора
        sizes = [G.nodes[n]['size'] for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=sizes,
                             node_color=[G.nodes[n]['color'] for n in G.nodes],
                             alpha=0.8)
        
        # Ширина ребер пропорциональна размеру связи
        widths = [6*d['weight'] for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, width=widths,
                              edge_color=self.style['entanglement_color'],
                              alpha=0.6)
        
        # Подписи
        nx.draw_networkx_labels(G, pos, font_size=self.style['font_size'],
                              font_weight='bold')
        
        plt.title('Tensor Network Structure', fontsize=self.style['font_size']+2)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_density_matrix(self) -> np.ndarray:
        """Получение матрицы плотности"""
        state = self.simulator.get_state_vector()
        return np.outer(state, state.conj())
    
    def _calculate_all_mutual_info(self, density_matrix) -> np.ndarray:
        """
        Расчет матрицы взаимной информации для всех пар кубитов
        Args:
            density_matrix: Полная матрица плотности
        Returns:
            Матрицу взаимной информации размером n_qubits x n_qubits
        """
        n_qubits = self.simulator.num_qubits
        mutual_info = np.zeros((n_qubits, n_qubits))
        
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                mutual_info[i,j] = self._calculate_mutual_info(density_matrix, i, j)
                mutual_info[j,i] = mutual_info[i,j]
        
        return mutual_info
    
    def _calculate_mutual_info(self, density_matrix, qubit1, qubit2) -> float:
        """
        Расчет взаимной информации между двумя кубитами
        Args:
            density_matrix: Полная матрица плотности
            qubit1: Индекс первого кубита
            qubit2: Индекс второго кубита
        Returns:
            Взаимную информацию I(qubit1:qubit2)
        """
        # Редуцированная матрица плотности для двух кубитов
        rho_AB = self._partial_trace(density_matrix, [qubit1, qubit2])
        
        # Энтропия отдельных кубитов
        rho_A = self._partial_trace(density_matrix, [qubit1])
        rho_B = self._partial_trace(density_matrix, [qubit2])
        
        S_A = self._von_neumann_entropy(rho_A)
        S_B = self._von_neumann_entropy(rho_B)
        S_AB = self._von_neumann_entropy(rho_AB)
        
        # Взаимная информация: I(A:B) = S(A) + S(B) - S(A,B)
        return S_A + S_B - S_AB
    
    def _partial_trace(self, rho, keep_qubits: List[int]) -> np.ndarray:
        """
        Частичный след над матрицей плотности
        Args:
            rho: Матрица плотности (2^n x 2^n)
            keep_qubits: Список сохраняемых кубитов
        Returns:
            Редуцированная матрица плотности
        """
        n_qubits = int(np.log2(rho.shape[0]))
        keep_dim = 2**len(keep_qubits)
        trace_dim = 2**(n_qubits - len(keep_qubits))
        
        # Переиндексация кубитов
        qubits_sorted = sorted(keep_qubits)
        reorder = qubits_sorted + [q for q in range(n_qubits) if q not in qubits_sorted]
        
        # Реорганизация тензора
        tensor = rho.reshape([2]*n_qubits*2)
        tensor = np.moveaxis(tensor, reorder + [n_qubits + q for q in reorder], 
                            range(2*n_qubits))
        
        # Частичный след
        traced = tensor.reshape([keep_dim, trace_dim, keep_dim, trace_dim])
        return np.einsum('ijkl->ik', traced) / trace_dim
    
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """
        Расчет энтропии фон Неймана
        Args:
            rho: Матрица плотности
        Returns:
            Энтропию S = -Tr(ρ ln ρ)
        """
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[np.abs(eigvals) > 1e-10]  # Отсекаем нули
        return -np.sum(eigvals * np.log2(eigvals))