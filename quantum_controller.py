# quantum_controller.py
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                            QTextEdit, QProgressBar, QCheckBox, QGroupBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from quantum_core import QuantumSimulatorCUDA
from quantum_extensions import LatticeQCD
from quantum_visualizer import QuantumVisualizer

class QuantumController(QMainWindow):
    """Главный управляющий интерфейс для квантового симулятора"""
    
    THEMES = {
        "dark": {
            "bg": "#121212",
            "text": "#FFFFFF",
            "accent": "#BB86FC",
            "secondary": "#03DAC6",
            "widget_bg": "#1E1E1E"
        },
        "light": {
            "bg": "#FFFFFF",
            "text": "#000000",
            "accent": "#6200EE",
            "secondary": "#018786",
            "widget_bg": "#F5F5F5"
        },
        "blue": {
            "bg": "#0A192F",
            "text": "#CCD6F6",
            "accent": "#64FFDA",
            "secondary": "#1E90FF",
            "widget_bg": "#172A45"
        }
    }
    
    def __init__(self):
        super().__init__()
        self.simulator = None
        self.visualizer = None
        self.current_theme = "dark"
        self.initUI()
        self.setupConnections()
        
    def initUI(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Hybrid Quantum Simulator Controller")
        self.resize(1000, 700)
        
        # Центральный виджет и макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Панель управления
        self.setupControlPanel(main_layout)
        
        # Вкладки
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Вкладка симуляции
        self.setupSimulationTab()
        
        # Вкладка визуализации
        self.setupVisualizationTab()
        
        # Вкладка мониторинга
        self.setupMonitorTab()
        
        # Установка темы
        self.setTheme(self.current_theme)
        
    def setupControlPanel(self, layout):
        """Настройка панели управления"""
        control_group = QGroupBox("Управление симулятором")
        control_layout = QHBoxLayout()
        
        # Выбор темы
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.THEMES.keys())
        self.theme_combo.setCurrentText(self.current_theme)
        
        # Кнопки управления
        self.init_btn = QPushButton("Инициализировать")
        self.reset_btn = QPushButton("Сбросить")
        self.run_btn = QPushButton("Запустить")
        
        # Параметры симуляции
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setRange(1, 50)
        self.qubit_spin.setValue(5)
        
        # Добавление элементов
        control_layout.addWidget(QLabel("Кубиты:"))
        control_layout.addWidget(self.qubit_spin)
        control_layout.addWidget(self.init_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.run_btn)
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Тема:"))
        control_layout.addWidget(self.theme_combo)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
    
    def setupSimulationTab(self):
        """Настройка вкладки симуляции"""
        sim_tab = QWidget()
        self.tab_widget.addTab(sim_tab, "Симуляция")
        layout = QVBoxLayout(sim_tab)
        
        # Выбор алгоритма
        algo_group = QGroupBox("Алгоритмы")
        algo_layout = QVBoxLayout()
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Гровера", "Шора", "VQE", "QAOA"])
        
        self.algo_param1 = QLineEdit()
        self.algo_param2 = QLineEdit()
        
        algo_layout.addWidget(QLabel("Алгоритм:"))
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addWidget(QLabel("Параметр 1:"))
        algo_layout.addWidget(self.algo_param1)
        algo_layout.addWidget(QLabel("Параметр 2:"))
        algo_layout.addWidget(self.algo_param2)
        
        self.run_algo_btn = QPushButton("Выполнить алгоритм")
        algo_layout.addWidget(self.run_algo_btn)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)
        
        # Квантовые гейты
        gate_group = QGroupBox("Гейты")
        gate_layout = QVBoxLayout()
        
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["H", "X", "Y", "Z", "CX", "CCX", "RX", "RY", "RZ"])
        
        self.target_qubit = QSpinBox()
        self.target_qubit.setRange(0, 49)
        
        self.control_qubit = QSpinBox()
        self.control_qubit.setRange(-1, 49)
        self.control_qubit.setValue(-1)
        
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(-2*np.pi, 2*np.pi)
        self.angle_input.setValue(0.0)
        
        gate_layout.addWidget(QLabel("Гейт:"))
        gate_layout.addWidget(self.gate_combo)
        gate_layout.addWidget(QLabel("Целевой кубит:"))
        gate_layout.addWidget(self.target_qubit)
        gate_layout.addWidget(QLabel("Контрольный кубит:"))
        gate_layout.addWidget(self.control_qubit)
        gate_layout.addWidget(QLabel("Угол (для RX/RY/RZ):"))
        gate_layout.addWidget(self.angle_input)
        
        self.apply_gate_btn = QPushButton("Применить гейт")
        gate_layout.addWidget(self.apply_gate_btn)
        
        gate_group.setLayout(gate_layout)
        layout.addWidget(gate_group)
        
        # Лог операций
        self.op_log = QTextEdit()
        self.op_log.setReadOnly(True)
        layout.addWidget(QLabel("Журнал операций:"))
        layout.addWidget(self.op_log)
    
    def setupVisualizationTab(self):
        """Настройка вкладки визуализации"""
        vis_tab = QWidget()
        self.tab_widget.addTab(vis_tab, "Визуализация")
        layout = QVBoxLayout(vis_tab)
        
        # Управление визуализацией
        vis_control = QHBoxLayout()
        
        self.vis_type_combo = QComboBox()
        self.vis_type_combo.addItems(["3D состояние", "Граф запутанности", "Тензорная сеть"])
        
        self.vis_param1 = QDoubleSpinBox()
        self.vis_param1.setRange(0, 1)
        self.vis_param1.setValue(0.1)
        
        self.generate_vis_btn = QPushButton("Сгенерировать")
        self.save_vis_btn = QPushButton("Сохранить")
        
        vis_control.addWidget(QLabel("Тип:"))
        vis_control.addWidget(self.vis_type_combo)
        vis_control.addWidget(QLabel("Порог:"))
        vis_control.addWidget(self.vis_param1)
        vis_control.addWidget(self.generate_vis_btn)
        vis_control.addWidget(self.save_vis_btn)
        
        layout.addLayout(vis_control)
        
        # Область отображения
        self.vis_label = QLabel()
        self.vis_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.vis_label)
    
    def setupMonitorTab(self):
        """Настройка вкладки мониторинга"""
        mon_tab = QWidget()
        self.tab_widget.addTab(mon_tab, "Мониторинг")
        layout = QVBoxLayout(mon_tab)
        
        # Информация о системе
        sys_group = QGroupBox("Системная информация")
        sys_layout = QVBoxLayout()
        
        self.hw_info = QTextEdit()
        self.hw_info.setReadOnly(True)
        
        sys_layout.addWidget(self.hw_info)
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        # Графики производительности
        perf_group = QGroupBox("Производительность")
        perf_layout = QVBoxLayout()
        
        self.cpu_usage = QProgressBar()
        self.mem_usage = QProgressBar()
        self.gpu_usage = QProgressBar()
        
        perf_layout.addWidget(QLabel("Использование CPU:"))
        perf_layout.addWidget(self.cpu_usage)
        perf_layout.addWidget(QLabel("Использование памяти:"))
        perf_layout.addWidget(self.mem_usage)
        perf_layout.addWidget(QLabel("Использование GPU:"))
        perf_layout.addWidget(self.gpu_usage)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Таймер обновления
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.updateMonitor)
        self.monitor_timer.start(1000)
    
    def setupConnections(self):
        """Настройка соединений сигналов и слотов"""
        self.init_btn.clicked.connect(self.initSimulator)
        self.reset_btn.clicked.connect(self.resetSimulator)
        self.run_btn.clicked.connect(self.runSimulation)
        self.theme_combo.currentTextChanged.connect(self.setTheme)
        self.apply_gate_btn.clicked.connect(self.applyGate)
        self.run_algo_btn.clicked.connect(self.runAlgorithm)
        self.generate_vis_btn.clicked.connect(self.generateVisualization)
        self.save_vis_btn.clicked.connect(self.saveVisualization)
    
    def initSimulator(self):
        """Инициализация симулятора"""
        num_qubits = self.qubit_spin.value()
        try:
            self.simulator = QuantumSimulatorCUDA(num_qubits=num_qubits)
            self.visualizer = QuantumVisualizer(self.simulator)
            self.logOperation(f"Инициализирован симулятор с {num_qubits} кубитами")
            self.updateHardwareInfo()
        except Exception as e:
            self.logOperation(f"Ошибка инициализации: {str(e)}", error=True)
    
    def resetSimulator(self):
        """Сброс симулятора"""
        if self.simulator:
            self.simulator.reset()
            self.logOperation("Симулятор сброшен в начальное состояние")
    
    def runSimulation(self):
        """Запуск симуляции"""
        self.logOperation("Запуск симуляции...")
        # Здесь можно добавить конкретную логику симуляции
    
    def applyGate(self):
        """Применение квантового гейта"""
        if not self.simulator:
            self.logOperation("Симулятор не инициализирован", error=True)
            return
        
        gate = self.gate_combo.currentText()
        target = self.target_qubit.value()
        control = self.control_qubit.value()
        angle = self.angle_input.value()
        
        try:
            if gate in ["H", "X", "Y", "Z"]:
                self.simulator.apply_gate(gate.lower(), target)
                self.logOperation(f"Применен гейт {gate} к кубиту {target}")
            elif gate == "CX":
                if control < 0:
                    raise ValueError("Не указан контрольный кубит")
                self.simulator.apply_cnot(control, target)
                self.logOperation(f"Применен CNOT: контроль {control}, цель {target}")
            elif gate in ["RX", "RY", "RZ"]:
                self.simulator.apply_gate(gate.lower(), target, theta=angle)
                self.logOperation(f"Применен {gate}({angle:.2f}) к кубиту {target}")
        except Exception as e:
            self.logOperation(f"Ошибка применения гейта: {str(e)}", error=True)
    
    def runAlgorithm(self):
        """Выполнение квантового алгоритма"""
        algo = self.algo_combo.currentText()
        param1 = self.algo_param1.text()
        
        try:
            if algo == "Гровера":
                # Здесь должна быть реализация алгоритма Гровера
                self.logOperation("Запущен алгоритм Гровера")
            elif algo == "Шора":
                N = int(param1)
                factors = self.simulator.shors_algorithm_fixed(N)
                self.logOperation(f"Алгоритм Шора: множители {N} = {factors}")
        except Exception as e:
            self.logOperation(f"Ошибка выполнения алгоритма: {str(e)}", error=True)
    
    def generateVisualization(self):
        """Генерация визуализации"""
        if not self.visualizer:
            self.logOperation("Визуализатор не инициализирован", error=True)
            return
        
        vis_type = self.vis_type_combo.currentText()
        threshold = self.vis_param1.value()
        
        try:
            if vis_type == "3D состояние":
                self.visualizer.plot_3d_state("temp_vis.png")
            elif vis_type == "Граф запутанности":
                self.visualizer.plot_entanglement_graph("temp_vis.png", threshold)
            elif vis_type == "Тензорная сеть":
                self.visualizer.plot_tensor_network("temp_vis.png")
            
            # Отображение в интерфейсе
            self.displayImage("temp_vis.png")
            self.logOperation(f"Сгенерирована визуализация: {vis_type}")
        except Exception as e:
            self.logOperation(f"Ошибка визуализации: {str(e)}", error=True)
    
    def saveVisualization(self):
        """Сохранение визуализации в файл"""
        # Реализация сохранения файла
        self.logOperation("Визуализация сохранена")
    
    def displayImage(self, filename):
        """Отображение изображения в интерфейсе"""
        # Здесь должна быть реализация отображения изображения
        pass
    
    def updateHardwareInfo(self):
        """Обновление информации о системе"""
        if self.simulator:
            report = self.simulator.get_hardware_report()
            info_text = "\n".join(f"{k:>25}: {v}" for k, v in report.items())
            self.hw_info.setPlainText(info_text)
    
    def updateMonitor(self):
        """Обновление мониторинга"""
        # Здесь должна быть реализация обновления показателей CPU/GPU/памяти
        pass
    
    def logOperation(self, message: str, error: bool = False):
        """Логирование операций"""
        if error:
            self.op_log.append(f"<font color='red'>{message}</font>")
        else:
            self.op_log.append(message)
    
    def setTheme(self, theme_name: str):
        """Установка цветовой темы"""
        if theme_name not in self.THEMES:
            return
            
        self.current_theme = theme_name
        theme = self.THEMES[theme_name]
        
        # Применение темы ко всему приложению
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(theme["bg"]))
        palette.setColor(QPalette.WindowText, QColor(theme["text"]))
        palette.setColor(QPalette.Base, QColor(theme["widget_bg"]))
        palette.setColor(QPalette.Text, QColor(theme["text"]))
        palette.setColor(QPalette.Button, QColor(theme["accent"]))
        palette.setColor(QPalette.ButtonText, QColor(theme["bg"]))
        self.setPalette(palette)
        
        # Дополнительные стили
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {theme["accent"]};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: {theme["text"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: {theme["accent"]};
            }}
            QPushButton {{
                background-color: {theme["accent"]};
                color: {theme["bg"]};
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {theme["secondary"]};
            }}
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Установка стиля Fusion для всех платформ
    app.setStyle("Fusion")
    
    controller = QuantumController()
    controller.show()
    sys.exit(app.exec_())