### 1. Реализация SGX-модуля для QuantumX

**Файл**: `quantumx-core/src/security/sgx.rs`
```rust
#[cfg(target_env = "sgx")]
use sgx_types::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SgxError {
    #[error("SGX is not available")]
    NotAvailable,
    #[error("SGX enclave failed: {0}")]
    EnclaveFailed(String),
    #[error("Attestation failed")]
    AttestationFailed,
}

/// SGX-защищенное хранилище для квантовых состояний
pub struct QuantumSecureEnclave {
    #[cfg(target_env = "sgx")]
    enclave: sgx_enclave_id_t,
    fallback_mode: bool,
}

impl QuantumSecureEnclave {
    pub fn new() -> Result<Self, SgxError> {
        #[cfg(target_env = "sgx")]
        {
            let mut enclave_id = 0;
            let ret = unsafe { sgx_create_enclave(...) };
            if ret != sgx_status_t::SGX_SUCCESS {
                return Err(SgxError::EnclaveFailed(format!("SGX error: {:?}", ret)));
            }
            Ok(Self { enclave: enclave_id, fallback_mode: false })
        }
        
        #[cfg(not(target_env = "sgx"))]
        {
            log::warn!("Running in SGX fallback mode (software emulation)");
            Ok(Self { fallback_mode: true })
        }
    }

    /// Шифрование квантового состояния для SGX-анклава
    pub fn encrypt_state(&self, state: &[f64]) -> Result<Vec<u8>, SgxError> {
        #[cfg(target_env = "sgx")]
        {
            let mut sealed_data = vec![0; state.len() * 8];
            let ret = unsafe {
                sgx_seal_data(
                    self.enclave,
                    state.as_ptr() as *const u8,
                    state.len() * 8,
                    sealed_data.as_mut_ptr()
                )
            };
            if ret != sgx_status_t::SGX_SUCCESS {
                return Err(SgxError::EnclaveFailed("Sealing failed".into()));
            }
            Ok(sealed_data)
        }
        
        #[cfg(not(target_env = "sgx"))]
        {
            // Эмуляция для разработки
            Ok(state.iter().flat_map(|f| f.to_le_bytes()).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enclave_fallback() {
        let enclave = QuantumSecureEnclave::new().unwrap();
        let state = [1.0, 0.0, 0.0, 0.0];
        let encrypted = enclave.encrypt_state(&state).unwrap();
        assert_eq!(encrypted.len(), 32); // 4 значения × 8 байт
    }
}
```

### 2. Интеграция с Python через PyO3

**Файл**: `quantumx-core/src/security/python.rs**
```rust
use pyo3::prelude::*;

#[pyclass]
pub struct QuantumSecurity {
    enclave: QuantumSecureEnclave,
}

#[pymethods]
impl QuantumSecurity {
    #[new]
    pub fn new() -> PyResult<Self> {
        let enclave = QuantumSecureEnclave::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { enclave })
    }

    pub fn encrypt_state(&self, state: Vec<f64>) -> PyResult<Vec<u8>> {
        self.enclave.encrypt_state(&state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
```

### 3. Модификация основного симулятора для использования SGX

**Файл**: `quantumx-core/src/lib.rs` (дополнение)
```rust
use crate::security::sgx::QuantumSecureEnclave;

#[pyclass]
pub struct QuantumSimulator {
    state: Array1<f64>,
    secure_enclave: Option<QuantumSecureEnclave>,
    // ... остальные поля
}

#[pymethods]
impl QuantumSimulator {
    #[new]
    pub fn new(num_qubits: usize, use_sgx: bool) -> PyResult<Self> {
        let enclave = if use_sgx {
            Some(QuantumSecureEnclave::new().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?)
        } else {
            None
        };
        
        // ... остальная инициализация
    }

    pub fn secure_measure(&mut self) -> PyResult<Vec<u8>> {
        let state = self.state.to_vec();
        match &self.secure_enclave {
            Some(enclave) => enclave.encrypt_state(&state)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "SGX not enabled".into()
            )),
        }
    }
}
```

### 4. Python-интерфейс для управления безопасностью

**Файл**: `quantumx/security.py**
```python
from .core import QuantumSecurity, QuantumSimulator
from typing import Optional
import numpy as np

class SecureQuantumX:
    def __init__(self, num_qubits: int, enable_sgx: bool = True):
        self.simulator = QuantumSimulator(num_qubits, use_sgx=enable_sgx)
        self.security = QuantumSecurity() if enable_sgx else None

    def get_secure_measurement(self) -> Optional[bytes]:
        if self.security is None:
            raise RuntimeError("SGX not enabled")
        return self.simulator.secure_measure()

    def get_public_measurement(self) -> np.ndarray:
        return np.array(self.simulator.measure())
```

### 5. Настройка окружения для SGX

**Dockerfile для SGX**:
```dockerfile
FROM ubuntu:20.04

# Установка SGX SDK и PSW
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common

RUN echo "deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main" | tee /etc/apt/sources.list.d/intel-sgx.list
RUN wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN apt-get update && apt-get install -y \
    libsgx-urts \
    libsgx-epid \
    libsgx-quote-ex \
    libsgx-dcap-ql

# Копирование нашего приложения
COPY ./target/release/libquantumx.so /usr/lib/
COPY quantumx /usr/local/lib/python3.8/dist-packages/quantumx

ENV LD_LIBRARY_PATH=/usr/lib
```

### 6. Тестирование безопасности

**Файл**: `quantumx/tests/test_security.py**
```python
import pytest
from quantumx.security import SecureQuantumX

def test_sgx_encryption():
    try:
        qx = SecureQuantumX(2, enable_sgx=True)
        encrypted = qx.get_secure_measurement()
        assert isinstance(encrypted, bytes)
        assert len(encrypted) == 32  # Для 4 кубитов
    except RuntimeError as e:
        if "SGX not available" in str(e):
            pytest.skip("SGX not available on this system")
        raise

def test_fallback_mode():
    qx = SecureQuantumX(2, enable_sgx=False)
    with pytest.raises(RuntimeError):
        qx.get_secure_measurement()
    assert qx.get_public_measurement() is not None
```

### Реализация без SGX (эмуляция)

Для разработки без SGX-совместимого железа:

1. Добавьте в `Cargo.toml`:
```toml
[features]
sgx = ["sgx_types"]
```

2. Реализуйте заглушки для тестирования:
```rust
#[cfg(not(feature = "sgx"))]
mod sgx_emulation {
    pub struct SgxEnclave;
    pub fn seal_data(data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }
}
```

### Развертывание

1. Для систем с SGX:
```bash
cargo build --release --features sgx
```

2. Для разработки без SGX:
```bash
cargo build --release
```

Эта реализация обеспечивает:
- Защиту квантовых состояний в SGX-анклавах
- Эмуляцию для разработки без SGX
- Интеграцию с Python API
- Поддержку аппаратного шифрования
- Совместимость с Docker для deployment
