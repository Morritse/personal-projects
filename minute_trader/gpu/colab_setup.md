# Colab Setup Instructions

Copy this into a new code cell at the start of your notebook:

```python
# Install TA-Lib C library
%%bash
apt-get update
apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
cd ..
rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/
```

Then in a new cell:

```python
# Install Python packages
!pip install pandas numpy ta-lib yfinance alpaca-trade-api tqdm cupy-cuda12x numba

# Verify installations
import talib
import cupy as cp
print("TA-Lib and CuPy installed successfully!")
```

This will:
1. Install the TA-Lib C library using bash commands
2. Install required Python packages
3. Verify that everything installed correctly
