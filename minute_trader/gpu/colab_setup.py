# Install TA-Lib C library
!apt-get update
!apt-get install -y build-essential wget
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xvzf ta-lib-0.4.0-src.tar.gz
%cd ta-lib/
!./configure --prefix=/usr
!make
!sudo make install
%cd ..
!rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Install Python packages
!pip install pandas numpy ta-lib yfinance alpaca-trade-api tqdm cupy-cuda12x numba

# Verify installations
import talib
import cupy as cp
print("TA-Lib and CuPy installed successfully!")
