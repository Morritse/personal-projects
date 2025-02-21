# Smart Colab Setup Commands

Copy these commands into a new code cell. They'll only install what's needed:

```python
import sys
import subprocess
from IPython.display import clear_output

def is_package_installed(package_name):
    """Check if a package is installed and importable"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def get_cuda_version():
    """Get CUDA version from nvidia-smi"""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader,nounits'])
        version = output.decode('utf-8').strip()
        major = version[:2]  # Get major version (e.g., '11' from '11.2')
        return major
    except:
        return None

def check_talib_installation():
    """Check if TA-Lib C library is properly installed"""
    try:
        import talib
        # Try a simple function call to verify the C library works
        test_data = [1.0, 2.0, 3.0]
        talib.SMA(test_data)
        return True
    except Exception as e:
        if "libta_lib.so.0: cannot open shared object file" in str(e):
            return False
        return False

def install_talib():
    """Install TA-Lib C library and Python wrapper"""
    try:
        print("Installing TA-Lib...")
        # Install system dependencies
        !apt-get update -qq
        !apt-get install -y build-essential wget > /dev/null 2>&1
        
        # Download and install TA-Lib
        !wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        !tar -xzf ta-lib-0.4.0-src.tar.gz
        %cd ta-lib/
        !./configure --prefix=/usr > /dev/null 2>&1
        !make -j$(nproc) > /dev/null 2>&1
        !sudo make install > /dev/null 2>&1
        %cd ..
        !rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/
        
        # Install Python wrapper
        !pip install -q ta-lib
        print("TA-Lib installed successfully!")
        return True
    except Exception as e:
        print(f"Error installing TA-Lib: {str(e)}")
        return False

print("Setting up GPU environment...")

# Check CUDA availability and version
cuda_version = get_cuda_version()
if not cuda_version:
    print("CUDA not available. Please enable GPU runtime in Colab:")
    print("Runtime → Change runtime type → Hardware accelerator → GPU")
    sys.exit(1)
print(f"Found CUDA version: {cuda_version}")

# First uninstall any conflicting packages
!pip uninstall -y PyYAML alpaca-trade-api

# Install packages in correct order with specific versions
print("\nInstalling dependencies in order...")
!pip install -q 'PyYAML==6.0.1'
!pip install -q 'alpaca-trade-api>=2.3.0'

# Check TA-Lib installation
if not is_package_installed('talib') or not check_talib_installation():
    if not install_talib():
        print("Failed to install TA-Lib. Please try again.")
        sys.exit(1)

# Check and install other required packages
required_packages = {
    'cupy': f'cupy-cuda{cuda_version}x',  # Use detected CUDA version
    'numba': 'numba',
    'pandas': 'pandas>=1.5.0',
    'numpy': 'numpy>=1.21.0',
    'yfinance': 'yfinance>=0.2.18',
    'tqdm': 'tqdm'
}

print("\nChecking and installing required packages...")
for import_name, pip_name in required_packages.items():
    if not is_package_installed(import_name):
        print(f"Installing {pip_name}...")
        try:
            !pip install -q {pip_name}
            clear_output(wait=True)
        except Exception as e:
            print(f"Error installing {pip_name}: {str(e)}")
            if import_name == 'cupy':
                print("Trying alternative CuPy installation...")
                !pip install -q cupy  # Let pip figure out CUDA version

try:
    # Verify installations
    import talib
    import cupy as cp
    import numba
    import alpaca_trade_api
    print("\nVerifying CUDA setup...")
    print(f"CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"Numba CUDA available: {numba.cuda.is_available()}")
    print(f"Alpaca Trade API version: {alpaca_trade_api.__version__}")
    print("\nAll packages installed and verified successfully!")
except Exception as e:
    print(f"Error during verification: {str(e)}")
    sys.exit(1)

# Optional: Restart runtime recommendation
print("\nNote: If this is your first time installing these packages,")
print("you may want to restart the runtime to ensure everything is properly initialized.")
print("Runtime → Restart runtime")
```

Changes made:
1. Uninstall potentially conflicting packages first
2. Install dependencies in correct order with version constraints
3. Added version requirements for key packages
4. Added Alpaca API version verification
5. Improved error handling and feedback

Just copy everything between the triple backticks into a single code cell at the start of your notebook.
