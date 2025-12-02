# TandemDegradationModel

This repository contains a compact Python simulation for the degradation of 2‑terminal perovskite/silicon tandem solar modules. The script is designed to run end‑to‑end in Google Colab using only `numpy` and `matplotlib` plus the Python standard library.

## Running locally

1. Install the lightweight dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install numpy matplotlib
   ```
   If your environment is behind a restricted proxy (as in some sandboxes), see the proxy/offline tips below.

2. Execute the simulation and generate plots:
   ```bash
   python tandem_degradation.py
   ```

The script will produce JV curves, lifetime metrics, and energy‑yield trends for the example scenarios.

## Working behind a proxy or offline

If direct `pip install` fails because outbound internet is blocked, you have a few options:

1. **Use pip's proxy support** (when an HTTP/HTTPS proxy is available):
   ```bash
   # Example proxy URL; replace with your organization's proxy
   export http_proxy=http://proxy.example.com:8080
   export https_proxy=$http_proxy
   python -m pip install --proxy "$http_proxy" numpy matplotlib
   ```
   You can also persist the proxy in pip's config:
   ```bash
   python -m pip config set global.proxy http://proxy.example.com:8080
   ```

2. **Install from pre-downloaded wheels** (completely offline):
   * On a machine with internet, download wheels for your Python version:
     ```bash
     mkdir wheels
     python -m pip download --only-binary :all: --dest wheels numpy matplotlib
     ```
   * Copy the `wheels/` folder to the offline machine and install without contacting PyPI:
     ```bash
     python -m pip install --no-index --find-links wheels/ numpy matplotlib
     ```

3. **Use Google Colab**: open `tandem_degradation.py` in Colab (which already has `numpy` and `matplotlib`) and run it directly.
