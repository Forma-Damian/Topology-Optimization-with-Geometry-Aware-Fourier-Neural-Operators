## Topology Optimization with Geometry Aware Fourier Neural Operators

Usual topology optimization requires computationally expensive Finite Element Analysis. This project replaces the FEA approach with a Geometry-Aware Fourier Neural Operator (Geo-FNO). By freezing the Geo-FNO and making the input as a learnable parameter, backpropagation will be used directly from physical loss.

The main notebook is [`notebooks/final3.ipynb`](notebooks/final3.ipynb). It runs on a 64x64 grid: FEA generates training data, a U-Net predicts strain energy, then Neuro-OC optimizes density for ten test cases.

### Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Project layout

```txt
notebooks/final3.ipynb               # main report + full pipeline
notebooks/demo.ipynb                 # early progress demo
notebooks/gfno_vs_fea_example.ipynb  # extended demo (mid-term presentation)
src/fea.py                           # small standalone FEA prototype
data/samples/                        # optional local .npz output (gitignored)
results/                             # exported figures from optimization runs
requirements.txt
```
