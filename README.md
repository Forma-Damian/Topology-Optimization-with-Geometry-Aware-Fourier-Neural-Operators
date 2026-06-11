## Topology Optimization with Geometry Aware Fourier Neural Operators

Usual topology optimization requires computationally expensive Finite Element Analysis. This project replaces the FEA approach with a Geometry-Aware Fourier Neural Operator (Geo-FNO). By freezing the Geo-FNO and making the input as a learnable parameter, backpropagation will be used directly from physical loss.

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
src/fea.py           # density generation, solver, plotting, save
notebooks/demo.ipynb # progress demo
data/samples/        # generated training pairs (local, gitignored)
requirements.txt
```
