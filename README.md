
---

## 🧠 Key Formulas

- **Conditional expectation form**
  \[
  \mathrm{VaRC}_{i,\alpha} = 
  \frac{\mathbb{E}[L_i \, \delta(L-a)]}{\mathbb{E}[\delta(L-a)]}, \quad a=\mathrm{VaR}_\alpha(L)
  \]

- **Integration-by-parts decomposition**
  \[
  \mathbb{E}[L_i\delta(L-a)]
  = -[u_i e_i f_{\varepsilon_i|Z^L}(e_i|z^L)F_{L_{-i}|Z}(a-u_i e_i|z)]_0^{\min(1,a/u_i)}
    + \int u_i F_{L_{-i}|Z}(a-u_i e_i|z)\{f + e_i f'\}de_i
  \]

- **SPA-based density**
  \[
  f_L(a) \approx 
  \frac{\exp(K_L(t̂)-t̂a)}{\sqrt{2\pi K_L''(t̂)}}
  \left[1+\frac{1}{8}\bigl(\lambda_4 - \tfrac{5}{3}\lambda_3^2\bigr)\right]
  \]

---

## 🚀 Numerical Results

- **Portfolio:** 10 obligors, \(p_i = 0.01, …, 0.10\); \( \alpha=2, \beta=5, \tau=0.5, \rho_i=\sqrt{0.5}\)
- **Platform:** MacBook Pro (M3 Pro, 18 GB RAM)
- **Findings:**
  - **SAFA:** ≈ 7–10× faster than CE + RQMC  
  - **SASPA:** balances accuracy and speed, ideal for partial VaRC  
  - Standard errors an order of magnitude smaller than benchmark MC

Example (α = 0.95):

| Method | Avg S.E. | CPU (s) |
|---------|-----------|---------|
| PMC | 0.0007 | 587 |
| CE + RQMC | 0.0005 | 689 |
| **SAFA** | **0.0002** | **81** |
| **SASPA** | 0.0012 | 194 |

---

## 🧾 Dependencies

- Python ≥ 3.9  
- NumPy ≥ 1.24  
- SciPy ≥ 1.11  
- tqdm ≥ 4.65  
- matplotlib (optional for plots)

---

## ▶️ Quick Start

```bash
git clone https://github.com/hithisisjayden/VaRC-algorithms-for-Reviewers-check.git
cd VaRC-algorithms-for-Reviewers-check

# Run SAFA example
python SAFA_main.py

# Run SASPA example
python SASPA_main.py

# Reproduce numerical tables
python experiments/varc_experiments.py
