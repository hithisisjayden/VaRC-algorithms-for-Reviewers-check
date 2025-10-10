# Simulation-Analytical Approach for Calculating VaR Contributions in Credit Portfolios
## Kep points
- Benchmark methods: `PMC.py` and `CE_RQMC.py`.
- Simulation-analytical algorithms: `SAFA.py` and `SASPA.py`.
## Files
- `PMC.py`: Plain Monte Carlo method.
- `CE_RQMC.py`: Iterative CE method enhanced with RQMC.
- `SAFA.py`: Simulation-Analytical Full Allocation algorithm, which estimates the denominator $\mathbb{E}[\delta(L-a)]$ via the Euler allocation principle.
- `SASPA.py`: Simulation-Analytical Saddlepoint Approximation algorithm, which estimates the denominator $f_L(a)$ via SPA.
- `compute_loss_threshold.py`: Calculate the loss threshold or the desired loss level, which is involved in `CE_RQMC.py`.
- `VaRC results table.tex`: The numerical results for our simulation-analytical algorithms and benchmark methods.
