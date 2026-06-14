# Recompressing Society — ABM

Simple agent-based model exploring how many identities (`D`) individuals should store to coordinate on `P` population-partition problems.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd compression-abm
uv sync
```

## Usage

- Option 1: 
Open `model.py` e.g. in VS code or as a jupyter notebook. You can run individual code chunks (separated by ```#%%```) or the whole file. 
- Option 2: Run
```bash
uv run model.py 
```

## Model sketch

- `N` individuals each store `D` identities (binary assignments across the population)
- For each of `P` problems, half the population should take action `0`, the other half `1`.
- The first `D` problems are solved perfectly via stored identities; remaining `P−D` problems use the best-matching identity
- Utility = benefits from correct actions − costs from errors − identity storage cost `C(D)`
- Simulations sweep over `b` (benefit) and `c` (error cost), and  `b` (benefit) and `C0` (storage cost param) to find the optimal `D*` and report how `D*` changes as the parameters change.
We report this analysis for different functional forms of `C(D)` (convex, linear, concave) see below.

## Cost functions

| Name | Formula |
|------|---------|
| Linear | `C0 · D` |
| Exponential | `C0 · exp(C1 · D)` |
| Logarithmic | `C0 · log_2(C1 · D)` |