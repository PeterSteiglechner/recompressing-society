#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

#%%
# ------------------------------------
# -----   Simulation (vectorized) -----
# ------------------------------------
def make_solutions(seed, P, N):
    """P x N array of 0/1; every row has exactly N/2 ones."""
    assert N % 2 == 0
    rng = np.random.default_rng(seed)
    base = np.tile(np.repeat([0, 1], N // 2), (P, 1))
    return rng.permuted(base, axis=1)

def errors_for_all_D(seed, P, N):
    """
    Returns array e of length P, where e[D-1] = mean errors per agent, summed over
    the P-D unsolved problems, when the first D problems serve as identities.
    Independent of b, c and the storage cost, so it only needs simulating once per seed.
    """
    S = make_solutions(seed, P, N).astype(float)
    # M[p, j] = number of agents on whom problem p and identity j disagree
    M = N - 2 * (S @ S.T)
    # best[p, d-1] = fewest errors on problem p using the best of identities 0..d-1
    best = np.minimum.accumulate(M, axis=1)
    # only problems p >= D count (problem p < D is an identity, solved perfectly)
    rows = np.arange(P)[:, None]
    cols = np.arange(P)[None, :]
    mask = rows > cols                    # column index d-1 = D-1  ->  p >= D  <=>  p > D-1
    return (best * mask).sum(axis=0) / N

def sim_errors(n_runs, Ds, P, N):
    """(n_runs, len(Ds)) array of mean errors per agent."""
    Ds = np.asarray(Ds)
    all_e = np.stack([errors_for_all_D(s, P, N) for s in range(n_runs)])
    return all_e[:, Ds - 1]

def run_grid(grid_b, grid_c, Ds, P, N, n_runs, costFunc, C0, C1, C_name):
    """Utility = b*P - (b+c)*errors - C(D); the b,c grid is pure broadcasting."""
    Ds = np.asarray(Ds)
    err = sim_errors(n_runs, Ds, P, N)                      # runs x D
    bb, cc, DD = np.meshgrid(grid_b, grid_c, Ds, indexing="ij")
    e_mean = err.mean(0)[None, None, :]
    e_std = err.std(0)[None, None, :]
    u_mean = bb * P - (bb + cc) * e_mean - costFunc(DD, C0=C0, C1=C1)
    u_std = (bb + cc) * e_std                               # std across runs
    return pd.DataFrame({
        "N": N, "P": P, "D": DD.ravel(), "c": cc.ravel(), "b": bb.ravel(),
        "C0": C0, "C1": C1, "C(D)": C_name,
        "u_mean": u_mean.ravel(), "u_std": u_std.ravel(),
    })

def run_grid_C0(grid_b, c, grid_C0, Ds, P, N, n_runs, costFunc, C1, C_name):
    """Like run_grid, but the second axis is the cost parameter C1 and c is fixed."""
    Ds = np.asarray(Ds)
    err = sim_errors(n_runs, Ds, P, N)                       # runs x D
    bb, c0, DD = np.meshgrid(grid_b, grid_C0, Ds, indexing="ij")
    e_mean = err.mean(0)[None, None, :]
    e_std = err.std(0)[None, None, :]
    u_mean = bb * P - (bb + c) * e_mean - costFunc(DD, C0=c0, C1=C1)
    u_std = (bb + c) * e_std
    return pd.DataFrame({
        "N": N, "P": P, "D": DD.ravel(), "c": c, "b": bb.ravel(),
        "C0": c0.ravel(), "C1": C1, "C(D)": C_name,
        "u_mean": u_mean.ravel(), "u_std": u_std.ravel(),
    })

def get_Dstar(df, params):
    """Optimal D for every combination of `params` (any number). Ties -> smallest D."""
    df = df.sort_values("D")
    idx = df.groupby(params)["u_mean"].idxmax()
    other = [p for p in ["N", "P", "C(D)", "C0", "C1", "b", "c"] if p not in params]
    out = df.loc[idx, params + other + ["D", "u_mean"]]
    return out.rename(columns={"D": "Dstar", "u_mean": "maxU"}).reset_index(drop=True)

#%%
# ------------------------------------
# -----   Reference check (slow)  -----
# ------------------------------------
def play_one_round_ref(seed, D, P, N, b, c, costFunc, C0, C1):
    """Your loop version (with the all-P fix), using the same solutions as the fast code."""
    solutions = make_solutions(seed, P, N)
    identities = solutions[:D]
    best_ids = [np.argmin(np.abs(identities - solutions[p]).sum(axis=1)) for p in range(P)]
    util = np.array([(solutions[p] == identities[best_ids[p]]) * b
                     - (solutions[p] != identities[best_ids[p]]) * c for p in range(P)])
    return util.sum(axis=0) - costFunc(D, C0=C0, C1=C1)

def check(P=20, N=40, b=1.3, c=0.7, C0=1.0, seed=3):
    costFunc = lambda D, C0, C1: C0 * D
    for D in [1, 5, 20]:
        slow = play_one_round_ref(seed, D, P, N, b, c, costFunc, C0, np.nan).mean()
        e = errors_for_all_D(seed, P, N)[D - 1]
        fast = b * P - (b + c) * e - costFunc(D, C0, np.nan)
        assert np.isclose(slow, fast), (D, slow, fast)
    print("fast and reference versions agree")

#%%
# ------------------------------------
# -----   Plotting  -------------------
# ------------------------------------
ALL_PARAMS = ["N", "P", "C(D)", "C0", "C1", "b", "c"]

def _select(df, p):
    mask = np.ones(len(df), dtype=bool)
    for k, v in p.items():
        mask &= np.isclose(df[k].astype(float), v)
    return df[mask]

def _fmt(k, v):
    return f"{k}={v}" if isinstance(v, str) else (f"{k}={v}" if not np.isnan(v) and int(v)==v else  f"{k}={v:.3f}")

def plot_UoverD(df, df_Dstar, params, cmap="viridis", inset_loc=(0.60, 0.55, 0.36, 0.38)):
    """
    inset_loc: (x0, y0, width, height) in axes fractions. Move it if it covers curves,
               e.g. (0.12, 0.08, 0.36, 0.38) for the lower left.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16/2.54, 9/2.54))
    other_params = [q for q in ALL_PARAMS if q not in params[0]]

    # which parameter varies across the curves? (that one gets the colormap and the inset x-axis)
    varying = [k for k in params[0] if len({p[k] for p in params}) > 1]
    fixed = [k for k in params[0] if k not in varying]
    key = varying[0]
    vals = np.array([p[key] for p in params], dtype=float)

    norm = mpl.colors.Normalize(vals.min(), vals.max())
    cm = mpl.colormaps[cmap]
    color_of = lambda v: cm(0.05 + 0.9 * norm(v))

    # --- main plot ---
    dstar = []
    for p in params:
        curr_df = _select(df, p)
        assert len(curr_df) > 0 and curr_df["D"].is_unique
        curr_opt = _select(df_Dstar, p)
        col = color_of(p[key])
        ax.plot(curr_df["D"], curr_df["u_mean"], color=col, lw=1, alpha=0.5)
        ax.scatter(curr_opt["Dstar"], curr_opt["maxU"], marker="s", s=22,
                   color=col, edgecolor="k", linewidth=0.6, zorder=3)
        dstar.append(curr_opt["Dstar"].iloc[0])
    dstar = np.array(dstar)

    sm = mpl.cm.ScalarMappable(norm=norm,
                               cmap=mpl.colors.ListedColormap(cm(np.linspace(0.05, 0.95, 256))))
    fig.colorbar(sm, ax=ax, label=key, pad=0.02)

    ax.set_xlabel("D")
    ax.set_ylabel("mean utility")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

    first = curr_df.iloc[0]
    txt = [f"{k}={params[0][k]:.1f}" for k in fixed] + [_fmt(k, first[k]) for k in other_params]
    ax.text(0.02, 0.02, ";\n".join(txt), va="bottom", ha="left",
            transform=ax.transAxes, color="0.3", fontsize=7)

    # --- inset: D* over the varying parameter ---
    axin = ax.inset_axes(list(inset_loc))
    ax.add_patch(mpl.patches.FancyBboxPatch((inset_loc[0]-0.07, inset_loc[1]-0.12), inset_loc[2]+0.09, inset_loc[3]+0.17, boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes, fc="gainsboro", ec="0.7", alpha=0.7, zorder=2.5))
    order = np.argsort(vals)
    axin.plot(vals[order], dstar[order], color="0.6", lw=0.8, zorder=1)
    axin.scatter(vals, dstar, c=[color_of(v) for v in vals], s=14,
                 edgecolor="k", linewidth=0.4, zorder=2)
    axin.set_xlabel(key, fontsize=6, labelpad=1)
    axin.set_ylabel(r"$D^*$", fontsize=6, labelpad=1)
    axin.tick_params(labelsize=6, length=2, pad=1)
    axin.set_facecolor((1, 1, 1, 0.85))
    axin.grid(alpha=0.25, lw=0.4)
    axin.spines[["top", "right"]].set_visible(False)
    return fig, ax

def plot_heat(df, df_Dstar, params):
    fig, ax = plt.subplots(1, 1, figsize=(14/2.54, 12/2.54))
    other_params = [q for q in ALL_PARAMS if q not in params]
    first = df.iloc[0]
    sns.heatmap(df_Dstar.pivot_table(index=params[1], columns=params[0], values="Dstar"),
                vmin=1, vmax=df["D"].max(), cmap="hot", cbar=True, ax=ax,
                cbar_kws=dict(label=r"optimal nr of identities $D^*$"))
    ax.set_xticklabels([f"{float(t.get_text()):.1f}" for t in ax.get_xticklabels()])
    ax.set_yticklabels([f"{float(t.get_text()):.1f}" for t in ax.get_yticklabels()], rotation=0)
    ax.invert_yaxis()
    ax.text(0.98, 0.02, ";\n".join(_fmt(k, first[k]) for k in other_params),
            va="bottom", ha="right", transform=ax.transAxes, color="blue")
    ax.set_aspect("equal")
    return fig, ax 
#%%
# ------------------------------------
# -----   Run  ------------------------
# ------------------------------------
check()

P, N = 50, 100
C0, C1 = 1, np.nan
costFunc = lambda D, C0, C1: C0 * D
C_name = "C0 * D"

grid = np.round(np.arange(0, 2.01, 0.1), 10)   # rounding keeps float comparisons safe
Ds = np.arange(1, P + 1)                       # includes D = P
df_linear = run_grid(grid, grid, Ds, P, N, n_runs=1000,
                        costFunc=costFunc, C0=C0, C1=C1, C_name=C_name)
df_Dstar_linear = get_Dstar(df_linear, ["b", "c"])

#%%
os.makedirs("figs", exist_ok=True)
c = 1.0
plot_UoverD(df_linear, df_Dstar_linear, [dict(b=b, c=c) for b in grid], inset_loc=(0.6,0.1,0.36,0.38))
plt.tight_layout()
plt.savefig(f"figs/example_linear_UoverDCurves_c{c}.png", dpi=300)

plot_heat(df_linear, df_Dstar_linear, ["b", "c"])
plt.tight_layout()
plt.savefig("figs/linear_Dstar_overStoreCost_X_Benefit.png", dpi=300)

#%%


# ------------------------------------
# -----   Run 2 vs C0; Convex vs Concave Storage Cost  -------
# ------------------------------------
P = 50
N = 100
c=0.5
costFunc_exp = lambda D, C0, C1=0.035: C0 * np.exp(C1 * D)
costFunc_log = lambda D, C0, C1=1: C0 * np.log2(C1 * D)
for cfunc in["exp", "log"]:
    C1 = 0.035 if cfunc == "exp" else 1
    grid_b  = np.round(np.arange(0, 10.01, 1), 3)
    grid_C0 = np.round(np.arange(1, 50, 5), 3)
    Ds = np.arange(1, P + 1)
    df_explog = run_grid_C0(grid_b, c, grid_C0, Ds, P, N, n_runs=1000,
                        costFunc=costFunc_exp if cfunc=="exp" else costFunc_log, C1=C1, C_name=f"C0*{cfunc}(C1*D)")
    df_Dstar_explog = get_Dstar(df_explog, ["b", "C0"])

    C0plot = grid_C0[5]
    plot_UoverD(df_explog, df_Dstar_explog, [dict(b=b, C0=C0plot) for b in grid_b], inset_loc=((0.6 if cfunc=="exp" else 0.4),0.1,0.36,0.38))
    plt.tight_layout()
    plt.savefig(f"figs/example_{cfunc}_UoverDCurves_C0{C0}.png", dpi=300)

    fig, ax = plot_heat(df_explog, df_Dstar_explog, ["b", "C0"])
    plt.tight_layout()
    fig.autofmt_xdate(rotation=30)
    plt.savefig(f"figs/{cfunc}_Dstar_overStoreCost_X_Benefit.png", dpi=300)

#%%
