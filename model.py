# ---- 
# Recompressing Society
# model: a simple model how N individuals with D identities aim to solve P>D population partition problems.
# Setup: 
#   - We have N individuals
#   - The individuals solve P problems, where N/2 individuals should do action 0, the other N/2 action 1
#   - Making an error reduces utility by c. Doing the correct action increases utility by b 
#   - Individuals store D identities. Storing reduces utility.
#   - Individuals then solve the problems one by one:
#       - The individuals agree on one identity dimension to sort themselves in action 0/action 1 individuals.
#       - For the  first D problems this is exaclty the corresponding identity (these problems contribute with utility b per person)
#       - For the Dth to Pth problem, they need to find some identity dimension that 
#       - We assume that they choose the identity that minimises the collective error     
#    - We track the average utilty dependent on the number of identities individuals store.


#%%
import numpy as np
import matplotlib.pyplot as plt 
import inspect
import pandas as pd
import seaborn as sns
#%%
def play_one_round(seed, D, P, N, b, c, costFunc, C0, C1):
    assert(int(N/2) == N/2)
    np.random.seed(seed)

    # true solutions for D solutions
    solutions = np.array([np.random.choice([0] * int(N/2) + [1] * int(N/2), size=N, replace=False) for p in range(P)])
    
    identities = solutions[:D,:]

    # best identity for each of the P-D problem (the one that minimises error rate)
    best_ids = [np.argmin(np.abs(identities - solutions[p,:]).sum(axis=1)) for p in range(P)]
    
    # utilities of the N agents for P-D problems (benefits for correct + costs for incorrect)
    utilities_problems = np.array([
        (
            (solutions[p] == identities[best_ids[p]]) * b - 
            (solutions[p] != identities[best_ids[p]]) * c         )
        for p in range(P)
    ])

    # add utility benefit for D correct solutions (b * D for a single agent)
    # add utility cost for holding D identities (C(D) for a single agent)
    utilities = utilities_problems.sum(axis=0) -  costFunc(D, C0=C0, C1=C1)
    return utilities

def sim(n_runs, D, P, N, b, c, costFunc, C0, C1):
    utilities = []
    for s in range(n_runs):
        utilities.append(play_one_round(s, D, P, N, b, c, costFunc, C0, C1))
    return np.array(utilities)


def get_Dstar(df, params):
    """return the optimal D to fix"""
    results_Dstar = []
    other_params = [p for p in ["N","P","C(D)", "C0", "C1", "b", "c"] if p not in params]
    param_combis = df[params].value_counts().index.to_list()
    for param1, param2 in param_combis:
        curr_df = df.query(f"{params[0]}=={param1} and {params[1]}=={param2}")
        # get max utility
        maxU = curr_df["u_mean"].max()
        # get D for which we get max utility
        Dstar = curr_df.loc[curr_df["u_mean"]==maxU, "D"].values[0]
        other_params_values = [curr_df[c].unique()[0] for c in other_params]
        results_Dstar.append([param1, param2]+other_params_values+[Dstar, maxU])
    df_Dstar = pd.DataFrame(results_Dstar, columns=params+other_params+["Dstar", "maxU"] )
    return df_Dstar


# %%

# ------------------------------------
# -----   Plotting functions  -------
# ------------------------------------


# %%
def plot_UoverD(df, df_Dstar, params):
    fig, ax = plt.subplots(1,1, figsize=(16/2.54, 9/2.54))
    other_params = [p for p in ["N","P","C(D)", "C0", "C1", "b", "c"] if p not in params[0]]
    for p in params:
        curr_df = df.query(" and ".join([f"{n}=={v}" for n,v in p.items()]))
        assert(curr_df["D"].value_counts().unique()[0]==1)
        curr_optimum = df_Dstar.query(" and ".join([f"{n}=={v}" for n,v in p.items()]))
        # curr_df["u_mean_norm"] = curr_df["u_mean"] / curr_df["u_mean"].abs().max()
        curr_df.plot(x="D", y="u_mean", ax=ax, label=", ".join([f"{k}={v:.1f}" for k,v in p.items()]), legend=True)
        ax.scatter(curr_optimum["Dstar"], curr_optimum["maxU"], marker="s", )
    ax.set_ylabel("mean utiltiy")
    other_params_values = [curr_df[c].unique()[0] for c in other_params]
    ax.text(0.98, 0.02, ";\n".join([f"{k}={v}" if type(v)==str else f"{k}={v:.1f}" for k,v in dict(zip(other_params, other_params_values)).items()]), va="bottom", ha="right", transform=ax.transAxes, color="blue")
    ax.legend(ncol=3, fontsize=6)
    return 

def plot_heat(df, df_Dstar, params, ):
    fig, ax = plt.subplots(1,1, figsize=(14/2.54, 12/2.54))
    other_params = [p for p in ["N","P","C(D)", "C0", "C1", "b", "c"] if p not in params]
    other_params_values = [df[c].unique()[0] for c in other_params]
    sns.heatmap(df_Dstar.pivot_table(index=params[1], columns=params[0], values="Dstar"), vmin=1, vmax=df["D"].max(), cmap="hot", cbar=True, ax=ax, cbar_kws=dict(label=r"optimal nr of identities $D^*$"))
    ax.set_xticklabels([f"{float(t.get_text()):.1f}" for t in ax.get_xticklabels()])
    ax.set_yticklabels([f"{float(t.get_text()):.1f}" for t in ax.get_yticklabels()], rotation=0)
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    ax.text(0.98, 0.02, ";\n".join([f"{k}={v}" if type(v)==str else f"{k}={v:.1f}" for k,v in dict(zip(other_params, other_params_values)).items()]), va="bottom", ha="right", transform=ax.transAxes, color="blue")
    ax.set_aspect("equal")
    return


#%%
# ------------------------------------
# -----   TEST b vs c; Linear Storage Cost  -------
# ------------------------------------
P = 50
N = 100
# b = 1
# c = 1
C0 = 1
C1 = np.nan
costFunc_linear = lambda D, C0, C1: C0 * D
costFunc = costFunc_linear
C_name = inspect.getsource(costFunc).split(":")[1].split("\n")[0]

grid = np.arange(0,2.01,step=0.1)
bc_combis = [(b,c) for b in grid for c in grid]
n_runs = 10
res = []
for b,c in bc_combis:
    if c==grid[0]: print(b)
    for D in np.arange(1,P+1,2):
        u = sim(n_runs, D, P, N, b, c, costFunc, C0, C1)
        res.append([N, P, D, c, b, C0, C1, C_name, np.mean(u, axis=(0,1)), np.std(np.mean(u, axis=0))])

df_linear = pd.DataFrame(res, columns=["N", "P", "D", "c", "b", "C0", "C1", "C(D)", "u_mean","u_std"])
df_Dstar_linear = get_Dstar(df_linear, ["b","c"])

#%%
params = [dict(b=b, c=1.) for b in grid]
plot_UoverD(df_linear, df_Dstar_linear, params)
plt.tight_layout()
plt.savefig("figs/example_linear_UoverDCurves.png", dpi=300)
# %%
plot_heat(df_linear, df_Dstar_linear, ["b", "c"])
plt.tight_layout()
plt.savefig("figs/linear_Dstar_overStoreCost_X_Benefit.png", dpi=300)
# %%

# ------------------------------------
# -----   TEST b vs C0; Convex vs Concave Storage Cost  -------
# ------------------------------------
P = 50
N = 100
# b = 1
c = 1
# C0 = 1
costFunc_exp = lambda D, C0, C1=0.035: C0 * np.exp(C1 * D)
costFunc_log = lambda D, C0, C1=1: C0 * np.log2(C1 * D)

b_grid = np.arange(0,10,step=1)
C0_grid = np.arange(0,50,step=5)
bC0_combis = [(b, C0) for b in b_grid for C0 in C0_grid]
n_runs = 1

for costFunc, C1 in zip([costFunc_exp, costFunc_log], [0.035, 1]):
    res = []
    for b, C0 in bC0_combis:
        if C0==C0_grid[0]: print(b)
        C_name = inspect.getsource(costFunc).split(":")[1].split("\n")[0]
        for D in np.arange(1,P,2):
            u = sim(n_runs, D, P, N, b, c, costFunc, C0, C1)
            res.append([N, P, D, c, b, C0, C1, C_name, np.mean(u), np.std(u)])

    df = pd.DataFrame(res, columns=["N", "P", "D", "c", "b", "C0", "C1", "C(D)", "u_mean","u_std"])
    df_Dstar = get_Dstar(df, ["b","C0"])

    if costFunc==costFunc_exp:
        df_exp = df.copy()
        df_Dstar_exp = df_Dstar.copy()
    else:
        df_log = df.copy()
        df_Dstar_log = df_Dstar.copy()

# %%
params = [dict(b=2., C0=C0) for C0 in C0_grid]
plot_UoverD(df_exp, df_Dstar_exp, params)
plt.tight_layout()
plt.savefig("figs/example_exp_UoverDCurves.png", dpi=300)

# %%
plot_heat(df_exp, df_Dstar_exp, ["C0", "b"])
plt.tight_layout()
plt.savefig("figs/exp_Dstar_overStoreCost_X_Benefit.png", dpi=300)
# %%
params = [dict(b=2., C0=C0) for C0 in C0_grid]
plot_UoverD(df_log, df_Dstar_log, params)
plt.tight_layout()
plt.savefig("figs/example_log_UoverDCurves.png", dpi=300)

# %%
plot_heat(df_log, df_Dstar_log, ["C0", "b"])
plt.tight_layout()
plt.savefig("figs/log_Dstar_overStoreCost_X_Benefit.png", dpi=300)
# %%
