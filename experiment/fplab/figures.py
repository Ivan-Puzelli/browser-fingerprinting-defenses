# SPDX-License-Identifier: MIT
"""Shared scientific figures: identical PNG previews and vector PDF exports."""
import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from .data import ARMS, COLOR, LABEL, root
from . import canvas as canvas_mod
from . import entropy as entropy_mod
from . import stability as stability_mod


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.has_trait("kernel")
    except ImportError:
        return False


if not _in_notebook():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paper figures are designed at column width; diagnostic plots stay larger.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42})


def figdir():
    d = root() / "figures"
    d.mkdir(exist_ok=True)
    return d


def _save(fig, name, show=True, *, tight=True):
    path = figdir() / name
    if tight:
        fig.tight_layout(pad=.6)
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    if show and _in_notebook():
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    print(f"saved figures/{name} and vector PDF")
    return path


def _clean(ax, axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color="#e4e4e4", linewidth=.5)
    ax.set_axisbelow(True)


def stability_by_level(t_lv):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    cols = ["L2", "L3", "L4 same-fp"]
    x, w = np.arange(3), .13
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i-2.5)*w, t_lv.loc[arm, cols], w,
               label=LABEL[arm], color=COLOR[arm])
    ax.set(xticks=x, xticklabels=["L2: reloads", "L3: fresh sessions", "L4: two origins"],
           ylabel="visitorId stability / agreement", ylim=(0, 1.05),
           title="Exact library identifier, not raw-canvas stability")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.28), frameon=False)
    _clean(ax, "y")
    return _save(fig, "stability_by_level.png")


def cross_domain_similarity(xd):
    fig, ax = plt.subplots(figsize=(6.6, 3))
    arms = list(xd)
    vals = [xd[a]["component_agreement"] for a in arms]
    ax.barh(range(len(arms)), vals, color=[COLOR[a] for a in arms])
    for i, (a, v) in enumerate(zip(arms, vals)):
        ax.text(v+.008, i, f"{v:.3f} ({len(xd[a]['differing'])} differ)", va="center", fontsize=7)
    ax.set(yticks=range(len(arms)), yticklabels=[LABEL[a] for a in arms], xlim=(0, 1.23),
           xticks=np.arange(0, 1.01, .2), xlabel="Mean component agreement (raw canvas included)",
           title="L4: paired origins on one host")
    ax.invert_yaxis()
    _clean(ax)
    return _save(fig, "cross_domain_similarity.png")


def canvas_mechanism(t_mech):
    arms = list(t_mech.index)
    fig, axes = plt.subplots(1, 2, figsize=(3.39, 2.65), sharey=True)
    for ax, field, label, xmax in zip(axes, ["pct_changed", "max_delta"],
                                    ["Pixels changed (%)", "Max. channel delta"], [119, 292]):
        vals = t_mech[field]
        ax.barh(range(len(arms)), vals, color=[COLOR[a] for a in arms], height=.6)
        for i, v in enumerate(vals):
            ax.text(v+xmax*.02, i, f"{v:.2f}" if 0 < v < 10 and field == "pct_changed"
                    else f"{v:.0f}", va="center", fontsize=6.5)
        ax.set(xlabel=label, xlim=(0, xmax))
        ax.set_xticks([0, 50, 100] if field == "pct_changed" else [0, 125, 250])
        _clean(ax)
    axes[0].set_yticks(range(len(arms)), [LABEL[a] for a in arms])
    axes[0].invert_yaxis()
    return _save(fig, "canvas_mechanism.png")


def component_heatmap(S):
    """Compare each component with its engine baseline, in two labeled groups."""
    if set(S.columns) != set(ARMS):
        raise ValueError("Heatmap requires all six browser arms")
    S = S.loc[:, ARMS]
    colors = ['#df9a91', '#93b4d0', '#8bc0a3', '#e5e7eb']
    fig = plt.figure(figsize=(6.6, 6.15))
    ax = fig.add_axes([.25, .195, .725, .645])
    fig.text(.5, .977, 'Browser defenses by component', fontsize=10.5,
             ha='center', va='top')
    labels = ['1  Matches engine baseline', '2  Different, stable',
              '3  Varies across sessions', '4  Unavailable']
    # Matplotlib fills a two-column legend by column, so reorder for row reading.
    order = [0, 2, 1, 3]
    fig.legend(handles=[Patch(facecolor=colors[i], edgecolor='none', label=labels[i]) for i in order],
               loc='upper center', bbox_to_anchor=(.5, .945), ncol=2,
               frameon=False, fontsize=8, handlelength=1.4, columnspacing=2,
               labelspacing=.6)
    ax.imshow(S.values, aspect='auto', cmap=ListedColormap(colors), vmin=0, vmax=3,
              interpolation='nearest')
    ax.set_yticks(range(len(S)), S.index, fontsize=8)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.set_xticks([])
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(S), 1), minor=True)
    ax.grid(which='minor', color='white', alpha=.38, linewidth=.55)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.axvline(2.5, color='white', linewidth=2)
    for spine in ax.spines.values():
        spine.set_linewidth(.7)
        spine.set_color('#7d8790')
    for i in range(len(S)):
        for j in range(6):
            ax.text(j, i, str(int(S.iloc[i, j])+1), ha='center', va='center', fontsize=7.8)
    groups = [(0, '#a56e15', '#fff5df', 'Chromium engine', 'Baseline: Chrome'),
              (3, '#705498', '#f2ecfa', 'Gecko engine', 'Baseline: Firefox')]
    browser_names = ['Chrome', 'Brave', 'Brave\naggressive', 'Firefox', 'Firefox\nRFP', 'Tor']
    for start, ink, fill, name, baseline in groups:
        x = start / 6
        box = FancyBboxPatch((x+.004, -.252), .492, .227,
                             boxstyle='round,pad=0.004,rounding_size=0.021',
                             transform=ax.transAxes, facecolor=fill, edgecolor=ink,
                             linewidth=1.1, clip_on=False)
        ax.add_patch(box)
        for j in range(start, start+3):
            ax.text((j+.5)/6, -.063, browser_names[j], transform=ax.transAxes,
                    ha='center', va='center', fontsize=8.1, color=ink)
        ax.text(x+.25, -.151, name, transform=ax.transAxes,
                ha='center', va='center', fontsize=8.5, weight='bold', color=ink)
        ax.text(x+.25, -.205, baseline, transform=ax.transAxes,
                ha='center', va='center', fontsize=8.2, color=ink)
    return _save(fig, 'component_heatmap.png', tight=False)

def entropy_budget(table, totals, attrs, weighting="amiu"):
    fig, ax = plt.subplots(figsize=(6.6, 3))
    vals = [totals[a][0] for a in ARMS]
    ax.barh(range(len(ARMS)), vals, color=[COLOR[a] for a in ARMS])
    for i, v in enumerate(vals):
        ax.text(v+.5, i, f"{v:.2f}", va="center")
    ax.set(yticks=range(len(ARMS)), yticklabels=[LABEL[a] for a in ARMS],
           xlabel=f"Residual weighted score (bits; {entropy_mod.STUDY_NAME[weighting]})",
           xlim=(0, max(vals)*1.16), title="Residual score under the default classification")
    ax.invert_yaxis()
    _clean(ax)
    return _save(fig, "entropy_budget.png")


def classification_sensitivity(sensitivity):
    # Only the arms a case actually charges. Brave holds neither credit, so its
    # four points are identical by construction and would read as robustness.
    arms = ["firefox-rfp", "tor-standard"]
    fig, ax = plt.subplots(figsize=(3.39, 2.6))
    for j, a in enumerate(arms):
        y = np.arange(len(sensitivity)) + (j - (len(arms)-1)/2)*.23
        vals = sensitivity[a].to_numpy()
        ax.scatter(vals, y, s=22, color=COLOR[a], marker=["s", "D"][j], label=LABEL[a])
        for v, yi in zip(vals, y):
            ax.text(v+1.5, yi, f"{v:.0f}", va="center", fontsize=6)
    ax.set(yticks=np.arange(len(sensitivity)), yticklabels=sensitivity.index,
           xlabel="Reduction in weighted score (%)", xlim=(0, 104), ylim=(-.6, 3.6))
    ax.invert_yaxis()
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.23),
              frameon=False, handletextpad=.2, columnspacing=.7, fontsize=6.5)
    _clean(ax)
    return _save(fig, "classification_sensitivity.png")


def cross_browser_linkability(M):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(M.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], rotation=40, ha="right")
    ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
    for i in range(len(ARMS)):
        for j in range(len(ARMS)):
            ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center",
                    color="white" if M.values[i,j] < .6 else "black")
    fig.colorbar(im, ax=ax, shrink=.8, label="Component agreement")
    ax.set_title("Selected non-farbled components on one host\nAgreement is not a tracking probability")
    return _save(fig, "cross_browser_linkability.png")


def anonymity_sets(cap, deltas=(1.0, 2.0, 2.5), *, rfp_population=1_000_000):
    """Paper-layout scenarios, not measured anonymity sets.

    RFP population is an explicit user-selected assumption. Tor uses the paper\'s
    separate 12–15 hardware-group scenario, rounded to 150,000 at N=2 million.
    The input capacity table and the global population defaults are unchanged.
    """
    if tuple(deltas) != (1.0, 2.0, 2.5):
        raise ValueError("This figure uses delta scenarios 1, 2 and 2.5")
    if not np.isfinite(rfp_population) or rfp_population <= 0:
        raise ValueError("RFP assumed population must be finite and positive")
    cap = {arm: dict(c, k=dict(c["k"])) for arm, c in cap.items()}
    cap["firefox-rfp"]["population"] = rfp_population
    for c in cap.values():
        c["k"] = {d: entropy_mod.anonymity_set(c["bits"], c["population"], d)
                  for d in deltas}
    arms = ['chrome', 'brave', 'firefox', 'firefox-rfp', 'tor-standard']
    fig = plt.figure(figsize=(7.2, 3.7))
    ax = fig.add_axes([.145, .155, .83, .635])
    fig.text(.5, .963, 'Anonymity from nine attributes', fontsize=10.5,
             ha='center', va='top')
    markers = ['o', 'D', 's']
    legend = [Line2D([], [], linestyle='none', marker=m, color='#526172', markersize=5,
                     label=t) for m, t in zip(markers,
                     [r'Independent ($\delta=1$)', r'Correlated ($\delta=2$)',
                      r'More correlated ($\delta=2.5$)'])]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .915),
               ncol=3, frameon=False, fontsize=8, handletextpad=.35, columnspacing=1.5)
    ax.set_xscale('log')
    ax.set_xlim(1e-7, 2e6)
    ax.set_ylim(4.55, -.55)
    ax.axvspan(1e-7, 1, color='#f7eeee', zorder=0)
    ax.axvline(1, color='#737b84', lw=1, ls=(0, (3, 3)))
    ax.text(1, 1.035, r'$k=1$', transform=ax.get_xaxis_transform(), ha='center',
            fontsize=8, color='#59616b')
    for i, arm in enumerate(arms[:-1]):
        if cap[arm]['population'] is None:
            ax.text(.5, i, 'Not estimated: population unknown',
                    transform=ax.get_yaxis_transform(), ha='center', va='center',
                    fontsize=8, color=COLOR[arm],
                    bbox=dict(facecolor='white', edgecolor='none', pad=3))
            continue
        values = [cap[arm]['k'][d] for d in deltas]
        color = COLOR[arm]
        ax.plot(values, [i]*3, color=color, alpha=.42, lw=2.2, zorder=2)
        for value, marker in zip(values, markers):
            ax.scatter(value, i, color=color, marker=marker, s=35, zorder=3,
                       edgecolor='white', linewidth=.5)
        # Endpoint label: how many modelled peers remain under delta=2.5.
        end_label = (f'{values[-1]/1000:.3g}k' if values[-1] >= 1000 else f'{values[-1]:.3g}')
        ax.annotate(end_label, (values[-1], i), xytext=(6, 8),
                    textcoords='offset points', color=color, fontsize=8)
    # The current manuscript uses a separate hardware-bucket scenario for Tor,
    # not a delta-deflated marginal score. Keep this assumption explicit.
    tor_n = cap['tor-standard']['population']
    tor_range = (tor_n / 15, tor_n / 12)
    tor_mean = tor_n * .075  # Rounded manuscript scenario; not a measured minimum.
    assert tor_range[0] < tor_mean < tor_range[1]
    ax.plot(tor_range, [4, 4], color=COLOR['tor-standard'], lw=3, zorder=2)
    ax.scatter(tor_mean, 4, marker='*', color=COLOR['tor-standard'], s=95,
               zorder=3, edgecolor='white', linewidth=.5)
    ax.annotate(f'≈{tor_mean:,.0f}', (tor_mean, 4), xytext=(-5, 10), textcoords='offset points',
                fontsize=8, ha='center', color=COLOR['tor-standard'])
    ax.set_yticks(range(5), [LABEL[a] for a in arms])
    ax.text(.02, 3.27, f'RFP: assumed population = {rfp_population / 1e6:g} million',
            transform=ax.get_yaxis_transform(), ha='left', va='center',
            fontsize=7, color=COLOR['firefox-rfp'])
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    ax.set_xlabel(r'Modelled crowd size $k=N/2^{H_{res}/\delta}$ (log scale)', labelpad=9)
    ax.grid(axis='x', color='#e1e4e8', lw=.55, zorder=0)
    ax.tick_params(axis='y', length=0, pad=8)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#9ba1a8')
    return _save(fig, 'anonymity_sets.png', tight=False)

def robustness(rob, ylabel="Reduction (%)\nweighted score / unweighted count"):
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x, w = np.arange(len(rob)), .25
    for j, a in enumerate(["brave", "firefox-rfp", "tor-standard"]):
        vals = rob[a].to_numpy(dtype=float)
        ax.bar(x+(j-1)*w, vals, w, color=COLOR[a], label=LABEL[a])
        for xi, v in zip(x+(j-1)*w, vals):
            ax.text(xi, v+1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, [r.replace(" ", "\n", 1) for r in rob.index])
    ax.set(ylabel=ylabel, ylim=(0, 120), yticks=range(0, 101, 20),
           title="Alternative weights, holding the classification fixed")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    _clean(ax, "y")
    return _save(fig, "robustness.png")


def drop_channels(sweep):
    scenarios = list(sweep)
    # Group identical scores without claiming identical fingerprints.
    grouped = {}
    for a in ARMS:
        sig = tuple(round(sweep[s][a], 8) for s in scenarios)
        grouped.setdefault(sig, []).append(a)
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for j, (sig, arms) in enumerate(grouped.items()):
        ax.scatter(np.arange(len(scenarios))+(j-1.5)*.15, sig, s=30,
                   color=COLOR[arms[0]], marker=["o", "s", "D", "^"][j % 4],
                   label=" / ".join(LABEL[a] for a in arms))
    ax.set_xticks(range(len(scenarios)), [s.replace("-", "drop\n", 1) if s != "none"
                 else "all\nchannels" for s in scenarios])
    ax.set(ylabel="Residual weighted score (bits)", ylim=(0, 55),
           title="Independent channel-removal scenarios (not cumulative)\nEqual scores do not imply equal fingerprints")
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.23))
    _clean(ax, "y")
    return _save(fig, "drop_channels.png")


def similarity_separation(sep):
    groups = [(LABEL[sep["arm"]] + " / " + LABEL[sep["arm"]], sep["within"], "#333333")]
    groups += [(LABEL[sep["arm"]] + " / " + LABEL[o], v, COLOR[o]) for o, v in
               sorted(sep["others"].items(), key=lambda kv: -kv[1].max())]
    fig, ax = plt.subplots(figsize=(3.39, 2.85))
    if sep["window"]:
        lo, hi = sep["window"]
        threshold = (lo+hi)/2
        rounded = round(threshold, 2)
        if lo < rounded <= hi:
            threshold = rounded
        ax.axvspan(lo, hi, color="#55A868", alpha=.15)
        ax.axvline(threshold, color="#397747", ls="--", lw=.9)
        ax.text(threshold-.008, -.7, f"Threshold {threshold:.6g}", fontsize=7,
                ha="right", color="#397747")
    for i, (label, vals, color) in enumerate(groups):
        ax.plot([vals.min(), vals.max()], [i, i], color=color, lw=5, solid_capstyle="butt")
        ax.text(.455, i-.18, f"{label} (n={len(vals):,})", fontsize=6.8, va="bottom")
    ax.set(xlim=(.45, 1), ylim=(len(groups)-.5, -1), yticks=[],
           xlabel=f"Agreement across {sep['n_attributes']} attributes", xticks=[.5,.6,.7,.8,.9,1])
    _clean(ax)
    ax.spines["left"].set_visible(False)
    return _save(fig, "similarity_separation.png")


def canvas_delta_spectrum(mech):
    families = [("Brave: altered channels", ["brave", "brave-aggressive"]),
                ("Gecko defenses: altered channels", ["firefox-rfp", "tor-standard"])]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.5), sharex=True, sharey=True)
    for ax, (title, arms) in zip(axes, families):
        for j, a in enumerate(arms):
            h = mech.get(a, {}).get("hist")
            if not h:
                continue
            ks = np.array(sorted(h))
            pct = 100*np.array([h[k] for k in ks])/sum(h.values())
            ax.scatter(ks, pct, s=13, marker=["o", "x"][j], color=COLOR[a], alpha=.8,
                       label=f"{LABEL[a]}: {sum(h.values()):,} changed channels")
        ax.set(yscale="log", ylabel="Share of changed channels (%)", title=title)
        ax.legend(frameon=False, loc="upper right")
        _clean(ax, "y")
    axes[-1].set(xlabel="Absolute channel delta (only observed magnitudes shown)", xlim=(-4, 259))
    return _save(fig, "canvas_delta_spectrum.png")


def all_figures(rows, weighting="amiu"):
    import pandas as pd
    from . import similarity as sim
    from .linkage import immune_matrix
    stability_by_level(stability_mod.by_level(rows))
    cross_domain_similarity(stability_mod.cross_domain(rows))
    canvas_mechanism(canvas_mod.mechanism_frame(rows))
    canvas_delta_spectrum(canvas_mod.mechanism(rows))
    component_heatmap(stability_mod.dispositions(rows))
    table, totals = entropy_mod.budget(rows, weighting)
    entropy_budget(table, totals, entropy_mod.weighted_attrs(weighting), weighting)
    rob = pd.DataFrame({entropy_mod.STUDY_NAME[w]: entropy_mod.reduction(
        entropy_mod.budget(rows, w)[1]) for w in entropy_mod.WEIGHTINGS}).T
    _, unw, n_attr = entropy_mod.unweighted_reduction(rows)
    rob.loc[f"Unweighted, {n_attr} attributes"] = [unw[a] for a in ARMS]
    robustness(rob)
    classification_sensitivity(entropy_mod.classification_sensitivity(rows, weighting))
    anonymity_sets(entropy_mod.capacity(rows, weighting))
    drop_channels(entropy_mod.drop_channel_sweep(rows, weighting))
    similarity_separation(sim.separation(rows, "brave"))
    cross_browser_linkability(immune_matrix(rows)[0])
