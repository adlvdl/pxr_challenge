import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 1e — Scaffold analysis

    Decomposes every unique molecule into a hierarchy of ring-based scaffolds and
    analyses how those scaffolds are distributed across the four datasets.

    **Scaffold types produced:**

    | Type | Description |
    |------|-------------|
    | `ring_system` | Individual fused/bridged polycyclic ring system or isolated ring |
    | `linked_ring_systems` | Two ring systems joined by a linker (zero-atom for direct bonds) |
    | `full_scaffold` | For the most part equivalent to a Bemis-Murcko scaffold |

    The interactive scatter at the end lets you see which scaffolds are shared
    between the training and test sets, with on-click structure rendering.

    **Input:** `data/processed/all_compounds_activity_data.csv` (produced by **1a**).
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import altair as alt
    import itertools
    from pathlib import Path
    from typing import Optional
    from collections import defaultdict, deque

    import matplotlib.pyplot as plt
    import matplotlib.figure

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    return (
        Chem,
        Optional,
        alt,
        defaultdict,
        deque,
        itertools,
        mo,
        pl,
        rdDepictor,
        rdMolDraw2D,
    )


@app.cell
def _(Chem, Optional, defaultdict, deque, itertools, pl):
    # ── Internal helpers ──────────────────────────────────────────────────────────

    def _build_ring_systems(mol: Chem.Mol) -> list[frozenset[int]]:
        """
        Return one frozenset of atom indices per ring system.

        A ring system is the union of all SSSR rings that share at least one
        atom (fused, bridged, or spiro rings merge into one system).
        """
        atom_rings = mol.GetRingInfo().AtomRings()
        if not atom_rings:
            return []

        n = len(atom_rings)
        adj: dict[int, set[int]] = defaultdict(set)
        for i, j in itertools.combinations(range(n), 2):
            if set(atom_rings[i]) & set(atom_rings[j]):
                adj[i].add(j); adj[j].add(i)

        visited = [False] * n
        systems: list[frozenset[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            component: list[int] = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                queue.extend(adj[node])
            systems.append(frozenset(a for ri in component for a in atom_rings[ri]))
        return systems

    def _linker_atoms(
        mol_adj: dict[int, set[int]],
        rs1: frozenset[int],
        rs2: frozenset[int],
        non_ring: set[int],
    ) -> Optional[frozenset[int]]:
        """
        Find the set of non-ring atoms forming the shortest path between two ring systems.

        Returns an empty frozenset for a direct ring–ring bond, or None when the
        two systems are not connected.
        """
        for a in rs1:
            if any(nb in rs2 for nb in mol_adj[a]):
                return frozenset()

        prev: dict[int, int] = {}
        queue = deque()
        for a in rs1:
            for nb in mol_adj[a]:
                if nb in non_ring and nb not in prev:
                    prev[nb] = a; queue.append(nb)

        while queue:
            cur = queue.popleft()
            if cur in rs2:
                path: list[int] = []
                node = cur
                while node not in rs1:
                    if node in non_ring:
                        path.append(node)
                    node = prev[node]
                return frozenset(path)
            for nb in mol_adj[cur]:
                if nb not in prev and (nb in non_ring or nb in rs2):
                    prev[nb] = cur; queue.append(nb)
        return None

    def _to_canonical(mol: Chem.Mol, atom_set: frozenset[int]) -> Optional[str]:
        """
        Canonical SMILES for an atom subset; None on failure or empty set.

        Kekulizes the parent molecule first so extracted aromatic fragments can
        be re-parsed correctly by ``MolFromSmiles``.
        """
        if not atom_set:
            return None
        try:
            mol_kek = Chem.RWMol(mol)
            Chem.Kekulize(mol_kek, clearAromaticFlags=False)
            smi = Chem.MolFragmentToSmiles(mol_kek, sorted(atom_set), kekuleSmiles=True)
            if not smi:
                return None
            m = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(m) if m is not None else None
        except Exception:
            return None

    # ── Public function ───────────────────────────────────────────────────────────

    def decompose_scaffold_network(smiles: list[str]) -> pl.DataFrame:
        """
        Decompose each molecule into three exhaustive levels of ring-based scaffolds.

        **Scaffold types:**

        - ``ring_system``: A single fused/bridged ring system with all substituents removed.
        - ``linked_ring_systems``: A pair of ring systems connected by a linker.
        - ``full_scaffold``: All ring systems + linkers (≥ 3 ring systems, all connected).

        Molecules with no rings produce zero rows. Invalid SMILES are recorded with a
        ``parse_error`` message and null scaffold columns.

        Args:
            smiles: Input SMILES strings.

        Returns:
            Long-format Polars DataFrame with columns:
            - ``smiles``              : original input SMILES
            - ``scaffold_smiles``     : canonical scaffold SMILES (null on failure)
            - ``scaffold_type``       : scaffold type string (null on failure)
            - ``scaffold_heavy_atoms``: heavy-atom count of the scaffold (null on failure)
            - ``parse_error``         : error message or null
        """
        in_smi_col:  list[str]           = []
        sc_smi_col:  list[Optional[str]] = []
        sc_type_col: list[Optional[str]] = []
        sc_ha_col:   list[Optional[int]] = []
        err_col:     list[Optional[str]] = []

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                in_smi_col.append(smi); sc_smi_col.append(None)
                sc_type_col.append(None); sc_ha_col.append(None)
                err_col.append(f"Invalid SMILES: {smi!r}")
                continue

            ring_systems = _build_ring_systems(mol)
            if not ring_systems:
                continue  # acyclic — no scaffold rows

            ring_atom_set: set[int] = set(a for rs in ring_systems for a in rs)
            non_ring: set[int] = set(range(mol.GetNumAtoms())) - ring_atom_set

            mol_adj: dict[int, set[int]] = defaultdict(set)
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                mol_adj[u].add(v); mol_adj[v].add(u)

            seen: dict[str, str] = {}

            def _emit(atom_set: frozenset[int], stype: str) -> None:
                csmi = _to_canonical(mol, atom_set)
                if csmi and csmi not in seen:
                    seen[csmi] = stype

            # Level 1: individual ring systems
            for rs in ring_systems:
                _emit(rs, "ring_system")

            # Level 2: pairs of ring systems + linker
            linker_map: dict[tuple[int, int], Optional[frozenset[int]]] = {}
            rs_connected: dict[int, set[int]] = defaultdict(set)

            for i, rs1 in enumerate(ring_systems):
                for j, rs2 in enumerate(ring_systems):
                    if j <= i:
                        continue
                    linker = _linker_atoms(mol_adj, rs1, rs2, non_ring)
                    linker_map[(i, j)] = linker
                    if linker is not None:
                        rs_connected[i].add(j); rs_connected[j].add(i)
                        _emit(rs1 | rs2 | linker, "linked_ring_systems")

            # Level 3: full scaffold (≥ 3 ring systems, all connected)
            if len(ring_systems) >= 3:
                rs_visited: list[bool] = [False] * len(ring_systems)
                for rs_start in range(len(ring_systems)):
                    if rs_visited[rs_start]:
                        continue
                    comp: list[int] = []
                    q = deque([rs_start])
                    while q:
                        node = q.popleft()
                        if rs_visited[node]:
                            continue
                        rs_visited[node] = True
                        comp.append(node)
                        q.extend(rs_connected[node])

                    if len(comp) >= 3:
                        all_atoms: frozenset[int] = frozenset(
                            a for rs_idx in comp for a in ring_systems[rs_idx]
                        )
                        for a, b in itertools.combinations(comp, 2):
                            lk = linker_map.get((min(a, b), max(a, b)))
                            if lk:
                                all_atoms = all_atoms | lk
                        _emit(all_atoms, "full_scaffold")

            for csmi, stype in seen.items():
                sc_mol = Chem.MolFromSmiles(csmi)
                ha = sc_mol.GetNumHeavyAtoms() if sc_mol is not None else None
                in_smi_col.append(smi); sc_smi_col.append(csmi)
                sc_type_col.append(stype); sc_ha_col.append(ha)
                err_col.append(None)

        return pl.DataFrame({
            "smiles":               pl.Series(in_smi_col,  dtype=pl.Utf8),
            "scaffold_smiles":      pl.Series(sc_smi_col,  dtype=pl.Utf8),
            "scaffold_type":        pl.Series(sc_type_col, dtype=pl.Utf8),
            "scaffold_heavy_atoms": pl.Series(sc_ha_col,   dtype=pl.Int32),
            "parse_error":          pl.Series(err_col,     dtype=pl.Utf8),
        })

    return (decompose_scaffold_network,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    all_compounds
    return (all_compounds,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Decompose scaffolds for all unique molecules

    Each unique molecule (by InChIKey) is decomposed into ring-based scaffolds.
    Deduplication before the call avoids running the expensive ring-perception
    step multiple times for the same structure.
    """)
    return


@app.cell
def _(all_compounds, decompose_scaffold_network, pl):
    _unique_mols = (
        all_compounds
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles", "in_single_dose", "in_dose_response",
                 "in_counter", "in_test"])
        .drop_nulls(subset=["smiles"])
    )

    _scaffold_long = decompose_scaffold_network(
        _unique_mols.get_column("smiles").to_list()
    )

    # Join dataset membership flags back onto the long-format scaffold table
    _membership = _unique_mols.select(
        ["smiles", "in_single_dose", "in_dose_response", "in_counter", "in_test"]
    )

    scaffold_hits = (
        _scaffold_long
        .filter(pl.col("parse_error").is_null())
        .join(_membership, on="smiles", how="left")
    )

    scaffold_hits
    return (scaffold_hits,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Scaffold counts per dataset

    For each unique scaffold, count how many distinct molecules (by SMILES)
    contain it, broken down by dataset membership.

    Because a molecule can belong to multiple datasets, per-dataset counts can
    sum to more than `n_total` — this is intentional and informative.
    """)
    return


@app.cell
def _(pl, scaffold_hits):
    scaffold_counts = (
        scaffold_hits
        # Deduplicate within molecule (safety guard)
        .unique(subset=["scaffold_smiles", "smiles"])
        .group_by(["scaffold_smiles", "scaffold_type"])
        .agg(
            pl.col("scaffold_heavy_atoms").first().alias("scaffold_heavy_atoms"),
            pl.col("smiles").n_unique().alias("n_total"),
            pl.col("in_single_dose") .filter(pl.col("in_single_dose")) .len().alias("n_single_dose"),
            pl.col("in_dose_response").filter(pl.col("in_dose_response")).len().alias("n_dose_response"),
            pl.col("in_counter")     .filter(pl.col("in_counter"))      .len().alias("n_counter"),
            pl.col("in_test")        .filter(pl.col("in_test"))         .len().alias("n_test"),
        )
        .select([
            "scaffold_smiles", "scaffold_type", "scaffold_heavy_atoms",
            "n_total", "n_single_dose", "n_dose_response", "n_counter", "n_test",
        ])
        .sort(["scaffold_type", "n_total"], descending=[False, True])
    )

    scaffold_counts
    return (scaffold_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Scaffold coverage of the test set by dose-response training compounds

    Each point represents a scaffold present in **at least one test compound** and
    **at least one dose-response training compound**.

    - **x-axis** — number of dose-response training molecules containing this scaffold
    - **y-axis** — number of test molecules containing this scaffold
    - **colour** — scaffold heavy-atom count

    Scaffolds sitting high on the y-axis but far left are test-set enriched (low
    training coverage); scaffolds far right and low are training-enriched.
    Click a point to see the scaffold structure drawn on the right.
    """)
    return


@app.cell
def _(alt, mo, pl, scaffold_counts):
    _plot_df = scaffold_counts.filter(
        (pl.col("n_test") >= 1) & (pl.col("n_dose_response") >= 1)
    )

    _sel = alt.selection_point(fields=["scaffold_smiles"], name="scaffold_sel", empty=False)

    _scatter = (
        alt.Chart(_plot_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X(
                "n_dose_response:Q",
                title="Dose-response training molecules",
                scale=alt.Scale(type="log", base=10, domain=[0.9, 5000]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11,
                              values=[1, 10, 100, 1000, 5000], format="~s"),
            ),
            y=alt.Y(
                "n_test:Q",
                title="Test-set molecules",
                scale=alt.Scale(type="log", base=10, domain=[0.9, 500]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11,
                              values=[1, 10, 100, 500], format="~s"),
            ),
            color=alt.condition(
                _sel,
                alt.value("#f5c518"),
                alt.Color("scaffold_heavy_atoms:Q",
                           scale=alt.Scale(scheme="viridis"),
                           legend=alt.Legend(title="Heavy atoms")),
            ),
            size=alt.condition(_sel, alt.value(150), alt.value(60)),
            tooltip=[
                alt.Tooltip("scaffold_smiles:N",      title="Scaffold"),
                alt.Tooltip("scaffold_type:N",        title="Type"),
                alt.Tooltip("scaffold_heavy_atoms:Q", title="Heavy atoms"),
                alt.Tooltip("n_dose_response:Q",      title="# dose-response"),
                alt.Tooltip("n_test:Q",               title="# test"),
                alt.Tooltip("n_total:Q",              title="# total"),
            ],
        )
        .add_params(_sel)
        .properties(
            title="Shared scaffolds — dose-response training vs. test set",
            width=500,
            height=400,
        )
        .configure_title(fontSize=12)
    )

    scaffold_coverage_chart = mo.ui.altair_chart(_scatter)
    return (scaffold_coverage_chart,)


@app.cell
def _(
    Chem,
    mo,
    pl,
    rdDepictor,
    rdMolDraw2D,
    scaffold_counts,
    scaffold_coverage_chart,
):
    _PANEL_W = 300
    _PANEL_H = 260

    def _scaffold_to_svg(smi: str, width: int = _PANEL_W, height: int = _PANEL_H) -> str:
        """Render a scaffold SMILES as an SVG string via RDKit MolDraw2DSVG."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def _strip_xml(svg: str) -> str:
        return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg

    _sel_rows = scaffold_coverage_chart.value

    if _sel_rows is None or len(_sel_rows) == 0:
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; height:{_PANEL_H + 80}px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:14px; border:1px dashed #ccc;
                        border-radius:6px'>
                Click a point to see the scaffold structure
            </div>
        """)
    else:
        _scaffold_smi = _sel_rows.row(0, named=True)["scaffold_smiles"]
        _r = (
            scaffold_counts
            .filter(pl.col("scaffold_smiles") == _scaffold_smi)
            .row(0, named=True)
        )
        _svg = _scaffold_to_svg(_r["scaffold_smiles"])
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; font-family:monospace; font-size:11px'>
                <div style='padding:6px; background:#f5f5f5; border-radius:4px;
                            margin-bottom:4px; line-height:1.6'>
                    <b>Type:</b> {_r['scaffold_type']}<br>
                    <b>Heavy atoms:</b> {_r['scaffold_heavy_atoms']}<br>
                    <b># dose-response:</b> {_r['n_dose_response']}<br>
                    <b># test:</b> {_r['n_test']}<br>
                    <b># total:</b> {_r['n_total']}
                </div>
                {_strip_xml(_svg)}
            </div>
        """)

    mo.hstack([scaffold_coverage_chart, _panel], align="start")
    return


if __name__ == "__main__":
    app.run()
