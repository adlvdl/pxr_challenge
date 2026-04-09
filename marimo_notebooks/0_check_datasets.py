import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Import libraries

    For this first notebook, most of the libraries will be for chemoinformatics or data mining
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    from rdkit import Chem
    import mols2grid

    #This removes some warnings that appear when calculating INCHIs
    from rdkit import RDLogger 
    RDLogger.DisableLog('rdApp.*')
    return Chem, mo, pl


@app.cell
def _(Chem):
    #adapted from https://rdkit.blogspot.com/2015/02/new-drawing-code.html
    #first seen at https://iwatobipen.wordpress.com/2024/01/19/new-type-of-python-notebook-marimo-cheminformatics-rdkit/
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    def smitosvg(smi,molSize=(200,200),kekulize=True):
        mc = Chem.MolFromSmiles(smi)
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.MolFromSmiles(smi)
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # It seems that the svg renderer used doesn't quite hit the spec.
        # Here are some fixes to make it work in the notebook, although I think
        # the underlying issue needs to be resolved at the generation step
        return svg.replace('svg:','')

    return (smitosvg,)


@app.cell
def _(mo):
    mo.md(r"""
    To check whether there are duplicates you should never compare SMILES strings directly.
    The better way is to obtain their INChIKey, based on their INChI representation.
    The INChI is unique and canonical, while for SMILES different toolkits might provide
    different canonical SMILES for the same chemical structure.
    INChIs also have the benefit to represent different tautomers with the same string.
    """)
    return


@app.cell
def _(Chem):
    def smi_to_inchikey(smi: str):
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    def smi_to_inchi(smi: str):
        return Chem.MolToInchi(Chem.MolFromSmiles(smi))

    return smi_to_inchi, smi_to_inchikey


@app.cell
def _(pl, smi_to_inchi, smi_to_inchikey, smitosvg):
    # the rename is there because some datasets use smiles and others SMILES
    # and also I prefer lowercase
    def process_dataset(df:pl.DataFrame):
        return df.rename({"SMILES":"smiles"}).with_columns(
            pl.col("smiles").map_elements(smi_to_inchikey).alias("inchikey"),
            pl.col("smiles").map_elements(smi_to_inchi).alias("inchi"),
            pl.col("smiles").map_elements(smitosvg).alias("svg")
        )

    return (process_dataset,)


@app.cell
def _(pl, process_dataset):
    # Read datasets

    train: pl.DataFrame         = process_dataset(pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv"))
    test          = process_dataset(pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv"))
    train_counter  = process_dataset(pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv"))
    train_single   = process_dataset(pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv"))
    return test, train, train_counter, train_single


@app.cell
def _(mo):
    mo.md(r"""
    ## Dose response training dataset
    """)
    return


@app.cell
def _(train: "pl.DataFrame"):
    #Check overall structure of training dataset

    print("Number of rows: ", train.shape[0])
    print("Number of unique IDs: ", train.get_column("Molecule Name").unique().len())
    print("Number of unique SMILES: ", train.get_column("smiles").unique().len())
    print("Column names: ", train.columns)
    print("Number of unique INCHIKEYS:", train.get_column("inchikey").unique().len())
    return


@app.cell
def _(mo):
    mo.md(r"""
    There seems to be 4139 unique INCHIKEYs, 1 less than unique IDs or SMILES.
    Lets check the structures
    """)
    return


@app.cell
def _(mo, pl, smi_to_inchi, smitosvg, train: "pl.DataFrame"):

    dup_inchikey = train.group_by("inchikey").len().\
        filter(pl.col("len")>1).get_column("inchikey").to_list()[0]
    smiles = train.filter(pl.col("inchikey")==dup_inchikey).get_column("smiles").to_list()
    print("IDs: ", train.filter(pl.col("inchikey")==dup_inchikey).get_column("Molecule Name").to_list())
    print("SMILES with same INCHIKEY: \n", smiles[0], "\n", smiles[1])
    print("their INCHIs: \n", smi_to_inchi(smiles[0]), 
            "\n", smi_to_inchi(smiles[1]))

    mo.Html(smitosvg(smiles[0])+"<br>"+smitosvg(smiles[1]))
    return (dup_inchikey,)


@app.cell
def _(dup_inchikey, pl, train: "pl.DataFrame"):
    train.filter(pl.col("inchikey")==dup_inchikey).get_column("Molecule Name").to_list()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Seems to be only a difference of the stereoinformation around a double bond,
    which INCHI seems to disregard. Looking at the H layer, it seems to be included
    in the adyacent ring system for delocalized hydrogens and resonant double bonds.
    Looking below at the activity data, it seems very similar, so you could just combine
    the records if you want to use the INCHIKEY as unique ID
    """)
    return


@app.cell
def _(dup_inchikey, pl, train: "pl.DataFrame"):
    train.filter(pl.col("inchikey")==dup_inchikey)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now lets continue looking at the other datasets.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Single dose training dataset
    """)
    return


@app.cell
def _(train_single):
    #Check overall structure of train_single dataset

    print("Number of rows: ", train_single.shape[0])
    print("Number of unique IDs: ", train_single.get_column("Molecule Name").unique().len())
    print("Number of unique SMILES: ", train_single.get_column("smiles").unique().len())
    print("Column names: ", train_single.columns)
    print("Number of unique INCHIKEYS:", train_single.get_column("inchikey").unique().len())
    return


@app.cell
def _(mo):
    mo.md(r"""
    OK, now we have a messier dataset. We have 21K rows, but only 10.8K unique structures
    (no SMILES to INCHIKEY mismatch this time around) and only 2747 unique IDs.
    One possibility is that Molecule Name is no longer the main ID in this dataset.
    Lets check a few rows.
    """)
    return


@app.cell
def _(train_single):
    train_single
    return


@app.cell
def _(mo):
    mo.md(r"""
    The table shows us that there are many missing rows for Molecule Name in this dataset.
    It looks like OCNT Batch contains a more stable ID.
    So I will mainly use the INCHIKEY as ID to compare the different datasets.
    So first let's check the overlap of this dataset to the main training dataset.
    According to the blog post there should be 2745 compounds from the single dose screen promoted
    to dose response.
    """)
    return


@app.cell
def _(train: "pl.DataFrame", train_single):
    print("SMILES in common: ", len(set(train_single.get_column("smiles").to_list()).intersection(
        train.get_column("smiles").to_list()
    )))
    print("INCHIKEYS in common: ", len(set(train_single.get_column("inchikey").to_list()).intersection(
        train.get_column("inchikey").to_list()
    )))
    print("Molecule Name in common: ", len(set(train_single.get_column("Molecule Name").to_list()).intersection(
        train.get_column("Molecule Name").to_list()
    )))
    print("OCNT Batch in common: ", len(set(train_single.get_column("OCNT Batch").to_list()).intersection(
        train.get_column("OCNT Batch").to_list()
    )))
    return


@app.cell
def _(mo):
    mo.md(r"""
    We start seeing the messines in action as different ways to compare overlap gives us different
    numbers, and none of them add up to 2745, the number mentioned in the blog post. The closest is the INCHIKEY which
    would be my choice in general, as its based on the chemical structure. It is expected that SMILES would
    provide the least number of common occurances due what we mentioned before, different SMILES for the
    same chemical structure.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Counter assay
    """)
    return


@app.cell
def _(train_counter):
    #Check overall structure of train_counter dataset

    print("Number of rows: ", train_counter.shape[0])
    print("Number of unique IDs: ", train_counter.get_column("Molecule Name").unique().len())
    print("Number of unique SMILES: ", train_counter.get_column("smiles").unique().len())
    print("Column names: ", train_counter.columns)
    print("Number of unique INCHIKEYS:", train_counter.get_column("inchikey").unique().len())
    return


@app.cell
def _(mo):
    mo.md(r"""
    We find again a slight mismatch between SMILES and INCHIKEYs,
    but it is possible this is the same issue as in the training dataset,
    as the counter screen is supposed to be a subset of the dose response screen.
    """)
    return


@app.cell
def _(dup_inchikey, pl, train_counter):
    train_counter.filter(pl.col("inchikey")==dup_inchikey)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Indeed, we find the same IDs, again with similar activities. Now lets check overlaps.
    """)
    return


@app.cell
def _(train: "pl.DataFrame", train_counter):
    print("SMILES in common: ", len(set(train_counter.get_column("smiles").to_list()).intersection(
        train.get_column("smiles").to_list()
    )))
    print("INCHIKEYS in common: ", len(set(train_counter.get_column("inchikey").to_list()).intersection(
        train.get_column("inchikey").to_list()
    )))
    print("Molecule Name in common: ", len(set(train_counter.get_column("Molecule Name").to_list()).intersection(
        train.get_column("Molecule Name").to_list()
    )))
    print("OCNT Batch in common: ", len(set(train_counter.get_column("OCNT Batch").to_list()).intersection(
        train.get_column("OCNT Batch").to_list()
    )))
    return


@app.cell
def _(mo):
    mo.md(r"""
    These results matches what was shown in the blog post, 2860 compounds where promoted
    from dose response to counter screen.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test dataset
    """)
    return


@app.cell
def _(test):
    #Check overall structure of test dataset

    print("Number of rows: ", test.shape[0])
    print("Number of unique IDs: ", test.get_column("Molecule Name").unique().len())
    print("Number of unique SMILES: ", test.get_column("smiles").unique().len())
    print("Column names: ", test.columns)
    print("Number of unique INCHIKEYS:", test.get_column("inchikey").unique().len())
    return


@app.cell
def _(mo):
    mo.md(r"""
    No mismatch between IDs, SMILES and INCHIKEYs in the dataset. Now again lets check for overlap,
    which should be zero.
    """)
    return


@app.cell
def _(test, train: "pl.DataFrame"):
    print("INCHIKEYS in common: ", len(set(test.get_column("inchikey").to_list()).intersection(
        train.get_column("inchikey").to_list()
    )))
    print("Molecule Name in common: ", len(set(test.get_column("Molecule Name").to_list()).intersection(
        train.get_column("Molecule Name").to_list()
    )))
    return


@app.cell
def _(mo):
    mo.md(r"""
    As expected no overlap with the training set, lets check now overlap with the single dose
    dataset.
    """)
    return


@app.cell
def _(test, train_single):
    print("INCHIKEYS in common: ", len(set(train_single.get_column("inchikey").to_list()).intersection(
        test.get_column("inchikey").to_list()
    )))
    print("Molecule Name in common: ", len(set(train_single.get_column("Molecule Name").to_list()).intersection(
        test.get_column("Molecule Name").to_list()
    )))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Also no overlap, which is good to see.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Saving datasets

    It is recommended to be organized when saving and handling datasets.
    Ideally, your organization has a well maintained data repository so you do not
    need to worry.
    For this, I plan to keep the data locally, to provide a snapshot of the data.
    It is also important to separate raw data from processed data for two reasons:
    - archival of the data process
    - be able to return to the original datasets if you change your preprocessing pipeline
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    I download the dataset and save the raw version on a folder with the date in case
    the dataset is updated or modified during the competition:
    """)
    return


@app.cell
def _(pl):


    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv").write_csv("../data/raw/20260403/dose_response_train.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv").write_csv("../data/raw/20260403/dose_response_test.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv").write_csv("../data/raw/20260403/counter_screen_train.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv").write_csv("../data/raw/20260403/structure_test.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv").write_csv("../data/raw/20260403/single_dose_train.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDIT 2026/04/09

    The datasets at HuggingFace were updated to solve a few of the issues identified in this notebook and
     reported by other participants. So I download them again but to a separate folder.
    More info on the changes at https://docs.google.com/document/d/14-2EL4Zk8g3NNO7bi33gA3fHz_ef98bBigunJMH3Ui4/edit?tab=t.0#heading=h.azqdv6g0boyn
    """)
    return


@app.cell
def _(pl):
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv").write_csv("../data/raw/20260409/dose_response_train.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv").write_csv("../data/raw/20260409/dose_response_test.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv").write_csv("../data/raw/20260409/counter_screen_train.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv").write_csv("../data/raw/20260409/structure_test.csv")
    pl.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv").write_csv("../data/raw/20260409/single_dose_train.csv")
    return


if __name__ == "__main__":
    app.run()
