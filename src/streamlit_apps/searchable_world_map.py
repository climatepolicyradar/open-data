from pathlib import Path
import duckdb
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st


from src.data_helpers import download_data
from src.config import CACHE_DIR, REVISION


@st.cache_resource
def load_data():
    download_data(
        cache_dir=str(CACHE_DIR),
        revision=REVISION,
    )

    db = duckdb.connect()

    # Authenticate (needed if loading a private dataset)
    # You'll need to log in using `huggingface-cli login` in your terminal first
    db.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, PROVIDER credential_chain);")

    # Create a view called 'open_data', and count the number of rows and distinct documents
    # in the view
    db.execute(
        f"CREATE VIEW open_data AS SELECT * FROM read_parquet('{CACHE_DIR}/*.parquet')"
    )

    return db


def get_geography_count_for_texts(texts: list[str]) -> pd.DataFrame:
    """
    Get the number of paragraphs containing any of the given texts, grouped by geography.

    Returns dataframe with columns 'geography ISO', and 'count'.
    """
    texts = [f"\\b{text.lower()}\\b" for text in texts]
    regex = f".*({'|'.join(texts)}).*"
    results_df = db.sql(
        f"""
        SELECT "document_metadata.geographies", COUNT(*)
        FROM open_data 
        WHERE "text_block.language" = 'en'
            AND (lower("text_block.text") SIMILAR TO '{regex}')
            AND "document_metadata.geographies" IS NOT NULL
            AND "document_metadata.geographies" <> ['XAA']
        GROUP BY "document_metadata.geographies"
        ORDER BY COUNT(*) DESC
        """
    ).to_df()

    results_df["document_metadata.geographies"] = results_df[
        "document_metadata.geographies"
    ].apply(lambda x: x[0])

    results_df = results_df.rename(
        columns={
            "document_metadata.geographies": "geography ISO",
            "count_star()": "count",
        }
    )

    return results_df


@st.cache_data
def get_num_paragraphs_in_db() -> int:
    return db.sql("SELECT COUNT(*) FROM open_data").to_df().iloc[0, 0]


@st.cache_data
def load_world_geometries():
    """
    Get world geometries in Eckert IV projection.

    Drop Antarctica and Seven seas (open ocean) geometries to make the map look nicer.
    """
    world = gpd.read_file(
        Path(__file__).parent / "../data/earth_vectors/ne_50m_admin_0_countries.shp"
    )
    world = world.to_crs(
        "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    world = world[~world["ADMIN"].isin(["Antarctica", "Seven seas (open ocean)"])]

    return world


def plot_country_map(
    keywords: list[str],
    normalize_counts: bool = False,
    axis=None,
):
    """
    Plot a map of the world with countries colored by the number of paragraphs containing any of the given keywords.

    Returns the raw results.
    """
    results_df = get_geography_count_for_texts(keywords)

    # normalise by paragraph_count_by_geography
    if normalize_counts:
        results_df["count"] = (
            results_df["count"] / paragraph_count_by_geography["count"]
        )
        legend_label = "Relative frequency in dataset"
    else:
        legend_label = "Number of paragraphs"

    min_count, max_count = results_df["count"].min(), results_df["count"].max()
    num_geographies = results_df["geography ISO"].nunique()

    world_with_counts = world.merge(
        results_df, left_on="ADM0_A3", right_on="geography ISO", how="left"
    )

    if axis:
        fig = axis.get_figure()
    else:
        fig, axis = plt.subplots(figsize=(18, 9), dpi=300)

    world_with_counts.plot(
        column="count",
        legend=False,
        figsize=(20, 10),
        ax=axis,
        vmin=min_count,
        vmax=max_count,
        cmap="viridis_r",
        edgecolor="face",
        linewidth=0.3,  # helps small states stand out
        missing_kwds={"color": "darkgrey", "edgecolor": "white", "hatch": "///"},
    )

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    fig.colorbar(
        mpl.cm.ScalarMappable(  # type: ignore
            norm=mpl.colors.Normalize(vmin=min_count, vmax=max_count),  # type: ignore
            cmap="viridis_r",
        ),
        cax=cax,
        orientation="horizontal",
        label=legend_label,
    )

    sns.despine(ax=axis, top=True, bottom=True, left=True, right=True)
    axis.set_xticks([])
    axis.set_yticks([])

    fig.tight_layout()

    axis.set_title(
        f"Number of paragraphs containing words '{', '.join(keywords)}'. {num_geographies} total geographies."
    )

    return results_df


def plot_normalised_unnormalised_subplots(
    kwds,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:  # type: ignore
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)

    df_unnorm = plot_country_map(
        kwds,
        normalize_counts=False,
        axis=ax1,
    )

    df_norm = plot_country_map(
        kwds,
        normalize_counts=True,
        axis=ax2,
    )

    return fig, df_unnorm, df_norm


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    db = load_data()
    world = load_world_geometries()
    num_paragraphs_in_db = get_num_paragraphs_in_db()
    paragraph_count_by_geography = get_geography_count_for_texts([".*"])

    st.title("Searchable World Map")
    st.markdown(
        "Search for keywords in the dataset and see where they appear on a world map."
    )
    with st.expander("You can use regex! Open for examples"):
        st.markdown(r"""
        - `natural(-|\s)resource`: match "natural-resource" and "natural resource"
        - `fish(es)?`: match "fish" and "fishes"
        - `elephants?`: match "elephant" and "elephants"
        """)

    kwds = st.text_input(
        "Enter keywords separated by commas (spaces next to commas will be ignored)"
    )

    if kwds:
        kwds = [word.strip() for word in kwds.split(",")]

        st.markdown("## all keywords")
        fig, data1, data2 = plot_normalised_unnormalised_subplots(kwds)
        n_paragraphs = data1["count"].sum()
        percentage = round(n_paragraphs / num_paragraphs_in_db * 100, 2)
        st.markdown(f"Num paragraphs: {n_paragraphs:,} ({percentage}%)")
        st.write(fig)

        if len(kwds) > 1:
            for keyword in kwds:
                st.markdown(f"## {keyword}")
                fig, data1, data2 = plot_normalised_unnormalised_subplots([keyword])
                n_paragraphs = data1["count"].sum()
                percentage = round(n_paragraphs / num_paragraphs_in_db * 100, 2)
                st.markdown(f"Num paragraphs: {n_paragraphs:,} ({percentage}%)")
                st.write(fig)
