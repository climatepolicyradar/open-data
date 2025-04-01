from typing import Optional
from pathlib import Path
import duckdb
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st
from datetime import datetime


from src.data_helpers import download_data
from src.config import CACHE_DIR, DATA_REVISION


@st.cache_resource
def load_data():
    download_data(
        cache_dir=str(CACHE_DIR),
        revision=DATA_REVISION,
    )

    db = duckdb.connect('data.db')  # Create a persistent database

    # Authenticate (needed if loading a private dataset)
    # You'll need to log in using `huggingface-cli login` in your terminal first
    db.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, PROVIDER credential_chain);")

    # Check if table exists
    table_exists = db.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'open_data'").fetchone()[0] > 0

    if not table_exists:
        # Create a persistent table with only the columns we need
        db.execute("""
            CREATE TABLE open_data AS 
            SELECT 
                "document_metadata.geographies",
                "document_metadata.corpus_type_name",
                "document_metadata.publication_ts",
                "text_block.text",
                "text_block.language",
                "text_block.type"
            FROM read_parquet('{}/*.parquet')
        """.format(CACHE_DIR))

        # Create indexes for common query patterns
        db.execute("CREATE INDEX idx_language ON open_data(\"text_block.language\")")
        db.execute("CREATE INDEX idx_corpus_type ON open_data(\"document_metadata.corpus_type_name\")")
        db.execute("CREATE INDEX idx_publication_ts ON open_data(\"document_metadata.publication_ts\")")
        db.execute("CREATE INDEX idx_text_type ON open_data(\"text_block.type\")")

    return db


def get_geography_count_for_texts(
    texts: list[str], corpus_type_names: Optional[list[str]], year_range: Optional[tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Get the number of paragraphs containing any of the given texts, grouped by geography.

    Returns dataframe with columns 'geography ISO', and 'count'.
    """
    texts = [f"\\b{text.lower()}\\b" for text in texts]
    regex = f".*({'|'.join(texts)}).*"

    if corpus_type_names is None:
        corpus_type_names_clause = ""
    else:
        corpus_type_names_string = (
            "(" + ",".join([f"'{name}'" for name in corpus_type_names]) + ")"
        )
        corpus_type_names_clause = f"""AND "document_metadata.corpus_type_name" IN {corpus_type_names_string} """

    year_clause = ""
    if year_range is not None:
        year_clause = f"""AND EXTRACT(YEAR FROM "document_metadata.publication_ts"::timestamp) BETWEEN {year_range[0]} AND {year_range[1]}"""

    sql_query = f"""
        SELECT "document_metadata.geographies", COUNT(*)
        FROM open_data 
        WHERE "text_block.language" = 'en'
            AND (lower("text_block.text") SIMILAR TO '{regex}')
            AND "document_metadata.geographies" IS NOT NULL
            AND "document_metadata.geographies" <> ['XAA']
            AND ("text_block.type" = 'title' OR  "text_block.type" = 'Text' OR "text_block.type" =  'sectionHeading')
            {corpus_type_names_clause}
            {year_clause}
        GROUP BY "document_metadata.geographies"
        ORDER BY COUNT(*) DESC
        """

    results_df = db.sql(sql_query).to_df()

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


def get_corpus_type_count_for_texts(
    texts: list[str], corpus_type_names: Optional[list[str]], year_range: Optional[tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Get the number of paragraphs containing any of the given texts, grouped by corpus type.

    Returns dataframe with columns 'corpus_type_name' and 'count'.
    """
    texts = [f"\\b{text.lower()}\\b" for text in texts]
    regex = f".*({'|'.join(texts)}).*"

    if corpus_type_names is None:
        corpus_type_names_clause = ""
    else:
        corpus_type_names_string = (
            "(" + ",".join([f"'{name}'" for name in corpus_type_names]) + ")"
        )
        corpus_type_names_clause = f"""AND "document_metadata.corpus_type_name" IN {corpus_type_names_string} """

    year_clause = ""
    if year_range is not None:
        year_clause = f"""AND EXTRACT(YEAR FROM "document_metadata.publication_ts"::timestamp) BETWEEN {year_range[0]} AND {year_range[1]}"""

    sql_query = f"""
        SELECT "document_metadata.corpus_type_name", COUNT(*)
        FROM open_data 
        WHERE "text_block.language" = 'en'
            AND (lower("text_block.text") SIMILAR TO '{regex}')
            AND "document_metadata.corpus_type_name" IS NOT NULL
            AND ("text_block.type" = 'title' OR  "text_block.type" = 'Text' OR "text_block.type" =  'sectionHeading')
            {corpus_type_names_clause}
            {year_clause}
        GROUP BY "document_metadata.corpus_type_name"
        ORDER BY COUNT(*) DESC
        """

    results_df = db.sql(sql_query).to_df()

    results_df = results_df.rename(
        columns={
            "document_metadata.corpus_type_name": "corpus_type_name",
            "count_star()": "count",
        }
    )

    return results_df


def get_time_series_for_texts(
    texts: list[str], corpus_type_names: Optional[list[str]], year_range: Optional[tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Get the number of paragraphs containing any of the given texts, grouped by year and corpus type.

    Returns dataframe with columns 'year', 'corpus_type_name', and 'count'.
    """
    texts = [f"\\b{text.lower()}\\b" for text in texts]
    regex = f".*({'|'.join(texts)}).*"

    if corpus_type_names is None:
        corpus_type_names_clause = ""
    else:
        corpus_type_names_string = (
            "(" + ",".join([f"'{name}'" for name in corpus_type_names]) + ")"
        )
        corpus_type_names_clause = f"""AND "document_metadata.corpus_type_name" IN {corpus_type_names_string} """

    year_clause = ""
    if year_range is not None:
        year_clause = f"""AND EXTRACT(YEAR FROM "document_metadata.publication_ts"::timestamp) BETWEEN {year_range[0]} AND {year_range[1]}"""

    sql_query = f"""
        SELECT 
            EXTRACT(YEAR FROM "document_metadata.publication_ts"::timestamp) as year,
            "document_metadata.corpus_type_name" as corpus_type_name,
            COUNT(*) as count
        FROM open_data 
        WHERE "text_block.language" = 'en'
            AND (lower("text_block.text") SIMILAR TO '{regex}')
            AND "document_metadata.publication_ts" IS NOT NULL
            AND "document_metadata.corpus_type_name" IS NOT NULL
            AND ("text_block.type" = 'title' OR  "text_block.type" = 'Text' OR "text_block.type" =  'sectionHeading')
            {corpus_type_names_clause}
            {year_clause}
        GROUP BY year, corpus_type_name
        ORDER BY year, corpus_type_name
        """

    results_df = db.sql(sql_query).to_df()
    return results_df


@st.cache_data
def get_num_paragraphs_in_db() -> int:
    return db.sql("SELECT COUNT(*) FROM open_data").to_df().iloc[0, 0]


@st.cache_data
def get_unique_corpus_type_names() -> list[str]:
    names = (
        db.sql(
            """SELECT DISTINCT "document_metadata.corpus_type_name" FROM open_data"""
        )
        .to_df()["document_metadata.corpus_type_name"]
        .tolist()
    )

    return [n for n in names if n is not None]


@st.cache_data
def get_year_range() -> tuple[int, int]:
    """Get the min and max years from the dataset.
        Due to errors in the input data, it currently returns a minimum of 1960
    """
    result = db.sql(
        """
        SELECT 
            MIN(EXTRACT(YEAR FROM "document_metadata.publication_ts"::timestamp)) as min_year,
            MAX(EXTRACT(YEAR FROM "document_metadata.publication_ts"::timestamp)) as max_year
        FROM open_data
        WHERE "document_metadata.publication_ts" IS NOT NULL
        """
    ).to_df()
    
    lowest = int(result.iloc[0, 0])
    if lowest < 1960: lowest = 1960
    highest =  int(result.iloc[0, 1])

    return lowest, highest


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
    corpus_type_names: Optional[list[str]] = None,
    year_range: Optional[tuple[int, int]] = None,
    normalize_counts: bool = False,
    axis=None,
):
    """
    Plot a map of the world with countries colored by the number of paragraphs containing any of the given keywords.

    Returns the raw results.
    """
    results_df = get_geography_count_for_texts(keywords, corpus_type_names, year_range)

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
        figsize=(18, 9),
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

    # Add a title with key stats; if it's too long, truncate the keywords
    keywords_joined = ", ".join(keywords)
    if len(keywords_joined) > 15:
        keywords_joined = f"{keywords_joined[0:15]}..."

    axis.set_title(
        f"Number of paragraphs containing: '{keywords_joined}'. From {num_geographies} geographies."
    )

    return results_df


def plot_corpus_type_bar(
    keywords: list[str],
    corpus_type_names: Optional[list[str]] = None,
    year_range: Optional[tuple[int, int]] = None,
    axis=None,
):
    """
    Plot a bar chart showing the number of paragraphs containing the keywords by corpus type.
    """
    results_df = get_corpus_type_count_for_texts(keywords, corpus_type_names, year_range)

    if axis:
        fig = axis.get_figure()
    else:
        fig, axis = plt.subplots(figsize=(12, 6), dpi=300)

    sns.barplot(data=results_df, x="count", y="corpus_type_name", ax=axis)
    axis.set_title(f"Number of paragraphs containing keywords by corpus type")
    axis.set_xlabel("Number of paragraphs")
    axis.set_ylabel("Corpus type")

    fig.tight_layout()
    return results_df


def plot_time_series(
    keywords: list[str],
    corpus_type_names: Optional[list[str]] = None,
    year_range: Optional[tuple[int, int]] = None,
    axis=None,
    relative = False
):
    """
    Plot a stacked line chart showing the number of paragraphs containing the keywords over time,
    with each corpus type represented as a different layer in the stack.
    """
    results_df = get_time_series_for_texts(keywords, corpus_type_names, year_range)

    if axis:
        fig = axis.get_figure()
    else:
        fig, axis = plt.subplots(figsize=(12, 6), dpi=300)

    # Pivot the data to create a stacked area plot
    pivot_df = results_df.pivot(index='year', columns='corpus_type_name', values='count').fillna(0)

    if relative:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis = 0)*100

    # Create the stacked area plot
    pivot_df.plot(kind='area', stacked=True, ax=axis, alpha=0.7)
    
    # Customize the plot
    axis.set_title(f"Number of paragraphs containing keywords over time by corpus type")
    axis.set_xlabel("Year")
    axis.set_ylabel("Number of paragraphs")
    
    #By default, the legend is in inverse order of the categories, so lets reverse
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], title='Corpora', loc='upper left', bbox_to_anchor=(1.03, 1))
    
    # Adjust layout to prevent legend cutoff
    fig.tight_layout()

    return results_df


def plot_normalised_unnormalised_subplots(
    kwds, corpus_type_names: Optional[list[str]] = None
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:  # type: ignore
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)

    df_unnorm = plot_country_map(
        kwds,
        corpus_type_names=corpus_type_names,
        normalize_counts=False,
        axis=ax1,
    )

    df_norm = plot_country_map(
        kwds,
        corpus_type_names=corpus_type_names,
        normalize_counts=True,
        axis=ax2,
    )

    return fig, df_unnorm, df_norm


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    db = load_data()
    world = load_world_geometries()
    num_paragraphs_in_db = get_num_paragraphs_in_db()
    
    # Sidebar controls
    st.sidebar.title("What data and graphs do you want?")
    
    # Corpus type selection
    corpus_type_names = get_unique_corpus_type_names()
    selected_corpus_types = st.sidebar.multiselect(
        "Select corpus types", corpus_type_names, default=corpus_type_names
    )


    # Year range selection
    min_year, max_year = get_year_range()
    selected_year_range = st.sidebar.slider(
        "Select year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    # Graph type selection
    st.sidebar.markdown("### Select visualizations")
    show_world_map = st.sidebar.checkbox("World Map", value=True)
    show_corpus_bar = st.sidebar.checkbox("Corpus Type Bar Chart", value=True)
    show_time_series = st.sidebar.checkbox("Corpus Time Series", value=True)
    show_time_relative = st.sidebar.checkbox("Relative time series", value=True)

    # Main content
    st.title("Data Explorer")
    st.markdown(
        "Explore the dataset by searching for keywords and viewing different visualizations."
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

        # Calculate total matches
        total_matches = get_geography_count_for_texts(kwds, selected_corpus_types, selected_year_range)["count"].sum()
        percentage = round(total_matches / num_paragraphs_in_db * 100, 2)
        st.markdown(f"Total matches: {total_matches:,} ({percentage}%)")

        # Display selected visualizations
        if show_world_map:
            #Calculate counts on whole dataset for the normalisation
            paragraph_count_by_geography = get_geography_count_for_texts([".*"], selected_corpus_types)
            
            st.markdown("## World Map")
            fig, data1, data2 = plot_normalised_unnormalised_subplots(kwds, selected_corpus_types)
            st.write(fig)

        if show_corpus_bar:
            st.markdown("## Corpus Type Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_corpus_type_bar(kwds, selected_corpus_types, selected_year_range, ax)
            st.write(fig)

        if show_time_series:
            st.markdown("## Time Series (absolute)")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_time_series(kwds, selected_corpus_types, selected_year_range, ax)
            st.write(fig)

        if show_time_relative:
            st.markdown("## Time Series (relative)")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_time_series(kwds, selected_corpus_types, selected_year_range, ax, relative = True)
            st.write(fig)

        # Individual keyword analysis
        if len(kwds) > 1:
            st.markdown("## Individual Keyword Analysis")
            for keyword in kwds:
                st.markdown(f"### {keyword}")
                
                if show_world_map:
                    fig, data1, data2 = plot_normalised_unnormalised_subplots(
                        [keyword], selected_corpus_types
                    )
                    st.write(fig)

                if show_corpus_bar:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_corpus_type_bar([keyword], selected_corpus_types, selected_year_range, ax)
                    st.write(fig)

                if show_time_series:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_time_series([keyword], selected_corpus_types, selected_year_range, ax)
                    st.write(fig)
                
                if show_time_relative:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_time_series([keyword], selected_corpus_types, selected_year_range, ax, relative = True)
                    st.write(fig)

