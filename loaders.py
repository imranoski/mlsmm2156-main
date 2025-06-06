# third parties imports
import pandas as pd

# local imports
from constants import Constant as C
from surprise import Reader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

reader = Reader(rating_scale=C.RATINGS_SCALE)

def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    df_ratings = df_ratings[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]]
    if surprise_format:
        return Dataset.load_from_df(df=df_ratings, reader=reader)
    else:
        return df_ratings


def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    df_items[C.YEAR] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)")
    df_items[C.YEAR] = pd.to_numeric(df_items[C.YEAR], errors='coerce').astype('Int64')
    df_items[C.GENRES_COL] = df_items[C.GENRES_COL].apply(lambda x: x.split("|"))
    return df_items

def load_visuals(mode = 'log'):
    if mode == 'log':
        df_visuals = pd.read_csv(C.VISUAL_PATH / C.VISUAL_LOG_FILENAME)
        print(C.VISUAL_PATH / C.VISUAL_LOG_FILENAME)
    elif mode == 'quantile':
        df_visuals = pd.read_csv(C.VISUAL_PATH / C.VISUAL_QUANTILE_FILENAME)
    elif mode == 'quantilelog':
        df_visuals = pd.read_csv(C.VISUAL_PATH / C.VISUAL_QUANTILELOG_FILENAME)
    df_visuals = df_visuals.set_index(C.VISUAL_MOVIE_ID)
    df_visuals.index.name = C.ITEM_ID_COL
    return df_visuals

def load_genome():
    import pandas as pd

    genome_scores = pd.read_csv(C.GENOME_PATH / C.GENOME_SCORES)  
    genome_tags = pd.read_csv(C.GENOME_PATH / C.GENOME_TAGS)     

    df = genome_scores.merge(genome_tags, on='tagId')

    df_genome = df.pivot_table(index='movieId', columns='tag', values='relevance', fill_value=0)

    return df_genome

def load_items_tfidf():
    genre_corpus = load_items()['genres'].apply(lambda x: ' '.join(x))

    tfidf = TfidfVectorizer()
    X_genres_tfidf = tfidf.fit_transform(genre_corpus)

    df_genres_tfidf = pd.DataFrame(
        X_genres_tfidf.toarray(),
        columns=tfidf.get_feature_names_out(),
        index=load_items().index
    )

    return df_genres_tfidf

def export_evaluation_report(df):
    df.to_csv(C.EVALUATION_PATH)
    """ 
    Export the report to the evaluation folder.

    The name of the report is versioned using today's date
    """
    return

#load_items_tfidf().to_csv('load_items_tfidf.csv')
print(load_items_tfidf())

#print(load_genome())
#load_genome().to_csv('load_genome.csv')