# third parties imports
import pandas as pd

# local imports
from constants import Constant as C
from surprise import Reader, Dataset
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

#nltk.download('vader_lexicon') # uncomment only for the first initialization of the program
sia = SentimentIntensityAnalyzer()

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
    df_genome_tags = pd.read_csv(C.GENOME_PATH / C.GENOME_TAGS)
    df_genome_scores = pd.read_csv(C.GENOME_PATH / C.GENOME_SCORES)
    merged = df_genome_tags.merge(df_genome_scores, how = 'inner', on = C.GENOME_TAG_ID)
    '''
    merged = df_genome_tags.merge(df_genome_scores, how = 'inner', on = C.GENOME_TAG_ID)

    print('Calculating sentiment score...')
    unique_tags = df_genome_tags['tag'].drop_duplicates()

    tag_sentiment = {tag: sia.polarity_scores(tag)['compound'] for tag in unique_tags}
    merged['sentiment'] = merged['tag'].map(tag_sentiment)
    print('Calculating average sentiment...')
    grouped = merged.groupby(C.ITEM_ID_COL).agg({'sentiment': 'mean'}).reset_index()
    grouped 
    grouped.to_csv('grouped_test.csv')
    return grouped
    '''
    grouped = merged.groupby(C.ITEM_ID_COL).agg({'tag': list,}).reset_index()
    print(grouped)
    return grouped


def export_evaluation_report(df):
    df.to_csv(C.EVALUATION_PATH)
    """ Export the report to the evaluation folder.

    The name of the report is versioned using today's date
    """
    return

print(load_genome())