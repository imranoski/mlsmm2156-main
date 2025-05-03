# third parties imports
import pandas as pd

# local imports
from constants import Constant as C
from surprise import Reader, Dataset

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
    return df_items


def export_evaluation_report(df):
    df.to_csv(C.EVALUATION_PATH)
    """ Export the report to the evaluation folder.

    The name of the report is versioned using today's date
    """
    return