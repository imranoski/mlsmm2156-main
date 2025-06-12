# third parties imports
from pathlib import Path
from datetime import datetime

class Constant:

    DATA_PATH = Path('data/small')  # -- fill here the dataset size to use 
    DATA_PATH_PLUS = Path('data/hackathon/')
 
    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'
    GENRES_COL = 'genres'
    YEAR = 'year'

    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    RATINGS_SCALE = (0.5, 5.0)  # -- fill in here the ratings scale as a tuple (min_value, max_value)
    EVALUATION_PATH = f'{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

    # Visuals
    VISUAL_MOVIE_ID = 'ML_Id'
    VISUAL_PATH = DATA_PATH_PLUS / 'content' / 'visuals'
    VISUAL_LOG_FILENAME = 'LLVisualFeatures13K_Log.csv'
    VISUAL_QUANTILE_FILENAME = 'LLVisualFeatures13K_Quantile.csv'
    VISUAL_QUANTILELOG_FILENAME = 'LLVisualFeatures13K_QuantileLog.csv'

    # - metrics
    AVG_SHOT_LENGTH = 'f1'
    MEAN_COLOR_VAR = 'f2'
    STD_COLOR_VAR = 'f3'
    MEAN_MOT_AVG = 'f4'
    MEAN_MOT_STD = 'f5'
    MEAN_LIGHTING = 'f6'
    SHOT_NUM = 'f7'

    # Genome
    GENOME_PATH = DATA_PATH_PLUS / 'content'
    GENOME_TAGS = 'genome-tags.csv'
    GENOME_SCORES = 'genome-scores.csv'
    GENOME_TAG_ID = 'tagId'
    GENOME_TAG = 'tag'
    GENOME_REL = 'relevance'