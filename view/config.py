import os

# This is for back-end configurations
# For interface configurations, use app.cfg instead

# __file__ refers to the current file
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

# v2t viewer
V2T_DATA_DIR = os.path.join(APP_STATIC, 'v2t')
V2T_IN_DIR = os.path.join(V2T_DATA_DIR, 'in')
V2T_MODEL_DIR = os.path.join(V2T_DATA_DIR, 'out/model')
V2T_METADATA_DIR = os.path.join(V2T_DATA_DIR, 'out/metadata')
V2T_FEAT_DIR = os.path.join(V2T_DATA_DIR, 'out/feature')

# Note: added `all` at the beginning -- totally 1 + 20 categories
MSRVTT_CATEGORIES = ['all', 'music', 'people', 'gaming', 'sports', 'news', 'education', 'tv', 'movie', 'animation', 'vehicles', 'howto', 'travel', 'science', 'animals', 'kids', 'documentary', 'food', 'cooking', 'beauty', 'ads']

DATASETS = ['msrvtt', 'yt2t', 'videonet', 'msrvtt5c', 'yt2t5c']