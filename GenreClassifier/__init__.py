"""
The flask application package.
"""

from flask import Flask
import gc
collected = gc.collect()
app = Flask(__name__, static_folder='static')

import GenreClassifier.views
