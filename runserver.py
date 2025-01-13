"""
This script runs the GenreClassifier application using a development server.
"""

from os import environ
import tracemalloc
from GenreClassifier import app
from GenreClassifier.GenreClassifierNN import *
if __name__ == '__main__':
    HOST = '0.0.0.0'
    try:
        PORT = int(environ.get('SERVER_PORT', '8501'))
    except ValueError:
        PORT = 8501
    tracemalloc.start()
    app.run(HOST, PORT,debug=True)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    for stat in top_stats[:10]:
        print(stat)

