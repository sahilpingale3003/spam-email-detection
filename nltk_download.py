import nltk
import os

# Set NLTK data path to a directory that is accessible
# On Render, standard paths work, but explicit is good if we want to bundle it.
# However, standard download usually goes to /opt/render/project/src/nltk_data or home.
# We'll just use default which defaults to user home / hierarchy.
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
print("NLTK data downloaded.")
