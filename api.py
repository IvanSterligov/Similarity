import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from tqdm import tqdm
import flask
from flask import request, jsonify
tqdm.pandas()

#set stopwords for filtering
stop_words = set(stopwords.words('english'))
# And also some specific academia stopwords:
stop_addon={'paper','study','present','aim','propose','purpose','chapter','background','article','introduction','investigate','consider', 'discuss','analyze','analyze','report','address','explores','examine','examines','explored','investigated','introduce','contribution'}
stop_words.update(stop_addon)

#set raw text filtering (not words, but exact text using regex)

subs=[r'©.*',r'abstract:',r'^abstract',r'copyright.*','we present ','we use ','this chapter','the review','we define','we develop','we consider','we study ', 'we analyse', 'we analyze', 'we review', 'we report', 'we provide', 'we prove', 'we propose', 'we investigate', 'we discuss','we show','this research', 'this paper','this article','the study','the purpose','the paper']

def preprocess_text(text):

    # Skip empty texts and the like
    if not isinstance(text, str):
        return text
    
    # Lowercase the text
    text = text.lower()

    # Remove raw strings with regex match ('abstract:' etc)

    for sub in subs:
        text = re.sub(sub, '', text)
    
    # normalise dash
    text = re.sub(r'[–—‐]', '-', text)

    # Remove special characters and punctuation - could be detrimental for chemistry and the like
    text = re.sub(r'[^-A-Za-z0-9\s-]+', '', text)

    # Remove all numbers except years (e.g., 2021)
    # text = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+\b', '', text) #useful for humanities
    
    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    words = text.split()

    # Exclude specific words from lemmatization
    preserved_words = ['discuss', 'has', 'assess']  # Add more words if needed
    words = [lemmatizer.lemmatize(word) if word not in preserved_words else word for word in words]

    # Remove stopwords
    text = ' '.join(word for word in words if word not in stop_words)

    return text

pickle_path=r'c:\test\all_pp.pickle' # path to pickled list with df and embeddings
source_column='abstract_pp' #set a df column name to look for abstracts
id_column='eid' #point to a df column with corresponding paper ids
# these are optional, only needed if you filter papers of the same authors who wrote the paper you query via id:
author_id_column='authors_ids' #point to a df column 
author_id_delimiter='; ' #specify author id delimiter

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load df and embeddings
with open(pickle_path, 'rb') as file:
    obj = pickle.load(file)
df=obj[0] #a table with publications metadata
df.fillna('', inplace=True) #change empty dois to '' to stop breaking json output with NaN
encoded_abstracts=obj[1] #encoded paper abstracts
print (f'loaded dataframe from {pickle_path} with {len(encoded_abstracts)} abstract embeddings')

app = flask.Flask(__name__)
app.config["DEBUG"] = True #remove for production


#this function finds papers not by ID, but by text (any text)
def find_similar_papers_by_text(text, sim_score=0.6):

    # Locate and encode the abstract we are searching similars for
    encoded_paper_abstract = model.encode([preprocess_text(text)], convert_to_tensor=True)[0]

    # Compute the cosine similarity between the paper and other abstracts

    similarity_scores = cosine_similarity(
        encoded_paper_abstract.cpu().numpy().reshape(1, -1), np.vstack(encoded_abstracts)
    )[0]

    # Create a list of dictionaries containing the similar paper information
    
    similar_papers = []
    for i, score in enumerate(similarity_scores):
        if score > sim_score and score < 0.98:  #Removes the paper itself from results. also somehow the random papers with very high similarity keep showing up... this filters them out
            selected_row = df.iloc[i]
            row_dict = selected_row.to_dict()
            row_dict['similarity_score']=score.item() #try to convert numpy float to regular float
            similar_papers.append(row_dict)
            similar_papers.sort(key=lambda x: x['similarity_score'], reverse=True) #Sort list of papers by sim score
            for x in similar_papers: 
                x['doi']
    return similar_papers

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Similarity Search</h1>
<p>A prototype API for semantic similarity search for academic papers</p>'''

@app.route('/api', methods=['GET'])
def api_id():
    if 'text' in request.args:
        text = str(request.args['text'])
    else:
        return "Error: No text field provided. Please specify query text"

    results=find_similar_papers_by_text(text)   
    return jsonify(results)

app.run()
