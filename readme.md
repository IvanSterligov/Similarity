This is a pack of scripts to do semantic similarity search for academic papers using your own paper metadata. You can search using titles or abstracts, the latter work better.

The scripts are heavily commented and should be simple to understand. They are intended to be used as a basis for future projects and use pickle as data storage. Scripts work both as is and also as a simple flask API.

Semantic similarity seach is a powerful vectorized search meathod which can replace or complement traditional tf-idf\bm25 methods for large datasets. 

Here I implement it using a very popular https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 model. It is chosen because:

- It works quite fine, I like the results
- It is small and fast
- It is trained using Semantic Scholar corpus of academic papers (besides other datasets)
- It is used in production-scale semantic academic search like JSTOR

It is not intended to work with longer texts (see HF description)

I add simple preprocessing (see and modify *preprocess.py*):

- remove case
- remove 'academic' stopwords ('we present ','we use ','this chapter','the review','we define','we develop','we consider','we study' and the like, copyrights, 'abstract' etc)
- remove special chars and punctuation (could be important for chamistry, maths etc): 

```
text = re.sub(r'[^-A-Za-z0-9\s-]+', '', text)
```
- normalize dash
- lemmatize using nltk WordNetLemmatizer

**Step 1. Encoding**

- Collect your papers in a dataframe with at least one text column to search ('abstract' by default) and a paper id column ('eid' by default) and any number of other columns like 'source', 'doi', 'times_cited' and possibly 'author_id' to filter by author
- Preprocess papers using *preprocess.py* (adds a column 'abstract_pp' to your df)
- Encode df using *encode.py*

This gets you a pickle list object with metadata in [0] and embeddings in [1]

**Step 2. Seach**

- Open *find_similar.py* and edit path to the pickle yu've created
- Change column names if necessary

The script contains three functions which could be run directly from the file

**1 By paper ID**

```
find_similar_papers(paper_id, sim_score=0.6, exclude_authored=False, verbose=False)
```

This returns a list of dicts for sound similar papers.

If exclude_authored is set to True, all the papers authored by any author of paper you query are omitted. Useful to find reviewers or potential future coauthors.

I recommend similarity score of 0,6 to 0,7, lower scores tend to result in a lot of not-so-similar papers

**2 By text**
```
find_similar_papers_by_text(text, sim_score=0.6)
```
This is simple. Use any text you like, but longer texts get truncated ("By default, input text longer than 256 word pieces is truncated" by the model)

This returns a list of dicts for sound similar papers.

**3 Search by text and return a dataframe with 1 result per row**

```
def similar_to_table(text, score=0.6):
    output_dict=find_similar_papers_by_text(text,sim_score=score)
    if output_dict:
        for item in output_dict:
            item['text']=text    
    df=pd.DataFrame(output_dict)
    return df
```

**Step 3. API**

*api.py* adds a simple flask api that returns found results in JSON. Use it locally out of the box or on a [web-facing server](https://flask.palletsprojects.com/en/3.0.x/deploying/gunicorn/). 

```
@app.route('/api', methods=['GET'])
def api_id():
    if 'text' in request.args:
        text = str(request.args['text'])
    else:
        return "Error: No text field provided. Please specify query text"

    results=find_similar_papers_by_text(text)   
    return jsonify(results)
```

Local api query by default looks like this:

```
http://127.0.0.1:5000/api?text=foreign%20authors%20in%20russian%20academic%20journals
```

This script pack is ok for meduim text databases like 1 million of papers (~4 gb RAM needed). Bigger datasets need more RAM, so you have to rewrite the code, for example to put most part of paper metadata in external SQL DB and query your DB using ids of found papers to pull out necessary metadata after you run vector similarity search on their embeddings.

That's it for now :)




















