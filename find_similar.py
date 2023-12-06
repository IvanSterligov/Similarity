import pandas as pd
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from preprocess import preprocess_text

# A function that takes an id of a paper and returns a list of dicts with similar papers from a table (column names become dict keys)
# We need a pickle object - a list with dataframe with papers to search through (item 0) and embeddings of abstracts in this dataframe (item 1)

pickle_path=r'c:\ivan\experts\all_pp.pickle' # path to pickled list with df and embeddings
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
df=obj[0]
encoded_abstracts=obj[1]
print (f'loaded dataframe from {pickle_path} with {len(encoded_abstracts)} abstract embeddings')

# The main function finds the paper by id in the df. adjust similarity score as needed (0)
def find_similar_papers(paper_id, sim_score=0.6, exclude_authored=False, verbose=False):

    #Print the paper data with the queried ID:
    row = df[df[id_column] == paper_id]
    # Check if the row was found
    if not row.empty:
        # Print the metadata of queried paper
        if verbose:
            print ('queried paper:')
            print(row.iloc[0], '\n')  
    else:
        print (f'no paper with id {paper_id} found')
        return []

    # Locate and encode the abstract we are searching similars for
    preprocessed_paper_abstract = row[source_column].values[0]    
    encoded_paper_abstract = model.encode([preprocessed_paper_abstract], convert_to_tensor=True)[0]

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
            row_dict['similarity_score']=score
            similar_papers.append(row_dict)
            if exclude_authored: # Remove papers with the same authors as the queried paper
                source_authors = df[df[id_column] == paper_id][author_id_column].values[0].split(author_id_delimiter)
                source_authors = [x.strip() for x in source_authors]
                for paper in similar_papers:
                    paper_authors = paper[author_id_column].split(author_id_delimiter)
                    paper_authors = [x.strip() for x in paper_authors]
                    if set(source_authors).intersection(set(paper_authors)):
                        similar_papers.remove(paper)
            similar_papers.sort(key=lambda x: x['similarity_score'], reverse=True) #Sort list of papers by sim score

    return similar_papers

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
            row_dict['similarity_score']=score
            similar_papers.append(row_dict)
            similar_papers.sort(key=lambda x: x['similarity_score'], reverse=True) #Sort list of papers by sim score
    return similar_papers

# this script returns the results as a dataframe (one row per similar paper, id=text queried)

def similar_to_table(text, score=0.6):
    output_dict=find_similar_papers_by_text(text,sim_score=score)
    if output_dict:
        for item in output_dict:
            item['text']=text    
    df=pd.DataFrame(output_dict)
    return df

# Here we can run the funcs to test it

'''

#export to excel

if __name__ == "__main__":
    text='In the 2022, corporate and government sector of the Russian Federation were restricted access to the most developed capital markets of Europe and North America regions. The demand of funding of the sectors can be partially satisfied by the national capital market. In order to find the cost of debt of the firm different simple methods are often used. Either the individual parameters of solvency by a company are not included in the methods or estimation of the risk premium for a company based on the developed capital market research. The main point of the article is to examine the practical oriented problem of cost of debt estimation. The research is also relevant according to the fundamental scientific side. Examining isolated Russian capital market, the relevant fundamental scientific problem of the influence of credit rating on the cost of debt is revealed. The 4 regression models are used for the research. The data include 2 populations: median credit spreads of the bond issuers on the local capital market for the last 6 months and their ratings, credit spreads of the bond issuers from 2020 to 2022 and their ratings. Due to the heterogeneity of the data and significant structural changes in the period of time, dynamic models have low explanatory power, but allow us to test the hypotheses posed in this paper.'
    x = similar_to_table(text)
    outfile=r'c:\ivan\experts\test_cf.xlsx'
    x.to_excel(outfile, index=False)
    print (f'exported results to {outfile}')
    sys.exit()

'''

'''

#find papers by text

if __name__ == "__main__":
    similar_papers = find_similar_papers_by_text(text='This paper presents a study of authors writing articles in the field of SNA and groups them by means of bibliographic network analysis. The dataset consists of works from the Web of Science database obtained by searching for “social network*”, works highly cited in the field, works published in the flagship SNA journals, and written by the most prolific authors (70,000+ publications and 93,000+ authors), up to and including 2018. Using a two-mode network linking publications with authors, we constructed and analysed different types of collaboration networks among authors. We used the temporal quantities approach to trace the development of these networks through time. The results show that most articles are written by 2 or 3 authors. The number of single authored papers has dropped significantly since the 1980s—from 70% to about 10%. The analysis of three types of co-authorship networks allowed us to extract the groups of authors with the largest number of co-authored works and the highest collaborative input, and to calculate the indices of collaborativeness. We looked at the temporal properties of the most popular nodes. We faced the problem of “multiple personalities” of mostly Chinese and Korean authors, which could be overcome with the adoption of standardized author IDs by publishers and bibliographic databases.', sim_score=0.6)

    for item in similar_papers:
        for key, value in item.items():
            print(key,":", value)
        print('---')
    sys.exit()
'''

'''

#find papers by id in your database

if __name__ == "__main__":
    similar_papers = find_similar_papers(paper_id='2-s2.0-85123921037', sim_score=0.6, exclude_authored=True)
    #similar_papers = find_similar_papers_by_text(text=r'We investigate the interplay between open access (OA), coauthorship, and international research collaboration. Although previous research has dealt with these factors separately, there is a knowledge gap in how these interact within a single data set. The data includes all Scopus-indexed journal articles published over 11 years (2009–2019) where at least one of the authors has an affiliation to a United Arab Emirates institution (30,400 articles in total). To assess the OA status of articles, we utilized Unpaywall data for articles with a digital object identifier, and manual web searches for articles without. There was consistently strong growth in publication volume counts as well as shares of OA articles across the years. The analysis provides statistically significant results supporting a positive relationship between a higher number of coauthors (in particular international) and the OA status of articles. Further research is needed to investigate potentially explaining factors for the relationship between coauthorship and increased OA rate, such as implementation of national science policy initiatives, varying availability of funding for OA publishing in different countries, patterns in adoption of various OA types in different coauthorship constellations, and potentially unique discipline-specific patterns as they relate to coauthorship and OA rate.', sim_score=0.6)

    for item in similar_papers:
        for key, value in item.items():
            print(key,":", value)
        print('---')
'''