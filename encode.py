import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Encodes embeddings of text abstracts in a given dataframe and saves df and embeddings as a pickle

# Set file paths
input_file=r'c:\test\all_pp.csv' #a path to dataframe with the abstracts to encode
source_column='abstract_pp' #set a df column name to look for abstracts
id_column='eid' #set a df column with corresponding paper ids
output_file=r'c:\test\all_pp.pickle' #a path to store embeddings as a pickle object

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# The main func to do embedding
def encode_abstracts(model, abstracts):
    with tqdm(total=len(abstracts), desc='Encoding abstracts') as pbar_encode:
        encoded_abstracts = []
        for abstract in abstracts:
            encoded_abstract = model.encode([abstract], convert_to_tensor=True)[0]
            encoded_abstracts.append(encoded_abstract.cpu().numpy())
            pbar_encode.update(1)
    return encoded_abstracts

# Load dataframe
df=pd.read_csv(input_file, header=0, sep='\t', encoding='utf-8')
# Remove duplicates by id (not abstract!) and empty rows
df.dropna(subset=[source_column], inplace=True)
df.drop_duplicates(subset=[id_column], keep='first', inplace=True)
# Load abstracts to a list
preprocessed_abstracts = []
for abstract in df[source_column]:
    preprocessed_abstracts.append(abstract)

print (f'loaded file {input_file} with {len(preprocessed_abstracts)} abstracts')

# Encode 
encoded_abstracts = encode_abstracts(model, preprocessed_abstracts)
output_data=[df,encoded_abstracts]
with open(output_file, 'wb') as file:
    pickle.dump(output_data, file)
print ('saved dataframe (list item 0) and encoded abstracts (list item 1) as a list object to:',output_file)


