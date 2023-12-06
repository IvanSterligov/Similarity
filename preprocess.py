import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from tqdm import tqdm
tqdm.pandas()

#Defines a function to preprocess texts and applies it to paper abstracts in a dataframe 

input_file=r'c:\test\all.csv' #a dataframe with the columnn with abstracts
output_file=r'c:\test\all_pp.csv' #where to store results
source_columnn='abstract' #the name of the column in input file containing texts to be preprocessed
target_column='abstract_pp' #the name of the column to store preprocessed texts in the output file

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

if __name__ == "__main__":

    df=pd.read_csv(input_file, header=0, sep=';', encoding='utf-8')

    print (f'loaded file: {input_file}')

    df['abstract_pp']=df['abstract'].progress_apply(preprocess_text)

    df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')

    print (f'exported to file: {output_file}')