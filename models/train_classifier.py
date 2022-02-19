import sys
import re
import nltk
nltk.download(['punkt','wordnet','stopwords'])

import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''Load data from database. Separate the data into features and labels'''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('etlstaging_tbl', con=engine)
    X = df[['message']].values
    X = X.tolist()
    X = [x[0] for x in X]
    Y = df.drop(columns=['id','message','original','genre'], axis=1).values
    cat_names = df.drop(columns=['id','message','original','genre'], axis=1).columns.tolist()
    return X, Y, cat_names


def tokenize(text):
    '''Function to case normalize and tokenize text into list of tokens
    
    input: 
      text str
    
    output:
      clean_tokens list
    '''
    text = str(text)
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''Build a model with a vectorizer, a tfid transformer and
    a multi output classifier using the random forest classifier
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
        ])
    parameters = {
        'vect__ngram_range': ((1,1),(1,2)),
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 2500, 5000),
        'tfidf__use_idf': (True, False)    
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate the model'''
    y_pred = model.predict(X_test)
    
    for i in range(Y_test.shape[1]):
        print('Results for category: {}'.format(category_names[i]))              
        print(classification_report(Y_test[:,i],y_pred[:,i]))


def save_model(model, model_filepath):
    '''Save the model as a pickle file'''
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()