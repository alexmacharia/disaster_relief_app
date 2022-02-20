# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

The following files are included

 - data/process_data.py - This file loads csv files, cleans the data and stores it in a sqlite database
 - data/disaster_messages.csv - provides the source data for processing
 - data/disaster_categories.csv - provides the source data for processing
 - data/DisasterResponse.db - database created after cleaning the source data and stores the data in a table
 - models/train_classifier.py - reads data from the sqlite database and tranins a classifier model, stores the model in a pickle file
 - models/classifier.pkl - stores the model and is used for classification of new data
 - app/run.py - runs a flask app that visualizes data from the sqlite database and clasifies new input based on saved model
