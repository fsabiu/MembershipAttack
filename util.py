from _collections import defaultdict
import dask.dataframe as dd
import mlflow
import numpy as np
import pandas as pd
import pickle
import sklearn
import sys
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense

path = '/mnt/dati/fsabiu/Code2'
#path = 'D:/Drive/Thesis/Code2'
sys.path.insert(0, path)

def encode_dataset(df, class_name):
    print("Encoding dataset ...")
    df = remove_missing_values(df)

    numeric_columns = get_numeric_columns(df)

    rdf = df

    df, feature_names, class_values = one_hot_encoding(df, class_name)

    real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)

    rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]

    features_map = get_features_map(feature_names, real_feature_names)

    return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map

def encode_Dask_dataset(data, class_name, dtypes, excluded_cols):
    
    cols_map = {}
    unique_values = {}
    categoric_colnames = []

    encoded_data = pd.DataFrame()

    for i, col in enumerate(data.columns): # Not numeric column

        if (dtypes[col] == 'object' and col not in excluded_cols): # If categorical column

            print("Binarizing column " + col + " - (" + str(i+1) + "/" + str(len(data.columns)) + ")")
            # Unique values
            unique_values[col] = data[col].compute().unique()
            columns = {}

            # Creating columns
            for value in unique_values[col]:
                columns[col + '-' + str(value)] = []
                categoric_colnames.append(col + '-' + str(value))

            cols_map[col] = columns

            for value in data[col].compute():
                for possible_value in unique_values[col]:
                    if (value == possible_value):
                        cols_map[col][col + '-' + str(possible_value)].append(1)
                    else:
                        cols_map[col][col + '-' + str(possible_value)].append(0)
        
        else: # If numeric column
            encoded_data[col] = data[col]

    for i, col in enumerate(cols_map.keys()): # original column
        print("Appending columns generated from " + col + " - (" + str(i+1) + "/" + len(cols_map.keys()) + ")")
        for new_col in cols_map[col].keys():
            print(new_col + "...")
            encoded_data[new_col] = cols_map[col][new_col]

    return encoded_data.compute()

def encode_split_dataset(dataset, class_name, dtypes, unique_values, excluded_cols):
    rows = 0
    chunk_size = 10000
    
    for index, data in enumerate(pd.read_csv('data/' + dataset + '/' + dataset +'.csv', dtype = dtypes, chunksize=chunk_size),start=1):
        rows += chunk_size
        print("Chunk " + str(index) + ": " + str(rows) + " rows")

        categoric_colnames = []
        cols_map = {}
        encoded_data = pd.DataFrame()

        for i, col in enumerate(data.columns): # Not numeric column
            if (dtypes[col] == 'object' and col not in excluded_cols): # If categorical column
                # Creating columns
                columns = {}

                for value in unique_values[col]:
                    columns[col + '-' + str(value)] = []
                    categoric_colnames.append(col + '-' + str(value))

                cols_map[col] = columns
                
                for value in data[col]:
                    for possible_value in unique_values[col]:
                        if (value == possible_value):
                            cols_map[col][col + '-' + str(possible_value)].append(1)
                        else:
                            cols_map[col][col + '-' + str(possible_value)].append(0)
            else: # if numeric column
                encoded_data[col] = data[col]
        
        for i, col in enumerate(cols_map.keys()): # original column
            print("Appending columns generated from " + col + " - (" + str(i+1) + "/" + str(len(cols_map.keys())) + ")")
            for new_col in cols_map[col].keys():
                print(new_col + "...")
                encoded_data[new_col] = cols_map[col][new_col]

        # Writing datasets
        print("Writing data...")
        data.to_csv('data/' + dataset + '/splitted/' + dataset + str(index) + '.csv')
        encoded_data.to_csv('data/' + dataset + '/encoded/' + dataset + str(index) + '.csv')
        
    return True

def get_features_map(feature_names, real_feature_names):
    features_map = defaultdict(dict)
    i = 0
    j = 0

    while i < len(feature_names) and j < len(real_feature_names):
        if feature_names[i] == real_feature_names[j]:
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1
        elif feature_names[i].startswith(real_feature_names[j]):
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1
    return features_map

def get_numeric_columns(df):
    """
    # Returns a list of numeric columns of the dataframe df

    # Parameters:
    #    df (Pandas dataframe): The dataframe from which extract the columns 

    # Returns:
    #    A list of columns names (strings) corresponding to numeric columns

    """
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns

def get_unique_df_values(dataset, dtypes):
    """
    # Returns a dictionary of type {dict: list}, corresponding to {columns -> unique values}

    # Parameters:
    #    df (Pandas dataframe): The dataframe from which extract the columns unique values
    #   dtypes (dictionary): Dictionary of column types
    
    # Returns:
    #    A dictionary of {columns -> unique values}

    """
    data = pd.read_csv(path + '/data/' + dataset + '/' + dataset +'.csv', dtype = dtypes)

    unique_values = {}

    for i, col in enumerate(data.columns): # Not numeric column

        if (dtypes[col] == 'object'): # If categorical column

            print("Saving values for column " + col + " - (" + str(i+1) + "/" + str(len(data.columns)) + ")")
            # Unique values
            unique_values[col] = data[col].unique()
    
    return unique_values

def get_real_feature_names(rdf, numeric_columns, class_name):
    if isinstance(class_name, list):
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
    else:
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
    return real_feature_names

def load_obj(file_path):
    with open(path + '/' + file_path + '.pkl', 'rb') as f:
        return pickle.load(f)

def make_report(model, params, metrics):
    
    # Starting MLFlow run
    mlflow.start_run(run_name=str(params))

    # Logging model parameters
    mlflow.log_params(params)

    # Logging model metrics
    mlflow.log_metrics(metrics)

    # Ending MLFlow run
    mlflow.end_run()

    return

def map_columns(data, class_name):
    # Dictionary of mapped columns
    newcols = {}
    colnames = list(data.columns)
    for i, col in enumerate(colnames):
        print("Mapping column: " + col + " ... (" + str(i+1) + "/" + str(len(colnames)) + ")")
        if(col == class_name or data[col].dtype == 'object'): #PRINC_SURG_PROC_CODE must be int64
            # Mapping dictionary
            mapDict = {'NaN': 0, 'nan': 0, 'X': 1, 'A':2}

            new_column = []
            append = new_column.append

            # Mapping values
            #for el in data[col].compute(): for dask dataframes
            for el in data[col]:
                if el not in mapDict.keys():   # If mapping is not defined  
                    # Define mapping
                    mapDict[el] = len(mapDict)
                
                # Map element    
                append(mapDict[el])

            # Appending mapped columns to dictionary   
            newcols[col] = new_column

    print("Number of columns: " + str(len(newcols)))

    for i, col in enumerate(newcols.keys()):
        print("Adding column: " + col + " ... (" + str(i+1) + "/" + str(len(newcols)) + ")")
        data[col] = pd.Series(newcols[col])    

    return data

def model_creation(hidden_layers, hidden_units, act_function, learning_rate, optimizer, size=None):
    model = tf.keras.models.Sequential()
    if size is not None:
        model.add(tf.keras.Input(shape=(size,)))

    for i in range(hidden_layers):
        model.add(Dense(hidden_units, activation = act_function))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )

    return model

def model_evaluation(modelType, model, X_test, y_test):

    metrics = {}
    y_pred = None

    if (modelType == 'NN'):
        y_pred = model.predict_classes(X_test)

    if (modelType == 'RF'):
        y_pred = model.predict(X_test)

    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Adding precision, recall, f1 and support for each class + overall <accuracy, macro avg, weighted avg>
    for class_name in report.keys():
        if(isinstance(report[class_name], dict)):
            for metric in report[class_name].keys():
                metrics['-'.join([str(class_name), metric])] = report[class_name][metric]
        else:
            metrics[class_name] = report[class_name]
    
    return metrics

def model_training(model, X_train, y_train, X_val, y_val, pool_size, batch_size, epochs, logdir):

    """earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode = "min", min_delta = 0.001, patience=10, restore_best_weights=True)"""

    model.fit(X_train, y_train, 
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (X_val, y_val),
        callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            # earlystop_callback
        ]
    )

    if False:
        model.save_weights(logdir+"final_model", save_format="h5")

    model.summary()

    return model

def one_hot_encoding(df, class_name):
    if not isinstance(class_name, list):
        dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        dfY = df[class_name].map(class_name_map)
        df = pd.concat([dfX, dfY], axis=1, join_axes=[dfX.index])
        feature_names = list(dfX.columns)
        class_values = sorted(class_name_map)
    else: # isinstance(class_name, list)
        dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
        # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
        class_values = sorted(class_name)
        dfY = df[class_values]
        df = pd.concat([dfX, dfY], axis=1, join_axes=[dfX.index])
        feature_names = list(dfX.columns)
    return df, feature_names, class_values

def prepare_dataset(dataset, explainer):

    data = None
    class_name = None

    if (dataset == 'adult' and explainer == 'lime'):
        data = pd.read_csv('data/' + dataset + '/' + dataset + '.csv')
        columns2remove = ['fnlwgt', 'education-num']
        data.drop(columns2remove, inplace=True, axis=1)
        data = data.drop_duplicates()
        class_name = 'class'

    if (dataset=='texas' and explainer == 'lime'):
        # Reading and processing files - done in file
        class_name = 'PRINC_SURG_PROC_CODE'
        years = ['2006', '2007', '2008', '2009']
        dfs = []
        
        """
        for year in years:
            data = pd.read_csv('data/' + dataset + '/hospital_texas_' + year + '_clean.csv')
            data.drop(['Unnamed: 0'], inplace=True, axis=1)
            dfs.append(data)
            print('Year ' + year + ' loaded')

        # Merging dataframes
        data = pd.concat(dfs, ignore_index= True)

        # Writing dataframe
        data.to_csv('data/' + dataset + '/' + dataset +'.csv', index=False)
        print('Full dataset written')
        """
        # Reading - done in file
        dtypes = load_obj('data/texas' + '/dtypes')
        data = pd.read_csv('data/' + dataset + '/' + dataset +'.csv', dtype = dtypes)
        
        types = dict()
        for col in data.columns:
            if(col in {'BLOOD_ADM_AMOUNT', 'PRIVATE_AMOUNT', 'SEMI_PRIVATE_AMOUNT', 'WARD_AMOUNT', 'ICU_AMOUNT', 'CCU_AMOUNT', 'OTHER_AMOUNT', 'PHARM_AMOUNT', 'MEDSURG_AMOUNT', 'DME_AMOUNT', 'USED_DME_AMOUNT', 'PT_AMOUNT', 'OT_AMOUNT', 'SPEECH_AMOUNT', 'IT_AMOUNT', 'BLOOD_AMOUNT', 'BLOOD_ADMIN_AMOUNT', 'OR_AMOUNT', 'LITH_AMOUNT', 'CARD_AMOUNT', 'ANES_AMOUNT', 'LAB_AMOUNT', 'RAD_AMOUNT', 'MRI_AMOUNT', 'OP_AMOUNT', 'ER_AMOUNT', 'AMBULANCE_AMOUNT', 'PRO_FEE_AMOUNT', 'ORGAN_AMOUNT', 'ESRD_AMOUNT', 'CLINIC_AMOUNT', 'TOTAL_CHARGES', 'TOTAL_NON_COV_CHARGES', 'TOTAL_CHARGES_ACCOMM', 'TOTAL_NON_COV_CHARGES_ACCOMM', 'TOTAL_CHARGES_ANCIL', 'TOTAL_NON_COV_CHARGES_ANCIL',}):
                types[col] = 'float32'
            else:
                types[col] = 'object'
        
        data.astype(types)

        """
        data = map_columns(data, class_name)
        
        # Writing dataset
        print("Writing dataframe")
        data.to_csv('data/' + dataset + '/' + dataset +'_mapped.csv', index=False)
        """
        return data, class_name

    if (dataset == 'texas_red' and explainer == 'lime'):
        class_name = 'PRINC_SURG_PROC_CODE'
        dtypes = load_obj('data/texas' + '/dtypes')
        
        # Reading primary external causes of injury, 
        data = pd.read_csv('data/' + 'texas' + '/' + 'texas' +'.csv', 
        usecols = ['E_CODE_1', 'ADMITTING_DIAGNOSIS', 'SEX_CODE', 'RACE' ,'PAT_AGE', 'LENGTH_OF_STAY','PRINC_SURG_PROC_CODE'],
        dtype=dtypes)

        print("Mapping columns...")
        data = map_columns(data, class_name)

        return data, class_name

    if (dataset== 'mobility' and explainer == 'lime'):
        data = pd.read_csv('data/' + dataset + '/' + dataset + '.csv', skipinitialspace=True, na_values='?', keep_default_na=True)
        
        # Removing columns        
        columns2remove = ['uid', 'wait']
        data.drop(columns2remove, inplace=True, axis=1)
        
        # Dropping duplicates
        data = data.drop_duplicates()
        class_name = 'class'

    # Shuffling data
    data = data.sample(frac=1).reset_index(drop=True)

    return data, class_name

def prepareNNdata(dataset, label):
    y = dataset[label].values
    dataset.drop([label], axis=1, inplace = True)

    cols = dataset.columns
    x = np.array(dataset[cols])

    return x, y

def prepareRFdata(dataset, label):
    y = dataset[label].values
    dataset.drop([label], axis=1, inplace = True)

    cols = dataset.columns
    x = np.array(dataset[cols])

    return x, y

def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                try:
                    mode = df[column_name].mode().values[0]
                    df[column_name].fillna(mode, inplace=True)
                except:
                    pass
    return df

def split(dataset, y):
    """
    # Returns a shuffled, stratified split of the input dataset

    # Parameters:
    #   dataset (Pandas dataframe): The dataframe from which extract the columns unique values
    #   y (string): name of the column representing the target, i.e. column on which stratify

    # Returns:
    #    A dictionary of {columns -> unique values}

    """
    # Shuffle data
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    ochenta, veinte = train_test_split(dataset, test_size = 0.2, random_state = 0, stratify = dataset[y])

    # 45% Black-Box data, 35% Attack data
    bb, att = train_test_split(ochenta, test_size = 0.45, random_state = 0, stratify = ochenta[y])

    # 85% Training Black-Box, 15% Validation Black-Box
    bb_train, bb_val = train_test_split(bb, test_size = 0.15, random_state = 0, stratify = bb[y])
    
    # 85% Training Shadow models, 15% Validation Shadow models
    sh_train, sh_val = train_test_split(att, test_size = 0.15, random_state = 0, stratify = att[y])

    # 10% records to explain, 10% test set
    r2E, test = train_test_split(veinte, test_size = 0.5, random_state = 0, stratify = veinte[y])

    return bb_train, bb_val, sh_train, sh_val, r2E, test

def save_obj(obj, file_path):
    with open(path + '/' + file_path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)