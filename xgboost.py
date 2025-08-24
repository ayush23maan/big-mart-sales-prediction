# %%
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# %%
def preprocess_train_data(train_data_path): # include type hints later and 
    # clean up the return from docstring
    """
    Split train data into features and target
    csv file specified in path is read as pandas df.
    The columns are renamed. 
    
    Arguments
    _________
    train_data_path : system path for training csv file
    
    Returns
    _______
    Pandas dataframe
    """
    
    stores_sales = pd.read_csv(train_data_path)

    # rename target column to business for intuition. Increase the business.
    stores_sales.rename(columns={'Item_Outlet_Sales':'business'}, inplace=True)

    # parse target(buisness) and features(situations) for intuition
    situations = stores_sales.drop(columns=['business'])
    business = stores_sales['business']

    return situations, business

# %%
def get_cat_feature_names(situations):
    """
    Gives the list of categorical column names

    Returns
    _______
    list of names
    """
    # store column data types as pandas series
    columns_types = situations.dtypes

    # filter the series for object datatype using object string and get index. 
    # Then convert index to array
    return columns_types[columns_types == 'object'].index.values

# %%
def get_num_feature_names(situations):
    """
    Gives the list of numerical column names

    Returns
    _______
    list of names
    """
    # store column data types as pandas series
    columns_types = situations.dtypes

    # filter the series for object datatype using object string and get index. 
    # Then convert index to array
    return columns_types[columns_types != 'object'].index.values

# %%
def encode_cat_cols(situations, cat_cols_encoder_types):
    """
    Encodes categroical columns in situations.

    2. Stores the encoders in a dict against each column name. The column encoder is 
    retrieved for a col name and fit/trained on that column data in situations. Then, the  
    column is then encoded in situations. (Do I need to store the encoder? 
    Will it be used later on? What about the encoding of future dataset?)

    3. The new encoded columns are added to situations with new names.
    The encoded name to original name dict is maintained. It is returned for future use.
    
    Returns
    _______
    situations df with encoded columns
    enc_col_names_to_col_names_dict
    
    """

    # col encoders dictionary will store the encoders for each colum
    col_encoders_dict = {}

    # enc_col_names_to_col_names_dict will store the encoded column name
    # to original column name mapping 
    enc_col_names_to_col_names_dict = {}

    # Pick up cat column from cat_cols_encoder_types one at a time
    for col, encoder_type in cat_cols_encoder_types.items():
        encoded_col_name = f"{col}_encoded"

        # storing mapping from encoded columns 
        # to original columns
        enc_col_names_to_col_names_dict[encoded_col_name] = col

        # Creating encoder for each column and storing in dict
        if encoder_type == 'label':
            col_encoders_dict[col] = LabelEncoder()
            # fitting the encoders on cat column of situation and
            # encoding the columns in situations
            situations[encoded_col_name] = col_encoders_dict[col].fit_transform(situations[col])
        else:
            col_encoders_dict[col] = OneHotEncoder(sparse_output=False)
            # fitting the encoders on cat column of situation and
            # encoding the columns in situations
            encoded_array = col_encoders_dict[col].fit_transform(situations[[col]])
            # Add the one-hot encoded columns to the DataFrame
            for i, category in enumerate(col_encoders_dict[col].categories_[0]):
                situations[f"{encoded_col_name}_{category}"] = encoded_array[:, i]

    return situations, col_encoders_dict

# %%
def get_unique_cat_values_from_situations(situations):
    situations_cat_cols_names = get_cat_feature_names(situations)
    situations_cat_cols_values = {}

    for cat_col in situations_cat_cols_names:
        # get one column of the situation
        # store the unique values in a dict
        situations_cat_cols_values[cat_col] = situations[cat_col].unique()
        # return the dict to visualise it

    return situations_cat_cols_values

# %%
def clean_cat_columns(situations):
    """
    standardise the categories within catgeorical columns.
    Removes duplicate categories

    Returns
    _______
    df with categories cleaned
    """
    # replace 'Low Fat' 'low fat' with LF in Item_Fat_Content
    situations.loc[(situations['Item_Fat_Content'] == 'Low Fat')\
                    | (situations['Item_Fat_Content'] == 'low fat'),\
                    'Item_Fat_Content'] = 'LF'
    
    # replace 'Low Fat' 'low fat' with LF in Item_Fat_Content
    situations.loc[situations['Item_Fat_Content'] == 'Regular','Item_Fat_Content'] = 'reg'    

    return situations

# %% [markdown]
# ## Cleaning

# %%
train_data_path = r'/Users/yadav.a.1/Downloads/train_v9rqX0R.csv'
situations, business  = preprocess_train_data(train_data_path=train_data_path)

# %%
situations.dtypes

# %% [markdown]
# ### Clean up categorical columns. 
# First See the values

# %%
situations_cat_cols_values_dict = get_unique_cat_values_from_situations(situations)

# %% [markdown]
# standardise categoriies to remove duplicates
# Item_Fat_Content : combine 'Low Fat' 'low fat' 'LF' and 'Regular''reg'

# %%
situations = clean_cat_columns(situations)

# %% [markdown]
# ### Cleaning up numerical columns
# 'Item_Weight' 'Item_Visibility' 'Item_MRP' 'Outlet_Establishment_Year'

# %%
num_cols = get_num_feature_names(situations)

# %% [markdown]
# Fill missing values in Item_Weight

# %% [markdown]
#  1. We'll normalize the numerical columns like Item_Visibility and Item_MRP. Finally, we'll apply k-NN imputation. Ww will also use one hot encoded Item_Identifier_encoded, and Item_Type_encoded and label encoded Item_Fat_Content_encoded to estimate the missing values in Item_Weight, experimenting with different values of k to find the best fit.
# 
#  2. Use itme identifier to fill in weight. Yes, that's exactly right. You'll create a dictionary mapping item identifiers to their weights, then use that dictionary to fill in the missing weights based on the item identifier. It's a straightforward and efficient way to handle the missing data in this case.

# %%
# create dict with item identifiers as keys and real weights as values
# get item identifers with non missing weights
weights_df = situations[~situations['Item_Weight'].isna()][['Item_Identifier', 'Item_Weight']].drop_duplicates()
weights_dict = weights_df.set_index('Item_Identifier')['Item_Weight'].to_dict()

# create a list of item identifiers with missing weights.
missing_weights_items = situations[situations['Item_Weight'].isna()]['Item_Identifier'].drop_duplicates()

# check if they have weights in the dict
found_items = set(missing_weights_items).intersection(set(weights_dict))

# %% [markdown]
# fill the 1138 item weights and just fill the category mean for the reamining 4.
# The fillna method will just look up those IDs in the dict that are missing the weights. Then it will map the corrsponign weight from the dict 
# 

# %%
def fill_missing_item_weight(situations, weights_dict):
    situations['Item_Weight'].fillna(situations['Item_Identifier'].map(weights_dict), inplace=True)
    situations['Item_Weight'].fillna(situations['Item_Weight'].mean(), inplace=True)

    return situations

# %%
situations = fill_missing_item_weight(situations, weights_dict) 

# %% [markdown]
# Fill missing values in Outlet_Size

# %%
# create dict with item identifiers as keys and real weights as values
# get item identifers with non missing weights
weights_df = situations[~situations['Outlet_Size'].isna()][['Outlet_Identifier', 'Outlet_Size']].drop_duplicates()
weights_dict = weights_df.set_index('Outlet_Identifier')['Outlet_Size'].to_dict()

# create a list of item identifiers with missing weights.
missing_weights_items = situations[situations['Outlet_Size'].isna()]['Outlet_Identifier'].drop_duplicates()

# check if they have weights in the dict
found_items = set(missing_weights_items).intersection(set(weights_dict))

# %% [markdown]
# Cannot fill outlet size using a map. Unique outlets are missing size. What re thes eoutlet types? What are there years?
# 

# %% [markdown]
# That's correct. Outlet_Identifier won't be necessary for k-NN imputation, and Outlet_Establishment_Year should be normalized. We can use standardization or min-max scaling for that.

# %% [markdown]
# 1. with this kn imputation i want your advice so i'm thinking i'll create a new data set with just the outlet features except the outlet identifier and then I'll normalize or whatever you know scale the establishment here so how is that sounding to get started
# 2. 


# %%
outlets = situations[['Outlet_Identifier', 'Outlet_Size','Outlet_Location_Type', 
            'Outlet_Establishment_Year', 'Outlet_Type']].drop_duplicates()

oulets_missing_size = outlets[outlets['Outlet_Size'].isna()]
oulets_with_size = outlets[~outlets['Outlet_Size'].isna()]


# %% [markdown]
# OUT017 tier 2 outlet of supermarket type that opened in 2007.There is just one oulet OUT035, similar to that. Good thing is its very similar so Ill use this small size to impute. 
# 
# OUT045 tier 2 outlet of supermarket type that opened in 2002. Again using OUT035 size Small.
# 
# OUT010 is tier 3 grocery in 1998. No similar store. So, I will just medium. 

# %%
def fill_missing_outlet_size(situations):    
    situations.loc[situations['Outlet_Identifier'] == 'OUT017', 'Outlet_Size'] = 'Small'
    situations.loc[situations['Outlet_Identifier'] == 'OUT045', 'Outlet_Size'] = 'Small'
    situations.loc[situations['Outlet_Identifier'] == 'OUT010', 'Outlet_Size'] = 'Medium'

    return situations

# %%
situations = fill_missing_outlet_size(situations)


# %% [markdown]
# ## Outliers

# %% [markdown]
# ## Encoding categorical variables

# %%
# Define which columns are ordinal vs. nominal
# Identifiers are treated as ordinal here for simplicity, as one-hot encoding them would create too many features.
ordinal_cols = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Item_Identifier']
nominal_cols = ['Item_Type', 'Outlet_Type', 'Outlet_Identifier']

# Create the dictionary to specify encoder types
cat_cols_and_encoder_types = {col: 'label' for col in ordinal_cols}
cat_cols_and_encoder_types.update({col: 'onehot' for col in nominal_cols})


# %%
# get categorical features of situations
only_situations_cat_cols_before_encoding = get_cat_feature_names(situations)
situations, situations_cat_cols_encoders = encode_cat_cols(
                                              situations, cat_cols_and_encoder_types)


# %% [markdown]
# Every item sells in 1-10 stores. Most items sell around 5 number of stores.

# %% [markdown]
# column names after encoding

# %%
all_situations_columns_after_encoding = situations.columns.values

# Seperate columns that are not in raw cat columns. So, non-cat and encoded cat will be in features
only_situations_features_after_encoding = [col for col in all_situations_columns_after_encoding if col not in only_situations_cat_cols_before_encoding]

# %% [markdown]
# ## Modelling

# %%
situations_all_columns = situations.copy()
situations = situations[only_situations_features_after_encoding]

# %%
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search with 3-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(situations, business)

# Define the parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1, 5]
}

# Initialize the XGBoost Regressor
xgb = XGBRegressor(random_state=42)

# Perform Grid Search with 3-fold cross-validation
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# Fit the grid search to the data
grid_search_xgb.fit(situations, business)

# Define the parameter grid for LightGBM
param_grid_lgb = {
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the LightGBM Regressor
lgb = LGBMRegressor(random_state=42)

# Perform Grid Search with 3-fold cross-validation
grid_search_lgb = GridSearchCV(estimator=lgb, param_grid=param_grid_lgb, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# Fit the grid search to the data
grid_search_lgb.fit(situations, business)

trained_model = trained_model_rf



# %% [markdown]
# ## Prediction

# %% [markdown]
# 1. Load test data
# 2. clean categorical columns
# 3. encode cat columns
# 4. Mask out original cat_columns from dataset
# 5. generate prediction
# 6. Just keep desired columns and prediction column

# %% [markdown]
# load test data

# %%
# load test data
stores_assortments = pd.read_csv(r'/Users/yadav.a.1/Downloads/test_AbJTz2l.csv')

# %% [markdown]
# clean categorical columns

# %%
# clean categorical columns
stores_assortments = clean_cat_columns(stores_assortments)

# %% [markdown]
# Fill missing values

# %% [markdown]
# # Make functions os missing vale imputator of outrlet size and item weight and use here


# %%
outlets = stores_assortments[['Outlet_Identifier', 'Outlet_Size','Outlet_Location_Type', 
            'Outlet_Establishment_Year', 'Outlet_Type']].drop_duplicates()

oulets_missing_size = outlets[outlets['Outlet_Size'].isna()]
oulets_with_size = outlets[~outlets['Outlet_Size'].isna()]

# %% [markdown]
# same columsn as train. just use teh function

# %%
stores_assortments = fill_missing_item_weight(stores_assortments, weights_dict)
stores_assortments = fill_missing_outlet_size(stores_assortments)

# %% [markdown]
# encode categorical columns


# %%
# get categorical cols of stores_assortments
stores_assortments_cat_cols_names = get_cat_feature_names(stores_assortments)

# got to each cat col of stores_assortments
for col in stores_assortments_cat_cols_names:
    print("\n", col)

    if col in situations_cat_cols_encoders:
        encoder = situations_cat_cols_encoders[col]
        # Now, proceed with transformation
        if cat_cols_and_encoder_types[col] == 'label':
            enc_col = f"{col}_encoded"
            stores_assortments[enc_col] = encoder.transform(stores_assortments[col])
            print(f"Label encoded {col}")
        else: # OneHotEncoder
            enc_col = f"{col}_encoded"
            encoded_array = encoder.transform(stores_assortments[[col]])
            for i, category in enumerate(encoder.categories_[0]):
                stores_assortments[f"{enc_col}_{category}"] = encoded_array[:, i]
            print(f"OneHot encoded {col}")
    else:
        print(f"Column '{col}' was not in the training data encoders dictionary.")

# %% [markdown]
# Mask out original cat_columns from dataset

# %%
# stores_assortments has all columns. 
# stores_assortments_cat_cols_names has original cat column names
# get all column names now
all_columns_after_encoding = stores_assortments.columns.values

# remove column names that are in stores_assortments_cat_cols_names and get new column names
# Seperate columns that are not in raw cat columns. So, non-cat and encoded cat will be in features
only_features = [col for col in all_columns_after_encoding if col not in stores_assortments_cat_cols_names]

# mask is ready


# %% [markdown]
# generate prediction
# In your "generate prediction" section
train_features_columns = situations.columns.tolist() # The columns from your final training df
test_features_for_prediction = stores_assortments[train_features_columns]

# Now predict using the perfectly aligned dataframe
stores_assortments['Item_Outlet_Sales'] = trained_model.predict(test_features_for_prediction)


# %% [markdown]
# ## Submissions

# %% [markdown]
# Just keep desiredt columns

# %%
# use desired features list
final_cols = ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']
stores_assortments[final_cols].to_csv(r'/Users/yadav.a.1/Downloads/onehot_and_imputation_rf.csv', index=False)
