##### Adam Mickiewicz University, faculty of mathematics and computer science
##### author: Mateusz Czajka
##### class: recommendation systems

# Content based hotel room recommender
The purpose of this project was to:
 1. Preprocess data
 2. Prepare user features
 3. Prepare item features
 4. Create fit method for recommender
 5. Create recommend method for recommender 
 6. Tune several different ML models (LinearRegression, SVR, RandomForest, XGBoostCBUI) to find the best parameters for each of them
 7. Run the final evaluation of recommender and present its results against the Amazon recommender's results

# Requirements
    python 3
    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install seaborn
    pip install scikit-learn
    pip install hyperopt
    
    

# Dataset
Original dataset was preprocessed to facilitate the work with data.

## Original dataset
 - The original dataset: [hotel-recommender](https://github.com/C7A7A/hotel-recommender)/[data](https://github.com/C7A7A/hotel-recommender/tree/main/data)/[hotel_data](https://github.com/C7A7A/hotel-recommender/tree/main/data/hotel_data)/**hotel_data_original.csv**
 
 Part of **original dataset**
 
![Original dataset image](/screenshots/original_dataset.png?raw=true "Original dataset")


## Preprocessed dataset

 - Preprocessed dataset: [hotel-recommender](https://github.com/C7A7A/hotel-recommender)/[data](https://github.com/C7A7A/hotel-recommender/tree/main/data)/[hotel_data](https://github.com/C7A7A/hotel-recommender/tree/main/data/hotel_data)/**hotel_data_preprocessed.csv**

 - Preprocessed dataset with **interactions** between **users** and **items** and selected data: [hotel-recommender](https://github.com/C7A7A/hotel-recommender)/[data](https://github.com/C7A7A/hotel-recommender/tree/main/data)/[hotel_data](https://github.com/C7A7A/hotel-recommender/tree/main/data/hotel_data)/**hotel_data_interactions_df.csv**

Part of **interactions preprocessed dataset**

![Preprocessed dataset image](/screenshots/preprocessed_dataset.png?raw=true "Preprocessed dataset")

# User Features
I've decided to find **a vector** of the most **popular feature values** for **each user** and encode every feature with **one-hot encoding**

---
group users and get most common value for each feature

    users_df = users_df.groupby('user_id').agg(get_most_common).reset_index()
    
get most common value function

    def get_most_common(srs):
	    x = list(srs)
	    my_counter = Counter(x)
	    return my_counter.most_common(1)[0][0]

Solution with **Counter** is about 5 times faster than **value_counts** and a bit faster than 
**Series.mode**, although it's still pretty slow

---
**One-hot encoding** using **get_dummies**

    def encode_and_bind(original_dataframe, feature_to_encode):
	    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
	    res = pd.concat([original_dataframe, dummies], axis=1)
	    res = res.drop([feature_to_encode], axis=1)
	    return res, list(dummies.columns)

Part of **users_df**:

![Users dataframe Image](/screenshots/users_df.png?raw=true "Users dataframe")

# Item Features
Item features were designed the same way as user features (**one-hot encoding** using **get_dummies)**

Part of **items_df**:

![Items dataframe image](/screenshots/items_df.png?raw=true "Items dataframe")

# Content based recommender

## fit

    def fit(self, interactions_df, users_df, items_df):

**Training of the recommender**.
 -  interactions_df: DataFrame with recorded interactions between users and items defined by user_id, item_id and features of the interaction
 - users_df: DataFrame with users and their features defined by user_id and the user feature columns
 - items_df: DataFrame with items and their features defined by item_id and the item feature
---
generating **negative interactions**

    def rand_correct_negative_interactions(interactions_df, rng, negative_interactions_size):
	    negative_interactions = []
	    
	    max_user = interactions_df['user_id'].max()
	    max_item = interactions_df['item_id'].max()

	    # print(max_user, max_item)
	    
	    positive_interactions = interactions_df[['user_id', 'item_id']].to_numpy()
	    
	    while (len(negative_interactions) < negative_interactions_size):
	        user_id = rng.choice(max_user + 1)
	        item_id = rng.choice(max_item + 1)
	        
	        negative_interaction = tuple((user_id, item_id))
	        
	        if not (negative_interaction == positive_interactions).all(axis=1).any():
	            negative_interactions.append(tuple((user_id, item_id, 0)))
 
    return negative_interactions

## recommend

    def recommend(self, users_df, items_df, n_recommendations=1):

**Serving of recommendations**. Scores items in items_df for each user in users_df and returns top n_recommendations for each user.

 - users_df: DataFrame with users and their features for which recommendations should be generated
 - items_df: DataFrame with items and their features which should be scored
 - n_recommendations: Number of recommendations to be returned for each user
returns DataFrame with user_id, item_id and score as columns returning n_recommendations top recommendations for each user

---
**new users** are treated as **average user** in dataset

    users_df = pd.merge(
            users_df, 
            self.users_df, 
            on='user_id', 
            how='left'
	   ).fillna(value=avg_user) 

**average user** representation

    avg_user = {
            'user_term_Christmas': 0, 'user_term_Easter': 0, 'user_term_HighSeason': 0, 	  'user_term_LowSeason': 0, 
            'user_term_MayLongWeekend': 0, 'user_term_NewYear': 0, 'user_term_OffSeason': 1, 
            'user_term_WinterVacation': 0, 
            'user_length_of_stay_bucket_[0-1]': 0, 'user_length_of_stay_bucket_[2-3]': 1, 
            'user_length_of_stay_bucket_[4-7]': 0, 'user_length_of_stay_bucket_[8-inf]': 0, 
            'user_rate_plan_Nonref': 1, 'user_rate_plan_Standard': 0, 
            'user_room_segment_[0-160]': 1, 'user_room_segment_[160-260]': 0, 
            'user_room_segment_[260-360]': 0, 'user_room_segment_[360-500]': 0, 
            'user_room_segment_[500-900]': 0, 
            'user_n_people_bucket_[1-1]': 0, 'user_n_people_bucket_[2-2]': 0, 
            'user_n_people_bucket_[3-4]': 1, 'user_n_people_bucket_[5-inf]': 0, 
            'user_weekend_stay_False': 0, 'user_weekend_stay_True': 1
        }

---
Sample **recommendations**

![Recommendations image](/screenshots/recommendations.png?raw=true "Recommendations")

# Results
Recommender **results** with different **parameters**

![Results image](/screenshots/results.png?raw=true "Results")

I've been able to **beat Amazon recommender** with 2 ML models (**LinearRegression** and **RandomForest**)

**Linear Regression** params

    n_neg_per_pos = 6

**RandomForest** params

	max_depth = 30 
	min_samples_split = 30 
    n_estimators = 100
    n_neg_per_pos = 7

**n_ner_per_pos** is used to determine how many **negative interactions** will be generated. 
