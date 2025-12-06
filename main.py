import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np



# Load Dataset
df = pd.read_csv('dataset/swiggy.csv')

# Row with name : NA are dropped because important feature like rating, rating_count, cost, cuisine, lic_no is also NA
df = df.dropna(subset=['name'])

# Converting cost to float
df["cost"] = df["cost"].replace('₹ ', "", regex=True).astype(float)

# Filling missing values in cost and cuisine columns
for col in ["cost", "cuisine"]:
     default_value = 0 if col == "cost" else "Unknown"
     df[col] = (
          df.groupby('name')[col]
          .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
          .fillna(default_value)
          )
     
# Converting rating to float
df["rating"]=df["rating"].replace('--',0).astype(float)

# Splitting city into area and main city
df[["area", "city_main"]] = df["city"].str.split(',', n=1, expand=True)
# cleaning area and city_main columns
df["area"] = df["area"].str.strip()
df["city_main"] = df["city_main"].str.strip()
# Filling missing values in city_main column
df['city_main'].fillna("Other", inplace=True)

# Drop city column as it's splitted into area and city_main
df.drop(columns=['city'], inplace=True)

# splitting cuisine into cuisine_1 and cuisine_2
df[["cuisine_1", "cuisine_2"]] = df["cuisine"].str.split(',', n=1, expand=True)

# Enconding cuisine_1 and cuisine_2 columns using OneHotEncoder and saving it as pickle file
oneHot_encoder_cuisine = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encode_oneHot_cuisine = oneHot_encoder_cuisine.fit_transform(df[["cuisine_1", "cuisine_2"]]) 
pickle.dump(oneHot_encoder_cuisine, open('pickles/oneHot_cuisine.pkl', 'wb'))
encoded_df_oneHot_cuisine = pd.DataFrame(encode_oneHot_cuisine, columns=oneHot_encoder_cuisine.get_feature_names_out(['cuisine_1', 'cuisine_2']))
df = pd.concat([df.reset_index(drop=True), encoded_df_oneHot_cuisine.reset_index(drop=True)], axis=1)
df.drop(columns=['cuisine_1', 'cuisine_2', 'cuisine'], inplace=True)

# Step 13:
# Enconding city_main column using OneHotEncoder and saving it as pickle file
oneHot_encoder_city = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encode_oneHot_city = oneHot_encoder_city.fit_transform(df[['city_main']])
pickle.dump(oneHot_encoder_city, open('pickles/oneHot_city.pkl', 'wb'))
encoded_df_oneHot_city = pd.DataFrame(encode_oneHot_city, columns=oneHot_encoder_city.get_feature_names_out(['city_main']))
df = pd.concat([df.reset_index(drop=True), encoded_df_oneHot_city.reset_index(drop=True)], axis=1)
df.drop(columns=['city_main'], inplace=True)

area_to_oneHot = False  # Set to False to use LabelEncoder instead of OneHotEncoder
if area_to_oneHot:
    # Encoding area column using OneHotEncoder and saving it as pickle file
    oneHot_encoder_area = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encode_oneHot_area = oneHot_encoder_area.fit_transform(df[['area']])
    pickle.dump(oneHot_encoder_area, open('pickles/oneHot_area.pkl', 'wb'))
    encoded_df_oneHot_area = pd.DataFrame(encode_oneHot_area, columns=oneHot_encoder_area.get_feature_names_out(['area']))
    df = pd.concat([df.reset_index(drop=True), encoded_df_oneHot_area.reset_index(drop=True)], axis=1)
    df.drop(columns=['area'], inplace=True)
else:
    # Encoding area column using LabelEncoder and saving it as pickle file
    label_encoder_area = LabelEncoder()
    df["area"]= label_encoder_area.fit_transform(df['area'])
    pickle.dump(label_encoder_area, open('pickles/label_area.pkl', 'wb'))

# Encoding name column using LabelEncoder and saving it as pickle file
# label_encoder_name = LabelEncoder()
# df["name"]=label_encoder_name.fit_transform(df["name"])
# pickle.dump(label_encoder_name, open('pickles/label_name.pkl', 'wb'))

# Encoding rating_count column using LabelEncoder and saving it as pickle file
label_encoder_rating_count = LabelEncoder()
df["rating_count"]=label_encoder_rating_count.fit_transform(df["rating_count"])
pickle.dump(label_encoder_rating_count, open('pickles/label_rating_count.pkl', 'wb'))

# Drop lic_no, address, menu and link columns as they are not impactfull for analysis
df.drop(columns=['lic_no','name', 'address', 'menu', 'link'], inplace=True)

# index id
df.set_index('id', inplace=True)

# clustering with optimal number of clusters
best_k = 6  # From elbow graph
kmean_model = KMeans(n_clusters=best_k).fit(df)
df['cluster'] = kmean_model.predict(df)
pickle.dump(kmean_model, open('pickles/kmeans_model.pkl', 'wb'))

print("Preprocessing and model training completed. Pickle files saved.")




from sklearn.metrics.pairwise import cosine_similarity

def recommend_restaurants(user_input, df, model,
                        #   label_encoder_name,
                          label_encoder_area,
                          label_encoder_rating_count,
                          oneHot_encoder_cuisine,
                          oneHot_encoder_city,
                          top_n=10):

    # -------------------------
    # Convert dict → DataFrame
    # -------------------------
    user_df = pd.DataFrame([user_input])

    # -------------------------
    # 1. NAME PLACEHOLDER FIX
    # -------------------------
    # If user does not enter any name (usually they won't)
    # we use a safe default name from training dataframe
    # safe_name = df['name'].iloc[0]   # first name in training data

    # Encode this name
    # enc_name = label_encoder_name.transform([safe_name]).reshape(1, -1)

    # -------------------------
    # 2. Encode numeric + label encoded values
    # -------------------------
    enc_rating = np.array([[user_df['rating'].iloc[0]]])
    enc_cost = np.array([[user_df['cost'].iloc[0]]])
    enc_rating_count = label_encoder_rating_count.transform(
        user_df['rating_count']
    ).reshape(1, -1)
    enc_area = label_encoder_area.transform(
        user_df['area']
    ).reshape(1, -1)

    # -------------------------
    # 3. OneHot encodings
    # -------------------------
    enc_cuisine = oneHot_encoder_cuisine.transform(
        user_df[['cuisine_1', 'cuisine_2']]
    )
    enc_city = oneHot_encoder_city.transform(
        user_df[['city_main']]
    )

    # -------------------------
    # 4. Build FINAL VECTOR (Exact feature order used in training)
    # -------------------------
    final_vector = np.hstack([
        # enc_name,           # feature_name (required)
        enc_rating,         # feature_rating
        enc_rating_count,   # feature_rating_count
        enc_cost,           # feature_cost
        enc_area,           # feature_area
        enc_cuisine,        # feature_cuisine_*
        enc_city            # feature_city_*
    ])

    # -------------------------
    # 5. Predict Cluster
    # -------------------------
    cluster = model.predict(final_vector)[0]
    cluster_df = df[df['cluster'] == cluster].copy()

    # -------------------------
    # 6. Cosine Similarity Ranking
    # -------------------------
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    cluster_vectors = cluster_df[feature_cols].values

    sim_scores = cosine_similarity(final_vector, cluster_vectors)[0]
    cluster_df["similarity_score"] = sim_scores

    # -------------------------
    # 7. Sort Top N Recommendations
    # -------------------------
    recommended = cluster_df.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    return recommended[[
        "area", "city",
        "rating", "rating_count", "cost",
        "cuisine_1", "cuisine_2",
        "similarity_score"
    ]]




user_input = {
    "rating": 4.0,
    "rating_count": "50+ ratings",
    "cost": 250,
    "area": "Indiranagar",
    "cuisine_1": "Biryani",
    "cuisine_2": "South Indian",
    "city_main": "Bangalore"
}

recommend_restaurants(
    user_input,
    df=df,
    model=kmean_model,
    # label_encoder_name=label_encoder_name,
    label_encoder_area=label_encoder_area,
    label_encoder_rating_count=label_encoder_rating_count,
    oneHot_encoder_cuisine=oneHot_encoder_cuisine,
    oneHot_encoder_city=oneHot_encoder_city,
    top_n=5
)