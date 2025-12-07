# Import libraries
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# loading the dataset
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
df[["area", "city_main"]] = df["city"].str.rsplit(',', n=1, expand=True)

# df["city_main"] = df["city"].astype(str).str.split(',').str[1].str.strip()

# cleaning area and city_main columns
df["area"] = df["area"].str.strip()

df["city_main"] = df["city_main"].str.strip()
# Filling missing values in city_main column
df['city_main'].fillna("Other", inplace=True)

location = {}
for index, row in df.iterrows():
    city = row['city_main'].lower()
    area = row['area'].lower()
    if city not in location:
        location[city] = []
    if area not in location[city]:
        location[city].append(area)

pickle.dump(location, open('pickles/city_area.pkl', 'wb'))


df.drop(columns=['city'], inplace=True)

df[["cuisine_1", "cuisine_2"]] = df["cuisine"].str.split(',', n=1, expand=True).apply(lambda c: c.str.strip())

remove_cuisine = ["8:15 To 11:30 Pm","Attractive Combos Available","Code valid on bill over Rs.99","Combo","Default","Discount offer from Garden Cafe Express Kankurgachi","Free Delivery ! Limited Stocks!","Grocery products","MAX 2 Combos per Order!","Meat","Popular Brand Store","Special Discount from (Hotel Swagath)","SVANidhi Street Food Vendor","Use Code JUMBO30 to avail", "Use code XPRESS121 to avail.","Unknown"]
for cuisine in ["cuisine_1", "cuisine_2"]:
     df[cuisine] = df[cuisine].replace(remove_cuisine, "Other")
     df[cuisine] = df[cuisine].replace("Bakery products", "Bakery")
     df[cuisine] = df[cuisine].replace("BEVERAGE", "Beverages")
     df[cuisine] = df[cuisine].replace("Biryani - Shivaji Military Hotel", "Biryani")

pickle.dump(df, open('pickles/processed_df.pkl', 'wb'))

cuisine_types = list(set(df["cuisine_1"].dropna()).union(df["cuisine_2"].dropna()))
pickle.dump(cuisine_types, open('pickles/cuisines.pkl', 'wb'))

# Enconding cuisine_1 and cuisine_2 columns using OneHotEncoder and saving it as pickle file
oneHot_encoder_cuisine = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encode_oneHot_cuisine = oneHot_encoder_cuisine.fit_transform(df[["cuisine_1", "cuisine_2"]]) 
pickle.dump(oneHot_encoder_cuisine, open('pickles/oneHot_cuisine.pkl', 'wb'))
encoded_df_oneHot_cuisine = pd.DataFrame(encode_oneHot_cuisine, columns=oneHot_encoder_cuisine.get_feature_names_out(['cuisine_1', 'cuisine_2']))
df = pd.concat([df.reset_index(drop=True), encoded_df_oneHot_cuisine.reset_index(drop=True)], axis=1)
df.drop(columns=['cuisine_1', 'cuisine_2', 'cuisine'], inplace=True)

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
    print(df["area"])
    # Encoding area column using LabelEncoder and saving it as pickle file
    label_encoder_area = LabelEncoder()
    df["area"]= label_encoder_area.fit_transform(df['area'])
    pickle.dump(label_encoder_area, open('pickles/label_area.pkl', 'wb'))


# Encoding rating_count column using LabelEncoder and saving it as pickle file
pickle.dump(df["rating_count"], open('pickles/rating_counts.pkl', 'wb'))
label_encoder_rating_count = LabelEncoder()
df["rating_count"]=label_encoder_rating_count.fit_transform(df["rating_count"])
pickle.dump(label_encoder_rating_count, open('pickles/encoder_rating_count.pkl', 'wb'))

# Drop lic_no, address, menu and link columns as they are not impactfull for analysis
df.drop(columns=['name','lic_no', 'address', 'menu', 'link'], inplace=True)

# index id
df.set_index('id', inplace=True)
    
# clustering with optimal number of clusters
from sklearn.cluster import KMeans
best_k = 6  # From elbow graph
model = KMeans(n_clusters=best_k).fit(df)
df['cluster'] = model.predict(df)

pickle.dump(df, open('pickles/encoded_df.pkl', 'wb'))
pickle.dump(model, open('pickles/kmeans_model.pkl', 'wb'))