import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import zipfile

zip_path = "pickles/pickles.zip"
extract_path = "pickles_unzip/"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)

# ---------------------
# Load Pickled Files
# ---------------------


with open("pickles_unzip/city_area.pkl", "rb") as f:
    city_area_dict = pickle.load(f)

with open("pickles_unzip/cuisines.pkl", "rb") as f:
    cuisines_list = pickle.load(f)

with open("pickles_unzip/rating_counts.pkl", "rb") as f:
    rating_list = pickle.load(f)

with open("pickles_unzip/oneHot_cuisine.pkl", "rb") as f:
    encoder_cuisine = pickle.load(f)

with open("pickles_unzip/oneHot_city.pkl", "rb") as f:
    encoder_city = pickle.load(f)

with open("pickles_unzip/label_area.pkl", "rb") as f:
    label_area = pickle.load(f)

with open("pickles_unzip/encoder_rating_count.pkl", "rb") as f:
    encoder_rating_count = pickle.load(f)

with open("pickles_unzip/kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("pickles_unzip/processed_df.pkl", "rb") as f:
    processed_df = pickle.load(f)

with open("pickles_unzip/encoded_df.pkl", "rb") as f:
    encoded_df = pickle.load(f)

# ---------------------
# Dummy Recommendation Function (replace later)
# ---------------------
def get_recommendations(city, area, cuisine, min_rating):
    # df = processed_df.copy()
    print(f"Inputs: city={city}, area={area}, cuisine={cuisine}, min_rating={min_rating}")
    user_df = pd.DataFrame([{
        "rating": 3,
        "rating_count": min_rating,
        "cost": processed_df['cost'].median(),
        "area": area.lower(),
        "cuisine_1": cuisine,
        "cuisine_2": cuisine,
        "city_main": city.capitalize()
    }])
    # -------------------------
    # 2. Encode numeric + label encoded values
    # -------------------------
    enc_rating = np.array([[user_df['rating'].iloc[0]]])
    enc_cost = np.array([[user_df['cost'].iloc[0]]])
    enc_rating_count = encoder_rating_count.transform(
        user_df['rating_count']
    ).reshape(1, -1)
    # enc_area = label_area.transform(
    #     user_df['area']
    # ).reshape(1, -1)
    # ---- FIX: Normalize area before encoding ----
    user_area = user_df['area'].iloc[0].lower().strip()

    # Make label_encoder classes lowercase
    label_area.classes_ = np.array([c.lower() for c in label_area.classes_])

    # Now encode safely
    enc_area = label_area.transform([user_area]).reshape(1, -1)

    # -------------------------
    # 3. OneHot encodings
    # -------------------------
    enc_cuisine = encoder_cuisine.transform(
        user_df[['cuisine_1', 'cuisine_2']]
    )
    enc_city = encoder_city.transform(
        user_df[['city_main']]
    )

    # -------------------------
    # 4. Build FINAL VECTOR (Exact feature order used in training)
    # -------------------------
    final_vector = np.hstack([
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
    cluster_df = encoded_df[encoded_df['cluster'] == cluster].copy()
    cluster_vector = cluster_df.drop(columns=['cluster']).values
    sim_scores = cosine_similarity(final_vector, cluster_vector)[0]
    cluster_df["similarity_score"] = sim_scores

    # cluster_df = encoded_df.copy()
    # cluster_vector = encoded_df.drop(columns=['cluster']).values
    # sim_scores = cosine_similarity(final_vector, cluster_vector)[0]
    # cluster_df["similarity_score"] = sim_scores
    

    recommendations_encoded = cluster_df.sort_values(by='similarity_score', ascending=False )
    recommendations=processed_df[processed_df['id'].isin(recommendations_encoded.index)].copy()
    recommendations["similarity_score"] = recommendations["id"].map(recommendations_encoded["similarity_score"])
    recommendations = recommendations.sort_values(
        by=['similarity_score', "city_main"],
        ascending=[False, True]
    )
    
    # print(recommendations.head(10))
    return recommendations.head(10)
    # return df


# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Restaurant Recommendation", layout="wide")

st.title("🍽️ Smart Restaurant Recommendation App")
st.write("Choose your preferences and get personalized restaurant suggestions.")

# CITY
city = st.selectbox("Select City", sorted(city_area_dict.keys()))

# AREA
area_list = city_area_dict.get(city, [])
area = st.selectbox("Select Area", ["Any"] + sorted(area_list))

# CUISINE
cuisine = st.selectbox("Select Cuisine", ["Any"] + sorted(cuisines_list))

# RATING
min_rating = st.select_slider(
    "Minimum Rating", 
    options=sorted(rating_list), 
    value=min(rating_list)
)

# ---------------------
# Sorting Options
# ---------------------
sort_option = st.radio(
    "Sort Recommendations By:",
    [
        "Defalut",
        "Rating: High to Low",
        "Cost: Low to High",
        "Cost: High to Low"
    ]
)

# ---------------------
# Submit Button
# ---------------------
if st.button("Get Recommendations"):
    st.subheader("Top Recommendations")

    results = get_recommendations(city, area, cuisine, min_rating)

    # APPLY SORTING
    if len(results) > 0:
        if sort_option == "Rating: High to Low":
            results = results.sort_values(by="rating", ascending=False)

        elif sort_option == "Cost: Low to High":
            results = results.sort_values(by="cost", ascending=True)

        elif sort_option == "Cost: High to Low":
            results = results.sort_values(by="cost", ascending=False)
    else:
        st.warning("No matching restaurants found.")
        st.stop()

    # DISPLAY RESULTS IN CARD FORMAT
    # for _, row in results.head(10).iterrows():
    #     with st.container():
    #         st.markdown(
    #             f"""
    #             <div style="
    #                 padding: 15px;
    #                 margin: 10px 0;
    #                 border-radius: 12px;
    #                 background: #f8f8f8;
    #                 border: 1px solid #ddd;">
                    
    #                 <h3 style="margin-bottom:5px;">{row.get('restaurant_name', '')}</h3>
    #                 <p><b>Area:</b> {row.get('area','')}</p>
    #                 <p><b>Cuisine:</b> {row.get('cuisines','')}</p>
    #                 <p><b>Rating:</b> ⭐ {row.get('rating','')}</p>
    #                 <p><b>Cost for Two:</b> ₹{row.get('cost','')}</p>
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )

    if results is not None and len(results) > 0:

        # Create a combined location column
        results["location"] = results["area"] + ", " + results["city_main"]

        # Select + rename columns for display
        display_df = results[[
            "name",
            "address",
            "location",
            "cuisine",
            "rating",
            "link"
        ]].rename(columns={
            "name": "Restaurant Name",
            "address": "Address",
            "location": "Location",
            "cuisine": "Cuisine",
            "rating": "Rating",
            "link": "Restaurant Link"
        })

        st.table(display_df)
