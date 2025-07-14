import streamlit as st
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import time

st.set_page_config(page_title="Rankd", layout="wide")
st.title("Rankd: Restaurant Ranking System")

# --- API Key and Input ---
api_key = st.text_input("Enter your Google API key", type="password")
city = st.text_input("Enter city or area (e.g. Madrid, Spain)", value="Madrid, Spain")
keywords_input = st.text_input("Enter search keywords (comma separated)", value="restaurants")
limit = st.slider("Number of restaurants to fetch", 10, 60, 30)

# --- Known Chains to Exclude ---
excluded_chains = [
    "Starbucks", "McDonald's", "Burger King", "TGB", "Domino's", "Telepizza",
    "KFC", "Vips", "Foster's", "Goiko", "Five Guys", "Papa John's", "Taco Bell"
]

def is_franchise(name):
    return any(chain.lower() in name.lower() for chain in excluded_chains)

def map_price_label(level):
    if level == 0 or pd.isna(level):
        return "❔ Unknown"
    elif level == 1:
        return "💸 Budget-Friendly"
    elif level == 2:
        return "💵 Mid-Range"
    elif level == 3:
        return "💳 Premium"
    elif level == 4:
        return "👑 High-End"
    else:
        return "❔ Unknown"

# --- Run search ---
if st.button("Fetch and Rank Restaurants"):
    with st.spinner("Calling Google Places API..."):

        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        query_keywords = " OR ".join(keywords)
        query = f"{query_keywords} in {city}"

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": api_key}

        places = []
        while len(places) < limit:
            res = requests.get(url, params=params).json()
            places.extend(res.get("results", []))

            if "next_page_token" in res:
                time.sleep(2)
                params["pagetoken"] = res["next_page_token"]
            else:
                break

        data = []
        for p in places[:limit * 2]:  # fetch more to allow for franchise filtering
            name = p.get("name", "")
            if is_franchise(name):
                continue  # Skip franchise
            place_id = p.get("place_id")

            # Get reviews and price_level using Place Details API
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {
                "place_id": place_id,
                "fields": "review,rating,user_ratings_total,price_level",
                "key": api_key
            }
            details_res = requests.get(details_url, params=details_params).json()
            result = details_res.get("result", {})
            reviews = result.get("reviews", [])
            price_level = result.get("price_level", None)

            # Compute sentiment score
            if reviews:
                sentiments = []
                for review in reviews:
                    text = review.get("text", "")
                    if text:
                        blob = TextBlob(text)
                        sentiments.append(blob.sentiment.polarity)
                sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0
            else:
                sentiment_score = 0  # Neutral if no reviews

            data.append({
                "name": name,
                "rating": p.get("rating"),
                "user_ratings_total": p.get("user_ratings_total"),
                "lat": p["geometry"]["location"]["lat"],
                "lng": p["geometry"]["location"]["lng"],
                "sentiment_score": sentiment_score,
                "price_level": price_level
            })

        df = pd.DataFrame(data).dropna()

        if df.empty:
            st.warning("No results found.")
        else:
            # Normalize and Rank
            scaler = MinMaxScaler()
            df[["rating_norm", "reviews_norm", "sentiment_norm"]] = scaler.fit_transform(
                df[["rating", "user_ratings_total", "sentiment_score"]]
            )
            df["rankd_score"] = (
                df["rating_norm"] * 0.5 +
                df["reviews_norm"] * 0.3 +
                df["sentiment_norm"] * 0.2
            )

            df = df.sort_values("rankd_score", ascending=False).head(limit).reset_index(drop=True)
            df["price_label"] = df["price_level"].apply(map_price_label)

            st.subheader("🏆 Top Restaurants by Rankd Score (with Sentiment & Franchise Filter)")
            for i, row in df.iterrows():
                st.markdown(f"""
                ### {row['name']}
                ⭐ {row['rating']} ({row['user_ratings_total']} reviews)  
                Sentiment: {row['sentiment_score']:.2f} · Score: {row['rankd_score']:.2f}  
                {row['price_label']}
                """)
                if row["rankd_score"] > 0.85 and row["price_level"] in [1, 2]:
                    st.markdown("💎 **Best Value Pick**")
                st.markdown("---")

            st.subheader("🗺️ Map View")
            st.map(df.rename(columns={"lat": "latitude", "lng": "longitude"}))
