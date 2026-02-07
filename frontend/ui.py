# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time

import pandas as pd
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")  # Docker service name

st.set_page_config(page_title="NanoRecSys", page_icon="🎬", layout="wide")

# Load Movie Metadata (Title Lookup)
# We assume the volume is mounted at /data inside the container
MOVIES_PATH = "/app/data/ml-20m/movies.csv"


@st.cache_data
def load_movies():
    if os.path.exists(MOVIES_PATH):
        try:
            df = pd.read_csv(MOVIES_PATH)
            # Ensure unique index if duplicates exist
            return df.drop_duplicates("movieId").set_index("movieId")
        except Exception:
            return None
    return None


movies_df = load_movies()

# --- Sidebar ---
st.sidebar.title("Configuration")
api_status = st.sidebar.empty()


def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            api_status.success("API Online 🟢")
        else:
            api_status.warning(f"API Error {r.status_code} 🟡")
    except Exception:
        api_status.error("API Offline 🔴")


if st.sidebar.button("Check Connectivity"):
    check_api()


def format_title(title):
    """
    Moves articles (The, A, An) from the end of the title to the beginning.
    Ex: "Matrix, The (1999)" -> "The Matrix (1999)"
    """
    if not isinstance(title, str):
        return str(title)

    # Match "Title, The (Year)"
    match = re.match(r"^(.*), (The|A|An) \((\d{4})\)$", title)
    if match:
        name, article, year = match.groups()
        return f"{article} {name} ({year})"

    # Match "Title, The" (no year case)
    match_no_year = re.match(r"^(.*), (The|A|An)$", title)
    if match_no_year:
        name, article = match_no_year.groups()
        return f"{article} {name}"

    return title


def generate_movie_html_cards(ids, scores=None, explanations=None):
    """Generates HTML for a horizontal scrolling list of cards."""
    cards = []

    for i, mid in enumerate(ids):
        title = "Unknown Title"
        genres = "Unknown"

        if movies_df is not None and mid in movies_df.index:
            raw_title = movies_df.loc[mid, "title"]
            title = format_title(raw_title)
            genres = movies_df.loc[mid, "genres"]

        score_html = ""
        if scores:
            score_html = f"<div style='margin-top:5px; font-weight:bold; color:#4CAF50'>Score: {scores[i]:.4f}</div>"

        explanation_html = ""
        if explanations and i < len(explanations) and explanations[i]:
            explanation_html = f"<div style='margin-top:5px; font-size:0.8em; background:#e8f5e9; padding:5px; border-radius:3px;'>💡 {explanations[i]}</div>"

        rank_html = (
            f"<span style='background:#eee; padding:2px 6px; border-radius:4px; font-size:0.8em; margin-right:5px;'>#{i + 1}</span>"
            if scores
            else ""
        )

        # HTML formatted without indentation to avoid Markdown code block interpretation
        card_html = f"""
<div style="flex: 0 0 auto; min-width: 220px; max-width: 220px; border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; flex-direction: column; justify-content: space-between;">
    <div>
        <div style="font-weight:bold; margin-bottom:5px; height: 50px; overflow:hidden; text-overflow:ellipsis;">
            {rank_html}{title}
        </div>
        <div style="font-size:0.8em; color:#666; margin-bottom:5px; height: 40px; overflow:hidden;">
            {genres}
        </div>
        <div style="font-size:0.75em; color:#999;">ID: {mid}</div>
    </div>
    <div>
        {score_html}
        {explanation_html}
    </div>
</div>"""
        cards.append(card_html.strip().replace("\n", ""))

    # Join cards
    all_cards_html = "".join(cards)

    container_html = f"""
<div style="display: flex; flex-direction: row; overflow-x: auto; gap: 15px; padding: 10px 5px 20px 5px; align-items: stretch;">{all_cards_html}</div>"""
    return container_html


# --- Main Page ---
st.title("NanoRecSys: Real-Time Recommender")

# Input Section
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    user_id = st.number_input(
        "User ID",
        min_value=1,
        # max_value=200000,
        value=1,
        step=1,
        help="New user IDs can be any integer; existing users are in the range 1-138493.",
    )
with col2:
    top_k = st.slider("Top K", 1, 50, 10)
with col3:
    # explain_mode = st.toggle("Generate Explanations", value=False)
    # TODO: Replace with toggle when available
    explain_mode = False
    history_mode = st.toggle("Show User History", value=False)
    # st.caption("Enables slower features")

# Action
if st.button("Generate Recommendations", type="primary", use_container_width=True):
    start_time = time.time()

    with st.spinner("Retrieving & Ranking..."):
        try:
            payload = {
                "user_id": user_id,
                "k": top_k,
                "explain": explain_mode,
                "include_history": history_mode,
            }
            response = requests.post(f"{API_URL}/recommend", json=payload, timeout=5)

            if response.status_code != 200:
                st.error(f"API Error: {response.text}")
            else:
                data = response.json()
                movie_ids = data.get("movie_ids") or []
                scores = data.get("scores") or []
                explanations = data.get("explanations") or []
                timings = data.get("debug_timing") or {}
                history_ids = data.get("history") or []

                # --- Results List (Horizontal) ---
                st.subheader(f"Top {len(movie_ids)} Recommendations")

                # Render using HTML for true horizontal scroll
                html_recs = generate_movie_html_cards(
                    movie_ids,
                    scores=scores,
                    explanations=explanations if explain_mode else None,
                )
                st.markdown(html_recs, unsafe_allow_html=True)

                # --- User History (Horizontal) ---
                if (
                    history_mode
                ):  # Remove "and history_ids" check to show message if empty
                    st.divider()
                    st.subheader(f"User History (Last {len(history_ids)} items)")

                    if not history_ids:
                        st.info(
                            "No history found for this user (or history loading failed)."
                        )
                    else:
                        # Render HTML
                        html_hist = generate_movie_html_cards(history_ids)
                        st.markdown(html_hist, unsafe_allow_html=True)

                # --- Latency Stats ---
                if timings:
                    st.subheader("System Latency")
                    t_cols = st.columns(4)
                    t_cols[0].metric("Total", f"{timings.get('total', 0):.1f} ms")
                    t_cols[1].metric(
                        "Embedding", f"{timings.get('embedding', 0):.1f} ms"
                    )
                    t_cols[2].metric(
                        "Retrieval", f"{timings.get('retrieval', 0):.1f} ms"
                    )
                    t_cols[3].metric("Ranking", f"{timings.get('ranking', 0):.1f} ms")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API. Is the Docker service running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")
