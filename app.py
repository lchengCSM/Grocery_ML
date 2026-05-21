import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="Houston Map Explorer", layout="wide")

st.title("🗺️ Houston Map Explorer")
st.markdown("Browse 100 Houston locations — drop a pin and promote it to a full data point.")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_excel("houston_points.xlsx", sheet_name="Houston Points")

base_df = load_data()

# --- Session state ---
if "extra_points" not in st.session_state:
    st.session_state.extra_points = []   # list of dicts matching df schema
if "pins" not in st.session_state:
    st.session_state.pins = []           # pending pins not yet promoted

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Map Settings")
    map_style = st.selectbox("Map Style", [
        "OpenStreetMap", "CartoDB Positron", "CartoDB DarkMatter"
    ])
    zoom_level = st.slider("Zoom", 2, 18, 11)

    st.divider()
    st.header("🔍 Filter Points")
    all_cats = sorted(base_df["Category"].unique())
    selected_cats = st.multiselect("Categories", all_cats, default=all_cats)
    rating_min, rating_max = float(base_df["Rating"].min()), float(base_df["Rating"].max())
    rating_range = st.slider("Rating", rating_min, rating_max, (rating_min, rating_max), step=0.1)

    st.divider()
    st.header("📌 Pending Pin Color")
    marker_color = st.selectbox("Color", ["red", "blue", "green", "purple", "orange", "darkblue", "pink"])

    st.divider()
    st.header("📊 Summary")
    st.metric("Spreadsheet Points", len(base_df))
    st.metric("Added Data Points", len(st.session_state.extra_points))
    st.metric("Pending Pins", len(st.session_state.pins))

tile_map = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Positron": "CartoDB positron",
    "CartoDB DarkMatter": "CartoDB dark_matter",
}
cat_colors = {
    "Restaurant": "orange",
    "Park": "green",
    "Hospital": "red",
    "School": "blue",
    "Shopping": "pink",
}

# Combine base + user-added data points
extra_df = pd.DataFrame(st.session_state.extra_points) if st.session_state.extra_points else pd.DataFrame(columns=base_df.columns)
df = pd.concat([base_df, extra_df], ignore_index=True)

# Filter
filtered = df[
    df["Category"].isin(selected_cats) &
    df["Rating"].between(rating_range[0], rating_range[1])
]

# --- Build map ---
m = folium.Map(location=[29.7604, -95.3698], zoom_start=zoom_level, tiles=tile_map[map_style])

# All data points (base + promoted)
for _, row in filtered.iterrows():
    color = cat_colors.get(row["Category"], "gray")
    is_added = int(row["ID"]) > len(base_df)
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8 if is_added else 7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9 if is_added else 0.75,
        weight=3 if is_added else 1,
        tooltip=f"{'✨ ' if is_added else ''}{row['Name']} | ⭐ {row['Rating']}",
        popup=folium.Popup(
            f"{'<b>✨ Added by you</b><br>' if is_added else ''}"
            f"<b>{row['Name']}</b><br>Category: {row['Category']}<br>"
            f"Rating: ⭐ {row['Rating']}<br>"
            f"({row['Latitude']:.5f}, {row['Longitude']:.5f})",
            max_width=220,
        ),
    ).add_to(m)

# Pending pins (not yet promoted)
for pin in st.session_state.pins:
    folium.Marker(
        location=[pin["lat"], pin["lon"]],
        popup=folium.Popup(f"📍 {pin['label']} (pending)", max_width=200),
        tooltip=f"📍 {pin['label']} (pending)",
        icon=folium.Icon(color=pin["color"], icon="map-marker", prefix="fa"),
    ).add_to(m)

# Legend
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.25);font-size:13px;">
  <b>Data Points</b><br>
  <span style="color:orange">●</span> Restaurant &nbsp;
  <span style="color:green">●</span> Park &nbsp;
  <span style="color:red">●</span> Hospital<br>
  <span style="color:blue">●</span> School &nbsp;
  <span style="color:pink">●</span> Shopping<br>
  <span style="font-size:11px;color:#888">✨ Bold ring = added by you</span><br><br>
  <b>Pending</b> 📍 Click map to place
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# --- Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.caption(
        f"Showing **{len(filtered)}** data points "
        f"({len(base_df)} original + {len(st.session_state.extra_points)} added) · "
        f"**{len(st.session_state.pins)}** pending pins"
    )
    map_data = st_folium(m, width="100%", height=580, returned_objects=["last_clicked"])

with col2:
    tab1, tab2 = st.tabs(["📋 Data Table", "📍 Pin & Promote"])

    with tab1:
        st.dataframe(
            filtered[["Name", "Category", "Rating"]].reset_index(drop=True),
            use_container_width=True,
            height=440,
        )
        st.download_button(
            "⬇️ Download CSV",
            data=filtered.to_csv(index=False),
            file_name="houston_points_all.csv",
            mime="text/csv",
        )

    with tab2:
        # Step 1: drop a pin via map click
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            lat = round(clicked["lat"], 5)
            lon = round(clicked["lng"], 5)
            st.info(f"**Clicked:** `{lat}, {lon}`")
            label = st.text_input("Pin label", value=f"Pin ({lat}, {lon})", key="pin_label")
            if st.button("📌 Drop pin here"):
                st.session_state.pins.append({"lat": lat, "lon": lon, "label": label, "color": marker_color})
                st.rerun()
        else:
            st.caption("Click the map to place a pin.")

        # Step 2: promote a pending pin to a data point
        if st.session_state.pins:
            st.divider()
            st.subheader("Promote to Data Point")
            st.caption("Fill in details to add a pin as a real data point on the map.")

            pin_labels = [p["label"] for p in st.session_state.pins]
            chosen_label = st.selectbox("Select pin", pin_labels, key="promote_select")
            chosen_pin = next(p for p in st.session_state.pins if p["label"] == chosen_label)

            new_name = st.text_input("Name", value=chosen_label, key="new_name")
            new_cat = st.selectbox("Category", all_cats, key="new_cat")
            new_rating = st.slider("Rating", 1.0, 5.0, 3.0, 0.1, key="new_rating")

            if st.button("✨ Add as data point"):
                new_id = len(base_df) + len(st.session_state.extra_points) + 1
                st.session_state.extra_points.append({
                    "ID": new_id,
                    "Name": new_name,
                    "Category": new_cat,
                    "Latitude": chosen_pin["lat"],
                    "Longitude": chosen_pin["lon"],
                    "Rating": new_rating,
                })
                st.session_state.pins = [p for p in st.session_state.pins if p["label"] != chosen_label]
                st.rerun()

            st.divider()
            st.caption("**All pending pins:**")
            for i, pin in enumerate(st.session_state.pins):
                with st.expander(f"📍 {pin['label']}", expanded=False):
                    st.write(f"Lat: {pin['lat']}  Lon: {pin['lon']}")
                    if st.button("🗑️ Remove", key=f"del_{i}"):
                        st.session_state.pins.pop(i)
                        st.rerun()

            if st.button("🧹 Clear all pins"):
                st.session_state.pins = []
                st.rerun()

        if st.session_state.extra_points:
            st.divider()
            st.caption(f"**{len(st.session_state.extra_points)} added data point(s):**")
            for i, pt in enumerate(st.session_state.extra_points):
                with st.expander(f"✨ {pt['Name']}", expanded=False):
                    st.write(f"Category: {pt['Category']}  Rating: ⭐ {pt['Rating']}")
                    st.write(f"Lat: {pt['Latitude']}  Lon: {pt['Longitude']}")
                    if st.button("🗑️ Remove", key=f"del_pt_{i}"):
                        st.session_state.extra_points.pop(i)
                        st.rerun()
            if st.button("🧹 Clear added points"):
                st.session_state.extra_points = []
                st.rerun()

st.divider()
st.caption("Click map → drop pin → fill details → promote to data point · Filters apply to all points")
