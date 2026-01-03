import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# --- 1. SETUP ---
# This loads your secret key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check if key is missing
if not api_key:
    st.error("⚠️ KEY MISSING! Please create a .env file and add GEMINI_API_KEY=your_key")
    st.stop()

# Connect to Google AI
genai.configure(api_key=api_key)

# --- 2. PAGE SETTINGS ---
st.set_page_config(page_title="CivicFix AI", page_icon="🏛️", layout="wide")

# --- 3. DATABASE (Fake Memory) ---
# We use this so data doesn't disappear when you click buttons
if "complaints" not in st.session_state:
    st.session_state["complaints"] = pd.DataFrame(columns=[
        "Description", "Category", "Urgency", "Status", "Language", "Summary"
    ])

# --- 4. THE AI BRAIN ---
def ask_gemini(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # This instructions tell the AI exactly what to do
    prompt = """
    Act as a Governance AI. Analyze this complaint.
    1. Translate it to English.
    2. Categorize it into ONE: [Roads, Sanitation, Electricity, Water, Police].
    3. Rate Urgency 1-10.
    4. Write a 5-word summary.
    
    Return ONLY JSON format like this:
    {"Translated": "text", "Category": "Roads", "Urgency": 8, "Summary": "pothole issue"}
    """
    
    try:
        response = model.generate_content([prompt, text])
        # Clean up the text to get just the JSON part
        clean_text = response.text.replace("```json", "").replace("```", "")
        return json.loads(clean_text)
    except:
        # If AI fails, return dummy data so app doesn't crash
        return {"Translated": text, "Category": "General", "Urgency": 5, "Summary": "Error"}

# --- 5. THE WEBSITE LAYOUT ---
# Sidebar to switch between User and Admin
page = st.sidebar.radio("Go to", ["📢 Citizen Portal", "👮 Admin Dashboard"])

if page == "📢 Citizen Portal":
    st.title("📢 Report a Problem")
    st.write("Speak or type. We accept Hindi, English, and Hinglish.")
    
    # Simple form
    user_input = st.text_area("Type your complaint here:", height=150)
    
    if st.button("🚀 Submit Complaint"):
        if user_input:
            with st.spinner("AI is analyzing..."):
                # Call the AI function
                data = ask_gemini(user_input)
                
                # Save the result
                new_row = {
                    "Description": user_input,
                    "Category": data["Category"],
                    "Urgency": data["Urgency"],
                    "Status": "Open",
                    "Language": "Detected",
                    "Summary": data["Summary"]
                }
                # Add to our list
                st.session_state["complaints"] = pd.concat(
                    [st.session_state["complaints"], pd.DataFrame([new_row])], 
                    ignore_index=True
                )
                
                st.success(f"✅ Sent to {data['Category']} Dept (Priority: {data['Urgency']}/10)")
                st.info(f"Summary: {data['Summary']}")
        else:
            st.warning("Please type something first!")

elif page == "👮 Admin Dashboard":
    st.title("👮 Mayor's Dashboard")
    
    df = st.session_state["complaints"]
    
    if not df.empty:
        # Top Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Issues", len(df))
        col2.metric("Critical Issues", len(df[df["Urgency"] > 7]))
        col3.metric("Most Busy Dept", df["Category"].mode()[0] if not df.empty else "N/A")
        
        st.divider()
        
        # Filter Dropdown
        dept = st.selectbox("Filter by Department:", ["All"] + list(df["Category"].unique()))
        
        # Show the table
        if dept != "All":
            show_df = df[df["Category"] == dept]
        else:
            show_df = df
            
        st.dataframe(show_df, use_container_width=True)
        
        # Simple Chart
        st.bar_chart(df["Category"].value_counts())
    else:
        st.info("No complaints yet. Go to the Citizen Portal to add one!")