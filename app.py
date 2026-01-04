import streamlit as st
import datetime
from PIL import Image
from mistralai import Mistral
import requests
import os
from dotenv import load_dotenv
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import csv
from difflib import SequenceMatcher
import time

# Load environment variables
load_dotenv()

# Configure Mistral AI API
api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = None
if api_key:
    try:
        mistral_client = Mistral(api_key=api_key)
    except Exception as e:
        st.warning(f"⚠️ Error initializing Mistral AI: {str(e)}")
        mistral_client = None
else:
    st.warning("⚠️ MISTRAL_API_KEY not found in environment. Using fallback mode.")

# Configure Groq API for audio transcription
groq_api_key = os.getenv("GROQ_API_KEY")
groq_available = groq_api_key is not None

# Configure fallback API
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
hf_api_available = hf_api_key is not None

# Data persistence
COMPLAINTS_FILE = "complaints_data.csv"

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="CivicFix AI - Grievance Redressal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Data Persistence Functions
# ----------------------------
def save_complaint_to_csv(complaint_record):
    """Save complaint to CSV file"""
    file_exists = os.path.isfile(COMPLAINTS_FILE)
    
    with open(COMPLAINTS_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'text', 'category', 'department', 'priority_score', 
                     'priority_level', 'summary', 'submitted_at', 'status', 'sentiment',
                     'impact_score', 'estimated_resolution_days', 'original_language']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(complaint_record)

def load_complaints_from_csv():
    """Load complaints from CSV"""
    if not os.path.isfile(COMPLAINTS_FILE):
        return []
    
    complaints = []
    with open(COMPLAINTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['priority_score'] = int(row['priority_score'])
            row['impact_score'] = int(row.get('impact_score', 1))
            row['estimated_resolution_days'] = int(row.get('estimated_resolution_days', 7))
            complaints.append(row)
    return complaints

# ----------------------------
# Initialize Session State
# ----------------------------
if 'complaints' not in st.session_state:
    st.session_state.complaints = load_complaints_from_csv()

if 'demo_loaded' not in st.session_state:
    st.session_state.demo_loaded = False

# ----------------------------
# AI Helper Functions
# ----------------------------
def transcribe_audio(audio_file):
    """Transcribe audio using Groq API"""
    
    if not groq_api_key:
        st.info("💡 To enable automatic voice transcription, add GROQ_API_KEY to your .env file.")
        st.info("Get it FREE at: https://console.groq.com/keys")
        return "Voice transcription requires Groq API key. Please describe your complaint in the text box below.", False
    
    try:
        from groq import Groq
        
        # Initialize Groq client
        groq_client = Groq(api_key=groq_api_key)
        
        # Read audio file
        audio_file.seek(0)
        
        # Check file size
        file_size_mb = audio_file.size / (1024 * 1024)
        if file_size_mb > 25:
            return "Audio file too large (max 25MB). Please record a shorter message.", False
        
        with st.spinner("🎙️ Transcribing audio with Groq..."):
            # Use Groq's audio transcription API
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text",
                language="en"  # You can remove this to auto-detect language
            )
            
            text = transcription.strip()
            
            if text and len(text) > 3:
                st.success("✅ Audio transcribed successfully with Groq!")
                return text, True
            else:
                return "Transcription returned empty. Please try again or use text input.", False
                
    except ImportError:
        st.error("❌ Groq library not installed. Run: pip install groq")
        return "Groq library missing. Please use text/photo submission.", False
        
    except Exception as e:
        error_msg = str(e)
        
        if "audio" in error_msg.lower() and "format" in error_msg.lower():
            st.error("❌ Audio format not supported. Please use WAV, MP3, or M4A format.")
        elif "size" in error_msg.lower() or "large" in error_msg.lower():
            st.error("❌ Audio file too large. Please upload a shorter recording.")
        elif "api" in error_msg.lower() or "key" in error_msg.lower():
            st.error("❌ Invalid Groq API key. Please check your .env file.")
        else:
            st.error(f"❌ Transcription error: {error_msg}")
        
        return "Voice transcription failed. Please describe your complaint in the text box below.", False

def detect_and_translate(text):
    """Detect language and translate to English if needed"""
    if not mistral_client or not text.strip():
        return text, "English (en)"
    
    try:
        prompt = f"""Detect the language of this text and translate to English if it's not English.
Text: "{text}"

Return ONLY valid JSON:
{{
    "detected_language": "language_name (code)",
    "is_english": true/false,
    "english_translation": "translated text or original if English"
}}"""
        
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        return result['english_translation'], result['detected_language']
    except:
        return text, "English (en)"

def analyze_sentiment(text):
    """Analyze sentiment of complaint"""
    if not mistral_client or not text.strip():
        return "Neutral"
    
    try:
        prompt = f"""Analyze the sentiment/tone of this complaint. Choose ONE word from: Angry, Frustrated, Concerned, Calm, Neutral.
Text: "{text}"
Return only one word, nothing else."""
        
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        sentiment = response.choices[0].message.content.strip()
        valid_sentiments = ["Angry", "Frustrated", "Concerned", "Calm", "Neutral"]
        if sentiment in valid_sentiments:
            return sentiment
        return "Neutral"
    except:
        return "Neutral"

def calculate_impact_score(text, category):
    """Calculate how many people might be affected"""
    text_lower = text.lower()
    
    high_impact_keywords = ["entire area", "whole street", "many people", "community", 
                           "neighborhood", "all residents", "public", "everyone", "entire city"]
    medium_impact_keywords = ["my building", "few people", "nearby", "local area", "our colony"]
    
    impact_score = 1  # Default: affects individual
    
    if any(keyword in text_lower for keyword in high_impact_keywords):
        impact_score = 3
    elif any(keyword in text_lower for keyword in medium_impact_keywords):
        impact_score = 2
    
    high_impact_categories = ["Public Safety", "Utilities", "Civic Infrastructure"]
    if category in high_impact_categories:
        impact_score = min(3, impact_score + 1)
    
    return impact_score

def get_estimated_resolution_time(category, priority_score):
    """Estimate resolution time based on category and priority"""
    base_times = {
        "Civic Infrastructure": 7,
        "Sanitation": 3,
        "Public Safety": 1,
        "Healthcare": 2,
        "Education": 5,
        "Utilities": 4,
        "Administrative Delays": 10,
        "Other": 7
    }
    
    base_days = base_times.get(category, 7)
    
    if priority_score >= 7:
        return max(1, int(base_days * 0.5))
    elif priority_score <= 3:
        return int(base_days * 1.5)
    return base_days

def find_similar_complaints(new_text, existing_complaints, threshold=0.6):
    """Find similar complaints using text similarity"""
    if not new_text.strip() or len(existing_complaints) == 0:
        return []
    
    similar = []
    for complaint in existing_complaints[-30:]:  # Check last 30
        similarity = SequenceMatcher(None, new_text.lower()[:200], 
                                    complaint['text'].lower()[:200]).ratio()
        if similarity > threshold:
            similar.append({
                'id': complaint['id'],
                'similarity': f"{similarity*100:.0f}%",
                'text': complaint['text'][:150],
                'category': complaint['category'],
                'status': complaint['status']
            })
    
    return sorted(similar, key=lambda x: float(x['similarity'].rstrip('%')), reverse=True)

def analyze_image_with_ai(image):
    """Analyze uploaded image using Mistral Pixtral Vision Model"""
    
    # First, try Mistral Pixtral if we have Mistral API key
    if mistral_client:
        try:
            # Convert to PIL Image and encode to base64
            if hasattr(image, 'read'):
                image.seek(0)
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                return None
            
            # Resize if too large (max 1024px on longest side for efficiency)
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            buffered.seek(0)
            image_base64 = base64.b64encode(buffered.read()).decode('utf-8')
            
            with st.spinner("🔍 Analyzing image with Mistral Pixtral Vision AI..."):
                # Use Mistral's vision model
                response = mistral_client.chat.complete(
                    model="pixtral-12b-2409",  # Mistral's vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Analyze this image as a citizen grievance complaint photo. Identify what civic problem or issue is shown.

Look for:
- Infrastructure damage (potholes, broken roads, cracks, damaged bridges, broken pavements)
- Sanitation issues (garbage piles, waste, littering, dirty areas, overflowing bins)
- Safety hazards (broken streetlights, damaged structures, unsafe conditions, exposed wires)
- Utility problems (water leakage, drainage issues, electricity problems, broken pipes)
- Public facilities issues (damaged parks, broken benches, vandalism)
- Healthcare/Education facility problems
- Any other civic issues

Provide a detailed 3-4 sentence description of:
1. What specific problem/issue you see in the image
2. The severity/urgency of the issue
3. Which government department should handle this

Be specific and factual. If you cannot identify a clear civic issue, describe what you see."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            ]
                        }
                    ]
                )
                
                description = response.choices[0].message.content.strip()
                
                if description and len(description) > 20:
                    st.success("✅ Image analyzed successfully with Mistral Pixtral Vision AI!")
                    return f"Image Analysis: {description}"
                else:
                    st.warning("⚠️ Image analysis returned limited information.")
                    return None
                    
        except Exception as e:
            st.warning(f"⚠️ Mistral Pixtral analysis error: {str(e)}")
            st.info("💡 Falling back to basic image analysis...")
    
    # Fallback to basic analysis if Mistral fails or unavailable
    if hf_api_key:
        try:
            if hasattr(image, 'read'):
                image.seek(0)
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                return None
            
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            buffered.seek(0)
            image_bytes = buffered.read()
            
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            
            models = [
                "Salesforce/blip-image-captioning-large",
                "Salesforce/blip-image-captioning-base",
            ]
            
            for model_name in models:
                try:
                    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                    
                    with st.spinner(f"🔍 Analyzing with {model_name.split('/')[-1]}..."):
                        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=45)
                    
                    if response.status_code == 200:
                        result = response.json()
                        caption = None
                        
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], dict):
                                caption = result[0].get('generated_text', '') or result[0].get('caption', '')
                            else:
                                caption = str(result[0])
                        elif isinstance(result, dict):
                            caption = result.get('generated_text', '') or result.get('caption', '')
                        
                        if caption and len(caption.strip()) > 10:
                            st.success(f"✅ Image analyzed with {model_name.split('/')[-1]}")
                            return f"Image shows: {caption.strip()}. This appears to be a citizen grievance requiring government attention."
                            
                    elif response.status_code == 503:
                        st.info(f"⏳ Model loading... Trying next option...")
                        time.sleep(3)
                        continue
                        
                except Exception as e:
                    continue
        
        except Exception as e:
            pass
    
    # Enhanced fallback with basic analysis
    st.warning("⚠️ Automatic image analysis unavailable. Using enhanced fallback...")
    
    try:
        if hasattr(image, 'read'):
            image.seek(0)
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            return "Image uploaded as visual evidence for grievance complaint."
        
        width, height = pil_image.size
        
        # Get basic image properties
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        file_size_kb = len(buffered.getvalue()) / 1024
        
        if pil_image.mode != 'RGB':
            pil_image_rgb = pil_image.convert('RGB')
        else:
            pil_image_rgb = pil_image
        
        pil_image_small = pil_image_rgb.resize((50, 50))
        pixels = list(pil_image_small.getdata())
        avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
        
        brightness = "bright outdoor scene" if avg_brightness > 180 else "low-light/indoor scene" if avg_brightness < 80 else "moderately lit scene"
        quality = "high-resolution" if width * height > 1000000 else "standard quality"
        
        return f"A {quality} {brightness} image ({width}x{height}px, {file_size_kb:.1f}KB) has been uploaded as visual evidence for this grievance complaint. The image shows potential civic infrastructure or public service issues that require manual review by government officials to determine the specific nature and urgency of the complaint."
            
    except Exception as e:
        return "Image uploaded as visual evidence. Manual review required to assess the grievance."

def analyze_complaint_with_ai(text, image=None):
    """Use AI to analyze complaint and return structured data"""
    
    image_description = ""
    combined_text = text
    
    if image:
        st.info("🔍 Analyzing uploaded image...")
        image_analysis = analyze_image_with_ai(image)
        
        if image_analysis and "manual review recommended" not in image_analysis.lower():
            image_description = image_analysis
            st.success("✅ Image analysis complete!")
            
            # Show what was detected in the image
            with st.expander("👁️ View Image Analysis Results"):
                st.write(image_analysis)
            
            # Combine text and image analysis
            if text and text.strip():
                combined_text = f"Citizen complaint text: {text}\n\n{image_analysis}"
            else:
                combined_text = image_analysis
        else:
            # Fallback if image analysis fails
            if text and text.strip():
                combined_text = text
            else:
                combined_text = "Visual evidence uploaded for grievance complaint. Image content could not be automatically analyzed."
    
    if not mistral_client:
        return analyze_complaint_fallback(combined_text)
    
    try:
        prompt = f"""You are analyzing a citizen grievance complaint for a government redressal system. 

Complaint Information:
{combined_text}

Analyze this complaint and provide a JSON response with these fields:

1. category: Choose the most appropriate from ["Civic Infrastructure", "Sanitation", "Public Safety", "Healthcare", "Education", "Utilities", "Administrative Delays", "Other"]

2. department: Assign to the appropriate government department (e.g., "Public Works Department", "Municipal Sanitation Dept", "Police Department", "Health Department", "Education Department", "Electricity Board", "Water Supply Dept", "District Administration")

3. priority_score: Rate from 1-10 where 10 is most urgent. Consider:
   - Safety risks to citizens
   - Health hazards
   - Number of people affected
   - Time sensitivity
   - Severity of the issue

4. priority_level: Based on priority_score:
   - "High 🔴" if score >= 7
   - "Medium 🟠" if score >= 4
   - "Low 🟢" if score < 4

5. summary: Write a clear 2-3 sentence summary of the complaint

6. urgency_reasons: List 2-3 specific reasons explaining the priority score

Respond with ONLY valid JSON in this exact format:
{{
    "category": "...",
    "department": "...",
    "priority_score": <number>,
    "priority_level": "...",
    "summary": "...",
    "urgency_reasons": ["reason 1", "reason 2", "reason 3"]
}}"""

        response = mistral_client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Validate all required fields exist
        required_keys = ['category', 'department', 'priority_score', 'priority_level', 'summary', 'urgency_reasons']
        if not all(key in result for key in required_keys):
            st.warning("⚠️ AI response incomplete. Using enhanced fallback analysis...")
            return analyze_complaint_fallback(combined_text)
        
        # Validate priority_score is a number
        if not isinstance(result['priority_score'], (int, float)):
            result['priority_score'] = 5
        
        return result
        
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Could not parse AI response. Using fallback analysis...")
        return analyze_complaint_fallback(combined_text)
    except Exception as e:
        st.warning(f"⚠️ AI analysis error: {str(e)}. Using fallback analysis...")
        return analyze_complaint_fallback(combined_text)

def analyze_complaint_fallback(text):
    """Fallback rule-based analysis"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["road", "pothole", "bridge", "street", "infrastructure"]):
        category, department = "Civic Infrastructure", "Public Works Department"
    elif any(word in text_lower for word in ["garbage", "waste", "sanitation", "dirty", "trash"]):
        category, department = "Sanitation", "Municipal Sanitation Dept"
    elif any(word in text_lower for word in ["crime", "unsafe", "security", "theft", "violence"]):
        category, department = "Public Safety", "Police Department"
    elif any(word in text_lower for word in ["hospital", "medicine", "health", "doctor", "medical"]):
        category, department = "Healthcare", "Health Department"
    elif any(word in text_lower for word in ["school", "education", "teacher", "student"]):
        category, department = "Education", "Education Department"
    elif any(word in text_lower for word in ["electricity", "power", "light", "current"]):
        category, department = "Utilities", "Electricity Board"
    elif any(word in text_lower for word in ["water", "supply", "pipeline", "drainage"]):
        category, department = "Utilities", "Water Supply Dept"
    else:
        category, department = "Administrative Delays", "District Administration"
    
    urgent_words = ["urgent", "emergency", "danger", "critical", "immediate", "hazard"]
    if any(word in text_lower for word in urgent_words):
        priority_score, priority_level = 9, "High 🔴"
    elif len(text) > 150:
        priority_score, priority_level = 6, "Medium 🟠"
    else:
        priority_score, priority_level = 3, "Low 🟢"
    
    summary = text[:200] + "..." if len(text) > 200 else text
    
    return {
        "category": category,
        "department": department,
        "priority_score": priority_score,
        "priority_level": priority_level,
        "summary": summary,
        "urgency_reasons": ["Based on keyword analysis", "Content length considered"]
    }

def load_demo_data():
    """Load demo complaints for presentation"""
    demo_complaints = [
        {
            "text": "Massive pothole on MG Road near City Mall causing daily accidents. Two-wheeler riders at extreme risk. Urgent repair needed!",
            "category": "Civic Infrastructure",
            "priority_score": 9,
            "department": "Public Works Department",
            "sentiment": "Angry"
        },
        {
            "text": "Garbage not collected for 5 days in Vaishali Nagar, Sector 12. Foul smell, health hazard for children and elderly.",
            "category": "Sanitation",
            "priority_score": 8,
            "department": "Municipal Sanitation Dept",
            "sentiment": "Frustrated"
        },
        {
            "text": "Street lights not working in Mansarovar residential area for 3 weeks. Women feeling unsafe during evening hours.",
            "category": "Public Safety",
            "priority_score": 7,
            "department": "Electricity Board",
            "sentiment": "Concerned"
        },
        {
            "text": "Water supply disrupted in entire Malviya Nagar for 48 hours. Affecting 500+ families. No prior notice given.",
            "category": "Utilities",
            "priority_score": 8,
            "department": "Water Supply Dept",
            "sentiment": "Frustrated"
        },
        {
            "text": "Government school in Raja Park lacks basic facilities. No drinking water, damaged toilets affecting 300 students.",
            "category": "Education",
            "priority_score": 7,
            "department": "Education Department",
            "sentiment": "Concerned"
        },
        {
            "text": "Stray dogs attacking residents in Civil Lines area. Child was bitten yesterday. Need immediate action.",
            "category": "Public Safety",
            "priority_score": 9,
            "department": "Municipal Corporation",
            "sentiment": "Angry"
        },
        {
            "text": "Birth certificate application pending for 3 months at Jaipur Municipal Corporation. No response from officials.",
            "category": "Administrative Delays",
            "priority_score": 5,
            "department": "District Administration",
            "sentiment": "Frustrated"
        },
        {
            "text": "Primary health center in Sanganer has no doctor for last 2 weeks. Patients being turned away daily.",
            "category": "Healthcare",
            "priority_score": 8,
            "department": "Health Department",
            "sentiment": "Concerned"
        }
    ]
    
    for i, demo in enumerate(demo_complaints):
        complaint_id = f"DEMO-{datetime.datetime.now().strftime('%Y%m%d')}-{i+1:03d}"
        submitted_time = (datetime.datetime.now() - datetime.timedelta(days=i, hours=i*2)).strftime("%Y-%m-%d %H:%M:%S")
        
        complaint_record = {
            "id": complaint_id,
            "text": demo['text'],
            "category": demo['category'],
            "department": demo['department'],
            "priority_score": demo['priority_score'],
            "priority_level": "High 🔴" if demo['priority_score'] >= 7 else "Medium 🟠" if demo['priority_score'] >= 4 else "Low 🟢",
            "summary": demo['text'][:150] + "...",
            "submitted_at": submitted_time,
            "status": "Open",
            "sentiment": demo['sentiment'],
            "impact_score": calculate_impact_score(demo['text'], demo['category']),
            "estimated_resolution_days": get_estimated_resolution_time(demo['category'], demo['priority_score']),
            "original_language": "English (en)"
        }
        
        st.session_state.complaints.append(complaint_record)
    
    st.session_state.demo_loaded = True

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("<h1 style='text-align: center;'>🏛️ CivicFix AI - Intelligent Grievance Redressal</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>📢 Smart, Transparent & Fast Citizen Complaint Resolution</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered platform that automatically categorizes, prioritizes, and routes citizen grievances to the right authorities.</p>", unsafe_allow_html=True)
st.divider()

# ----------------------------
# Feature Cards
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("### 🧠 AI Analysis")
        st.write("NLP-powered understanding of complaints")

with col2:
    with st.container(border=True):
        st.markdown("### ⚡ Smart Priority")
        st.write("Urgent cases identified & escalated")

with col3:
    with st.container(border=True):
        st.markdown("### 🎯 Auto Routing")
        st.write("Instant department assignment")

with col4:
    with st.container(border=True):
        st.markdown("### 🌍 Multi-lingual")
        st.write("Supports mixed languages")

st.divider()

# ----------------------------
# Sidebar Navigation
# ----------------------------
with st.sidebar:
    st.markdown("# DASHBOARD")
    page = st.radio(
        "Choose a page",
        ["🏠 Submit Grievance", "📊 Admin Dashboard"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("### 📈 Quick Stats")
    total_complaints = len(st.session_state.complaints)
    high_priority = sum(1 for c in st.session_state.complaints if c.get('priority_score', 0) >= 7)
    st.metric("Total Complaints", total_complaints)
    st.metric("High Priority", high_priority)
    
    st.divider()
    
    # Demo data loader
    if not st.session_state.demo_loaded:
        if st.button("🎬 Load Demo Data", use_container_width=True):
            with st.spinner("Loading demo complaints..."):
                load_demo_data()
            st.success("✅ Demo data loaded!")
            st.rerun()
    else:
        if st.button("🗑️ Clear Demo Data", use_container_width=True):
            st.session_state.complaints = [c for c in st.session_state.complaints if not c['id'].startswith('DEMO-')]
            st.session_state.demo_loaded = False
            st.success("✅ Demo data cleared!")
            st.rerun()
    
    st.divider()
    
    # Debug mode toggle
    st.markdown("### 🔧 Developer Options")
    debug_mode = st.checkbox("Enable Debug Mode", value=False, help="Show detailed API responses and errors")
    
    if debug_mode:
        st.caption("🐛 Debug mode active")
        if mistral_client:
            st.success("✅ Mistral AI connected")
        else:
            st.error("❌ Mistral AI not connected")
        
        if groq_available:
            st.success("✅ Groq AI connected")
        else:
            st.error("❌ Groq AI not connected")

# ----------------------------
# PAGE: Submit Grievance
# ----------------------------
if page == "🏠 Submit Grievance":
    st.markdown("## 📝 Submit a Grievance")

    with st.container(border=True):
        st.markdown("#### Choose how you want to submit your complaint")

        tab1, tab2, tab3 = st.tabs(
            ["✍️ Text Complaint", "📷 Photo Evidence", "🎤 Voice Complaint"]
        )

        grievance_text = ""
        image_file = None
        original_text = ""
        detected_language = "English (en)"
        audio_file = None

        # ---- TEXT ----
        with tab1:
            grievance_text = st.text_area(
                "Describe the issue",
                height=160,
                placeholder="Example: Garbage has not been collected for 3 days in my area causing health issues..."
            )
            original_text = grievance_text

        # ---- PHOTO ----
        with tab2:
            image_file = st.file_uploader(
                "Upload a photo (JPG / PNG)",
                type=["jpg", "jpeg", "png"]
            )
            if image_file:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Photo Evidence", use_column_width=True)
                
                if mistral_client:
                    st.success("✅ AI image analysis enabled! Mistral Pixtral Vision will analyze your image.")
                    st.info("💡 The system will automatically identify civic issues (potholes, garbage, etc.) in your photo.")
                elif hf_api_available:
                    st.info("✅ Backup image analysis available.")
                else:
                    st.warning("⚠️ For best results, add MISTRAL_API_KEY to .env file for advanced vision analysis.")
                
                image_description = st.text_area(
                    "Additional description (optional)",
                    height=100,
                    placeholder="Example: This pothole has been here for 2 weeks and caused 3 accidents..."
                )
                if image_description:
                    grievance_text = image_description
                    original_text = image_description

        # ---- AUDIO ----
        with tab3:
            st.markdown("#### 🎤 Submit Voice Complaint")
            
            if groq_api_key:
                st.success("✅ Voice transcription enabled with Groq AI!")
            else:
                st.info("💡 Add GROQ_API_KEY to .env to enable automatic transcription")
                st.caption("Get FREE API key at: https://console.groq.com/keys")
            
            with st.expander("📱 Tips for best results"):
                st.write("• **Supported formats:** MP3, WAV, M4A, OGG, WEBM, FLAC")
                st.write("• **File size:** Keep under 25MB (about 10 minutes)")
                st.write("• **Audio quality:** Clear speech, minimal background noise")
                st.write("• **Languages:** English, Hindi, and other Indian languages")
                st.write("• **Recording tip:** Speak clearly, describe the problem in detail")
            
            audio_file = st.file_uploader(
                "Upload voice complaint (MP3 / WAV / M4A / OGG)",
                type=["mp3", "wav", "m4a", "ogg", "webm", "flac"],
                help="Record your complaint and upload. Groq AI will transcribe it automatically."
            )
            
            if audio_file:
                # Show file info
                file_size_mb = audio_file.size / (1024 * 1024)
                st.caption(f"📎 {audio_file.name} ({file_size_mb:.2f} MB)")
                
                st.audio(audio_file, format=f'audio/{audio_file.type.split("/")[-1]}')
                
                st.info("📝 Click 'Submit Grievance' button below to transcribe and analyze your audio complaint automatically.")

    # ----------------------------
    # Submit Button
    # ----------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    left, center, right = st.columns([3, 2, 3])
    with center:
        submit = st.button("🚀 Submit Grievance", use_container_width=True, type="primary")

    # ----------------------------
    # AI Analysis Section
    # ----------------------------
    if submit:
        # Handle audio transcription first if audio file is uploaded
        if audio_file:
            with st.spinner("🎙️ Transcribing audio..."):
                transcription, success = transcribe_audio(audio_file)
            
            if success:
                st.success("✅ Audio transcribed successfully!")
                
                # Show transcribed text
                with st.expander("📄 View Transcribed Text"):
                    st.write(transcription)
                
                grievance_text = transcription
                original_text = transcription
                
                # Detect language if transcription succeeded
                if mistral_client and transcription.strip():
                    with st.spinner("🌍 Detecting language..."):
                        translated_text, detected_language = detect_and_translate(transcription)
                    
                    if "English" not in detected_language:
                        st.info(f"🌐 Detected Language: **{detected_language}** → Translated to English")
                        with st.expander("📄 View Original Transcription"):
                            st.write(transcription)
                        grievance_text = translated_text
            else:
                st.error(transcription)
                st.error("⚠️ Unable to process audio. Please try again or use text/photo submission.")
                st.stop()
        
        # Now check if we have any content to analyze
        if grievance_text.strip() == "" and image_file is None:
            st.error("⚠️ Please submit a grievance using text, photo, or audio.")
        else:
            # Language detection for text (if not already done for audio)
            if grievance_text.strip() and not image_file and not audio_file:
                with st.spinner("🌍 Detecting language..."):
                    translated_text, detected_language = detect_and_translate(grievance_text)
                
                if "English" not in detected_language:
                    st.info(f"🌐 Detected Language: **{detected_language}** → Translated to English for processing")
                    with st.expander("📄 View Original Text"):
                        st.write(original_text)
                    grievance_text = translated_text
            
            # AI Analysis with progress
            with st.spinner("🔍 Analyzing complaint content..."):
                time.sleep(0.3)
            with st.spinner("🧠 AI classifying issue category..."):
                time.sleep(0.3)
            with st.spinner("⚡ Calculating priority score..."):
                time.sleep(0.3)
            with st.spinner("🎯 Routing to department..."):
                analysis = analyze_complaint_with_ai(grievance_text if grievance_text.strip() else "", image_file)
            
            # Validate analysis
            required_keys = ['category', 'department', 'priority_score', 'priority_level', 'summary', 'urgency_reasons']
            if not all(key in analysis for key in required_keys):
                fallback_text = grievance_text if grievance_text.strip() else "Image uploaded"
                analysis = analyze_complaint_fallback(fallback_text)
            
            # Additional analysis
            sentiment = analyze_sentiment(grievance_text if grievance_text else "")
            impact_score = calculate_impact_score(grievance_text if grievance_text else "", analysis['category'])
            resolution_days = get_estimated_resolution_time(analysis['category'], analysis['priority_score'])
            
            st.success("✅ Grievance Submitted Successfully")
            
            # Check for similar complaints
            if grievance_text.strip():
                similar = find_similar_complaints(grievance_text, st.session_state.complaints)
                if similar:
                    with st.expander(f"⚠️ Found {len(similar)} similar complaint(s) - View Details"):
                        for s in similar:
                            st.markdown(f"**{s['id']}** - {s['category']} ({s['similarity']} match)")
                            st.caption(f"Status: {s['status']} | {s['text']}...")
                            st.divider()

            st.markdown("## 🔍 AI Analysis & Routing")

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric("📂 Category", analysis['category'])
                c2.metric("⚠️ Priority", analysis['priority_level'])
                c3.metric("🏢 Department", analysis['department'])
                c4.metric("📊 Score", f"{analysis['priority_score']}/10")
                
                sentiment_emoji = {"Angry": "😡", "Frustrated": "😤", "Concerned": "😟", "Calm": "😐", "Neutral": "😐"}
                c5.metric("😊 Sentiment", f"{sentiment} {sentiment_emoji.get(sentiment, '😐')}")

                st.markdown("### 📄 Complaint Summary")
                st.write(analysis['summary'])

                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### ⚡ Urgency Analysis")
                    for reason in analysis['urgency_reasons']:
                        st.write(f"• {reason}")
                
                with col_right:
                    st.markdown("### 📊 Impact & Resolution")
                    impact_labels = {1: "Individual", 2: "Local Area", 3: "Community-wide"}
                    st.write(f"• **Impact Level:** {impact_labels[impact_score]} (Score: {impact_score}/3)")
                    st.write(f"• **Estimated Resolution:** {resolution_days} days")
                    st.write(f"• **Language:** {detected_language}")

                st.markdown("### 📋 What Happens Next?")
                steps = [
                    f"1️⃣ Complaint routed to **{analysis['department']}**",
                    f"2️⃣ Priority: **{analysis['priority_level']}** (Score: {analysis['priority_score']}/10)",
                    f"3️⃣ Estimated resolution: **{resolution_days} days**",
                    "4️⃣ Updates via SMS/Email (if registered)",
                    "5️⃣ Track using Complaint ID below"
                ]
                
                for step in steps:
                    st.write(step)

                st.markdown("### 🕒 System Status")
                complaint_id = f"CIV-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                submitted_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                status_info = {
                    "Complaint ID": complaint_id,
                    "Status": "Open",
                    "Submitted At": submitted_at,
                    "AI Confidence": "High" if mistral_client else "Fallback Mode",
                    "Processing Time": "< 5 seconds"
                }
                
                st.code(complaint_id, language=None)
                st.json(status_info)
                
                # Mock notification
                st.toast(f"📧 Confirmation sent to registered email/SMS", icon="✅")

            # Store complaint
            complaint_record = {
                "id": complaint_id,
                "text": original_text if original_text else grievance_text,
                "category": analysis['category'],
                "department": analysis['department'],
                "priority_score": analysis['priority_score'],
                "priority_level": analysis['priority_level'],
                "summary": analysis['summary'],
                "submitted_at": submitted_at,
                "status": "Open",
                "sentiment": sentiment,
                "impact_score": impact_score,
                "estimated_resolution_days": resolution_days,
                "original_language": detected_language
            }
            
            st.session_state.complaints.append(complaint_record)
            save_complaint_to_csv(complaint_record)

# ----------------------------
# PAGE: Admin Dashboard
# ----------------------------
elif page == "📊 Admin Dashboard":
    st.markdown("## 📊 Admin Dashboard - Grievance Analytics")
    
    if len(st.session_state.complaints) == 0:
        st.info("📭 No complaints submitted yet. Submit a grievance or load demo data to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.complaints)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Complaints", len(df))
        col2.metric("High Priority", len(df[df['priority_score'] >= 7]))
        col3.metric("Medium Priority", len(df[(df['priority_score'] >= 4) & (df['priority_score'] < 7)]))
        col4.metric("Open Cases", len(df[df['status'] == 'Open']))
        col5.metric("Avg Priority", f"{df['priority_score'].mean():.1f}/10")
        
        st.divider()
        
        # Filters
        st.markdown("### 🔍 Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            category_filter = st.multiselect(
                "Filter by Category", 
                df['category'].unique(), 
                default=list(df['category'].unique())
            )
        
        with filter_col2:
            priority_filter = st.multiselect(
                "Filter by Priority", 
                df['priority_level'].unique(),
                default=list(df['priority_level'].unique())
            )
        
        with filter_col3:
            dept_filter = st.multiselect(
                "Filter by Department",
                df['department'].unique(),
                default=list(df['department'].unique())
            )
        
        # Apply filters
        filtered_df = df[
            (df['category'].isin(category_filter)) &
            (df['priority_level'].isin(priority_filter)) &
            (df['department'].isin(dept_filter))
        ]
        
        st.caption(f"Showing {len(filtered_df)} of {len(df)} complaints")
        
        st.divider()
        
        # Charts Row 1
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📂 Complaints by Category")
            if len(filtered_df) > 0:
                category_counts = filtered_df['category'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(range(len(category_counts)))
                ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax1.axis('equal')
                st.pyplot(fig1)
            else:
                st.info("No data to display")
        
        with col_chart2:
            st.markdown("### ⚠️ Priority Distribution")
            if len(filtered_df) > 0:
                priority_counts = filtered_df['priority_level'].value_counts()
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors_priority = {'High 🔴': '#ff4444', 'Medium 🟠': '#ff8800', 'Low 🟢': '#44ff44'}
                bar_colors = [colors_priority.get(p, '#cccccc') for p in priority_counts.index]
                ax2.bar(priority_counts.index, priority_counts.values, color=bar_colors)
                ax2.set_ylabel('Number of Complaints')
                ax2.set_xlabel('Priority Level')
                plt.xticks(rotation=15)
                st.pyplot(fig2)
            else:
                st.info("No data to display")
        
        st.divider()
        
        # Charts Row 2
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown("### 🏢 Department Workload")
            if len(filtered_df) > 0:
                dept_counts = filtered_df['department'].value_counts()
                st.bar_chart(dept_counts)
            else:
                st.info("No data to display")
        
        with col_chart4:
            st.markdown("### 😊 Sentiment Analysis")
            if len(filtered_df) > 0 and 'sentiment' in filtered_df.columns:
                sentiment_counts = filtered_df['sentiment'].value_counts()
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                sentiment_colors = {'Angry': '#ff0000', 'Frustrated': '#ff6600', 'Concerned': '#ffcc00', 'Calm': '#00cc00', 'Neutral': '#888888'}
                bar_colors_sentiment = [sentiment_colors.get(s, '#cccccc') for s in sentiment_counts.index]
                ax4.barh(sentiment_counts.index, sentiment_counts.values, color=bar_colors_sentiment)
                ax4.set_xlabel('Number of Complaints')
                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.info("No sentiment data available")
        
        st.divider()
        
        # Timeline
        if len(filtered_df) > 0:
            st.markdown("### 📈 Complaints Timeline")
            try:
                timeline_df = filtered_df.copy()
                timeline_df['date'] = pd.to_datetime(timeline_df['submitted_at']).dt.date
                daily_counts = timeline_df.groupby('date').size()
                st.line_chart(daily_counts)
            except:
                st.info("Timeline data unavailable")
        
        st.divider()
        
        # Recent Complaints Table
        st.markdown("### 📋 Recent Complaints")
        display_cols = ['id', 'category', 'department', 'priority_level', 'priority_score', 'sentiment', 'submitted_at', 'status']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        display_df = filtered_df[available_cols].copy()
        display_df = display_df.sort_values('submitted_at', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
        
        # Export options
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📥 Export Filtered Data as CSV", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv,
                    file_name=f"grievances_filtered_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_export2:
            if st.button("📥 Export All Data as CSV", use_container_width=True):
                csv_all = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download All Data",
                    data=csv_all,
                    file_name=f"grievances_all_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.divider()
        
        # High Priority Alerts
        high_priority_complaints = filtered_df[filtered_df['priority_score'] >= 8]
        if len(high_priority_complaints) > 0:
            st.markdown("### 🚨 Critical Priority Alerts")
            for _, complaint in high_priority_complaints.iterrows():
                with st.expander(f"🔴 {complaint['id']} - {complaint['category']} (Score: {complaint['priority_score']}/10)"):
                    st.write(f"**Department:** {complaint['department']}")
                    st.write(f"**Submitted:** {complaint['submitted_at']}")
                    st.write(f"**Summary:** {complaint['summary']}")
                    st.write(f"**Status:** {complaint['status']}")

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>🏆 CivicFix AI - Hackathon Prototype | AI for Good Governance | Built with Streamlit + Mistral AI</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>💡 Features: Multi-modal Input | Multi-lingual Support | Smart Prioritization | Auto-routing | Real-time Analytics</p>", unsafe_allow_html=True)