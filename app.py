# Import non-streamlit packages first
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping, box
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from patchify import patchify, unpatchify
import datetime
import torch
import torchvision
import math
from scipy import ndimage
import ee
import tempfile
import zipfile
from pathlib import Path
import requests
import time
import io
import warnings
import sys
import base64
import json
from importlib import import_module
import geopandas as gpd
import subprocess

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Now import streamlit as the first streamlit-related import
import streamlit as st

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="سیستم تشخیص تغییرات ساختمانی",
    page_icon="🏗️"
)

# Custom CSS for Persian styling with B Nazanin font and improved color theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'B Nazanin', 'Vazirmatn', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        text-align: center !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        font-size: 1.8rem !important;
        color: #34495e !important;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 !important;
    }
    
    h3 {
        font-size: 1.4rem !important;
        color: #5a6c7d !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e8f4f8;
        border-color: #3498db;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border-color: #2980b9 !important;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    [data-baseweb="notification"] {
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border-color: #28a745 !important;
        color: #155724 !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%) !important;
        border-color: #ffc107 !important;
        color: #856404 !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
        border-color: #dc3545 !important;
        color: #721c24 !important;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border-color: #17a2b8 !important;
        color: #0c5460 !important;
    }
    
    /* Selectbox and inputs */
    .stSelectbox, .stTextInput, .stSlider {
        border-radius: 10px;
    }
    
    [data-baseweb="select"] {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    [data-baseweb="select"]:hover {
        border-color: #3498db;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2);
    }
    
    /* Cards */
    .time-selection-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #dee2e6;
    }
    
    .time-selection-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #495057;
        border-bottom: 2px solid #6c757d;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e9f2 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #2c3e50;
        border: 1px solid #b8d4e0;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed #6c757d;
    }
    
    /* Columns spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Matplotlib figures */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Now import other streamlit-related packages
import folium
from streamlit_folium import folium_static, st_folium
import geemap
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Function to download model from Google Drive with fixed URL and better error handling
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """
    Download a file from Google Drive using the sharing URL with improved error handling
    """
    try:
        correct_file_id = "1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov"
        
        st.info(f"در حال دانلود مدل از Google Drive (شناسه فایل: {correct_file_id})...")
        
        try:
            import gdown
        except ImportError:
            st.info("در حال نصب کتابخانه gdown...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        download_methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id
        ]
        
        for i, method in enumerate(download_methods):
            try:
                st.info(f"روش دانلود {i+1} از 3 در حال اجرا...")
                
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    file_size = os.path.getsize(local_filename)
                    
                    try:
                        with open(local_filename, 'rb') as f:
                            header = f.read(10)
                            if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK'):
                                st.success(f"مدل با موفقیت دانلود شد! حجم: {file_size / (1024*1024):.1f} مگابایت")
                                return local_filename
                            else:
                                st.warning(f"روش {i+1}: فایل دانلود شده معتبر نیست")
                                if os.path.exists(local_filename):
                                    os.remove(local_filename)
                    except Exception as e:
                        st.warning(f"روش {i+1}: خطا در اعتبارسنجی فایل: {e}")
                        if os.path.exists(local_filename):
                            os.remove(local_filename)
                else:
                    st.warning(f"روش {i+1}: فایل دانلود شده خالی است")
                    if os.path.exists(local_filename):
                        os.remove(local_filename)
                        
            except Exception as e:
                st.warning(f"روش دانلود {i+1} با خطا مواجه شد: {str(e)}")
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                continue
        
        st.info("تمام روش‌های gdown ناموفق بودند. تلاش با روش دستی...")
        return manual_download_fallback(correct_file_id, local_filename)
            
    except Exception as e:
        st.error(f"خطا در تابع دانلود: {str(e)}")
        return None

def manual_download_fallback(file_id, local_filename):
    """
    Fallback manual download method using requests
    """
    try:
        import requests
        
        urls_to_try = [
            f"https://drive.google.com/uc?export=download&id={file_id}",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
        ]
        
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for i, url in enumerate(urls_to_try):
            try:
                st.info(f"روش دستی {i+1} از 3 در حال اجرا...")
                response = session.get(url, headers=headers, stream=True, timeout=30)
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/html' not in content_type:
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded_size = 0
                        
                        if total_size > 0:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        with open(local_filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    
                                    if total_size > 0:
                                        progress = downloaded_size / total_size
                                        progress_bar.progress(progress)
                                        status_text.text(f"دانلود شده: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        
                        if total_size > 0:
                            progress_bar.empty()
                            status_text.empty()
                        
                        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                            file_size = os.path.getsize(local_filename)
                            st.success(f"دانلود دستی موفقیت‌آمیز بود! حجم: {file_size / (1024*1024):.1f} مگابایت")
                            return local_filename
                    else:
                        st.warning(f"روش {i+1} به جای فایل، HTML برگرداند")
                        
            except Exception as e:
                st.warning(f"روش دستی {i+1} با خطا مواجه شد: {e}")
                continue
        
        st.error("تمام روش‌های دانلود خودکار ناموفق بودند. لطفاً به صورت دستی دانلود کنید:")
        
        st.info("**راهنمای دانلود دستی:**")
        st.markdown(f"""
        1. **این لینک را در تب جدید مرورگر باز کنید:** 
           https://drive.google.com/file/d/{file_id}/view
        
        2. **اگر خطای دسترسی دیدید:**
           - صاحب فایل باید اشتراک‌گذاری را به "هرکسی با لینک می‌تواند مشاهده کند" تغییر دهد
        
        3. **روی دکمه دانلود کلیک کنید**
        
        4. **فایل را با نام زیر ذخیره کنید:** `{local_filename}`
        
        5. **از بخش زیر آپلود کنید**
        """)
        
        uploaded_file = st.file_uploader(
            f"فایل مدل ({local_filename}) را پس از دانلود دستی آپلود کنید:",
            type=['pt', 'pth'],
            help="فایل را به صورت دستی از Google Drive دانلود کرده و اینجا آپلود کنید"
        )
        
        if uploaded_file is not None:
            with open(local_filename, 'wb') as f:
                f.write(uploaded_file.read())
            
            file_size = os.path.getsize(local_filename)
            st.success(f"مدل با موفقیت آپلود شد! حجم: {file_size / (1024*1024):.1f} مگابایت")
            return local_filename
        
        return None
        
    except Exception as e:
        st.error(f"دانلود دستی با خطا مواجه شد: {e}")
        return None

# Model loading section
gdrive_model_url = "https://drive.google.com/file/d/1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov/view?usp=drive_link"
model_path = "best_model_version_Unet++_v02_e7.pt"

if not os.path.exists(model_path):
    st.info("مدل به صورت محلی یافت نشد. در حال دانلود از Google Drive...")
    
    downloaded_model_path = download_model_from_gdrive(gdrive_model_url, model_path)
    
    if downloaded_model_path is None:
        st.error("دانلود خودکار ناموفق بود. لطفاً از گزینه دانلود دستی استفاده کنید.")
        st.stop()
else:
    st.success("مدل به صورت محلی یافت شد!")

# Verify the model file
if os.path.exists(model_path):
    try:
        file_size = os.path.getsize(model_path)
        st.info(f"حجم فایل مدل: {file_size / (1024*1024):.1f} مگابایت")
        
        with open(model_path, 'rb') as f:
            header = f.read(10)
            if not (header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK')):
                st.error("فایل مدل خراب یا نامعتبر است.")
                st.error(f"هدر فایل: {header}")
                st.info("لطفاً فایل مدل را مجدداً دانلود کنید.")
                
                try:
                    with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(200)
                        st.code(content, language='text')
                except:
                    pass
                
                os.remove(model_path)
                st.stop()
            else:
                st.success("فایل مدل معتبر است!")
                
    except Exception as e:
        st.error(f"خطا در اعتبارسنجی فایل مدل: {e}")
        st.stop()

# Install GEES2Downloader if not already installed
try:
    from geeS2downloader.geeS2downloader import GEES2Downloader
    st.sidebar.success("GEES2Downloader نصب شده است.")
except ImportError:
    st.sidebar.info("در حال نصب GEES2Downloader...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/cordmaur/GEES2Downloader.git"
        ])
        
        from geeS2downloader.geeS2downloader import GEES2Downloader
        st.sidebar.success("GEES2Downloader با موفقیت نصب شد!")
    except Exception as e:
        st.sidebar.error(f"نصب GEES2Downloader ناموفق بود: {str(e)}")
        st.sidebar.info("لطفاً GEES2Downloader را به صورت دستی نصب کنید")

# Initialize Earth Engine
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize()
        return True, "Earth Engine آماده است"
    except Exception as e:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if not base64_key:
                return False, "کلید سرویس Earth Engine در متغیرهای محیطی یافت نشد."
            
            key_json = base64.b64decode(base64_key).decode()
            key_data = json.loads(key_json)
            
            key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            with open(key_file.name, 'w') as f:
                json.dump(key_data, f)
            
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'],
                key_file.name
            )
            ee.Initialize(credentials)
            
            os.unlink(key_file.name)
            
            return True, "احراز هویت Earth Engine موفقیت‌آمیز بود!"
        except Exception as auth_error:
            return False, f"احراز هویت ناموفق بود: {str(auth_error)}"

ee_initialized, ee_message = initialize_earth_engine()
if ee_initialized:
    st.sidebar.success(ee_message)
else:
    st.sidebar.error(ee_message)
    st.error("احراز هویت Earth Engine برای استفاده از این برنامه ضروری است.")
    st.info("""
    لطفاً متغیر محیطی GOOGLE_EARTH_ENGINE_KEY_BASE64 را با کلید حساب سرویس خود تنظیم کنید.
    
    1. یک حساب سرویس در Google Cloud Console ایجاد کنید
    2. یک کلید JSON برای حساب سرویس تولید کنید
    3. کلید JSON را به base64 تبدیل کنید
    4. آن را به عنوان متغیر محیطی در Posit Cloud تنظیم کنید
    """)
    st.stop()

# Main title
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🏗️ سیستم تشخیص تغییرات ساختمانی با تصاویر ماهواره‌ای</h1>", unsafe_allow_html=True)

# Create tabs for different pages
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ انتخاب منطقه", 
    "📅 تصویر قبل", 
    "📅 تصویر بعد", 
    "🔍 تشخیص تغییرات"
])

# Global variables
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []

if 'last_map_data' not in st.session_state:
    st.session_state.last_map_data = None

if 'clipped_img' not in st.session_state:
    st.session_state.clipped_img = None
    
if 'clipped_img_2024' not in st.session_state:
    st.session_state.clipped_img_2024 = None

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
if 'saved_patches_paths' not in st.session_state:
    st.session_state.saved_patches_paths = []
    
if 'saved_patches_paths_2024' not in st.session_state:
    st.session_state.saved_patches_paths_2024 = []

if 'change_detection_result' not in st.session_state:
    st.session_state.change_detection_result = None

# Define Sentinel-2 bands
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S2_NAMES = ['آئروسل', 'آبی', 'سبز', 'قرمز', 'لبه قرمز 1', 'لبه قرمز 2', 
           'لبه قرمز 3', 'NIR', 'لبه قرمز 4', 'بخار آب', 'SWIR1', 'SWIR2']

# [Keep all the existing functions unchanged - download_sentinel2_with_gees2, normalized, 
# get_utm_zone, get_utm_epsg, convert_to_utm, apply_erosion, load_model, process_image]

# I'll continue with the function definitions in the next part...
# [Previous functions remain the same, just continuing from here]

def download_sentinel2_with_gees2(year, polygon, start_month, end_month, cloud_cover_limit=10):
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    try:
        start_date = f"{year}-{start_month:02d}-01"
        
        if end_month in [4, 6, 9, 11]:
            end_day = 30
        elif end_month == 2:
            end_day = 29 if int(year) % 4 == 0 else 28
        else:
            end_day = 31
            
        end_date = f"{year}-{end_month:02d}-{end_day}"
        
        area_sq_km = polygon.area * 111 * 111
        status_placeholder.info(f"مساحت منطقه انتخابی: ~{area_sq_km:.2f} کیلومتر مربع. در حال جستجوی تصاویر...")
        
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, f"sentinel2_{year}_{start_month:02d}_{end_month:02d}_median.tif")
        
        geojson = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
        ee_geometry = ee.Geometry.Polygon(geojson['coordinates'])
        
        status_placeholder.info(f"ایجاد مجموعه تصاویر برای {start_date} تا {end_date} با پوشش ابر < {cloud_cover_limit}%...")
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(ee_geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_limit)))
        
        count = collection.size().getInfo()
        status_placeholder.info(f"{count} تصویر با پوشش ابر < {cloud_cover_limit}% یافت شد")
        
        if count == 0:
            status_placeholder.warning(f"هیچ تصویری برای {year} با پوشش ابر < {cloud_cover_limit}% یافت نشد")
            
            higher_limit = min(cloud_cover_limit * 2, 100)
            status_placeholder.info(f"تلاش با محدودیت ابر بالاتر ({higher_limit}%)...")
            
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(ee_geometry)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', higher_limit)))
            
            count = collection.size().getInfo()
            status_placeholder.info(f"{count} تصویر با پوشش ابر < {higher_limit}% یافت شد")
            
            if count == 0:
                status_placeholder.error(f"حتی با محدودیت ابر بالاتر هیچ تصویری یافت نشد.")
                return None
        
        status_placeholder.info(f"ایجاد ترکیب میانه از {count} تصویر...")
        median_image = collection.median().select(S2_BANDS)
        
        def progress_callback(progress):
            progress_placeholder.progress(progress)
        
        bands_dir = os.path.join(temp_dir, "bands")
        os.makedirs(bands_dir, exist_ok=True)
        
        status_placeholder.info("دانلود باندها...")
        band_files = []
        
        region = ee_geometry.bounds().getInfo()['coordinates']
        
        for i, band in enumerate(S2_BANDS):
            try:
                status_placeholder.info(f"دانلود باند {band} ({i+1}/{len(S2_BANDS)})...")
                
                band_file = os.path.join(bands_dir, f"{band}.tif")
                
                url = median_image.select(band).getDownloadURL({
                    'scale': 10,
                    'region': region,
                    'format': 'GEO_TIFF',
                    'bands': [band]
                })
                
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(band_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    band_files.append(band_file)
                    progress_callback((i + 1) / len(S2_BANDS))
                else:
                    status_placeholder.error(f"دانلود باند {band} ناموفق بود")
            except Exception as e:
                status_placeholder.error(f"خطا در دانلود باند {band}: {str(e)}")
        
        if len(band_files) == len(S2_BANDS):
            status_placeholder.info("تمام باندها دانلود شدند. ایجاد GeoTIFF چندباندی...")
            
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            
            meta.update(count=len(band_files))
            
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            status_placeholder.success("GeoTIFF چندباندی با موفقیت ایجاد شد")
            return output_file
        else:
            status_placeholder.error(f"تنها {len(band_files)}/{len(S2_BANDS)} باند دانلود شد")
            return None
        
    except Exception as e:
        status_placeholder.error(f"خطا در دانلود داده: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def normalized(img):
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm

def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    zone_number = get_utm_zone(longitude)
    
    if latitude >= 0:
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"

def convert_to_utm(src_path, dst_path, polygon=None):
    with rasterio.open(src_path) as src:
        if polygon:
            centroid = polygon.centroid
            lon, lat = centroid.x, centroid.y
            dst_crs = get_utm_epsg(lon, lat)
            st.info(f"زون UTM تشخیص داده شده: {get_utm_zone(lon)} ({dst_crs})")
        else:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            dst_crs = get_utm_epsg(center_lon, center_lat)
            st.info(f"زون UTM تشخیص داده شده: {get_utm_zone(center_lon)} ({dst_crs})")
        
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
        
        return dst_path, dst_crs

def apply_erosion(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded_image = ndimage.binary_erosion(image, structure=kernel).astype(image.dtype)
    return eroded_image

@st.cache_resource
def load_model(model_path):
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b7',
            encoder_weights='imagenet',
            in_channels=12,
            classes=1,
            decoder_attention_type='scse'
        ).to(device)

        loaded_object = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            model.load_state_dict(loaded_object['model_state_dict'])
            st.info("مدل از دیکشنری checkpoint بارگذاری شد.")
        elif isinstance(loaded_object, dict):
            model.load_state_dict(loaded_object)
            st.info("مدل مستقیماً از state_dict بارگذاری شد.")
        else:
            st.error("فرمت فایل مدل قابل شناسایی نیست.")
            st.session_state.model_loaded = False
            return None, None

        model.eval()
        st.session_state.model_loaded = True
        st.success("مدل با موفقیت بارگذاری شد!")
        return model, device
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل: {str(e)}")
        st.session_state.model_loaded = False
        return None, None

def process_image(image_path, year, selected_polygon, region_number):
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Step 1: Clipping
        status_placeholder.info("مرحله 1 از 4: برش تصویر...")
        progress_placeholder.progress(0)
        
        with rasterio.open(image_path) as src:
            raster_bounds = box(*src.bounds)
            polygon_shapely = selected_polygon
            
            if not raster_bounds.intersects(polygon_shapely):
                status_placeholder.error("خطا: منطقه انتخابی با تصویر دانلود شده همپوشانی ندارد.")
                return False
            
            geoms = [mapping(selected_polygon)]
            
            try:
                clipped_img, clipped_transform = mask(src, geoms, crop=True)
                
                if clipped_img.size == 0 or np.all(clipped_img == 0):
                    status_placeholder.error("خطا: برش تصویر خالی است.")
                    return False
                
            except ValueError as e:
                if "Input shapes do not overlap raster" in str(e):
                    status_placeholder.error("خطا: منطقه انتخابی با داده‌های معتبر در تصویر همپوشانی ندارد.")
                    return False
                else:
                    raise
            
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "height": clipped_img.shape[1],
                "width": clipped_img.shape[2],
                "transform": clipped_transform
            })
            
            temp_dir = os.path.dirname(image_path)
            os.makedirs(os.path.join(temp_dir, "temp"), exist_ok=True)
            temp_clipped_path = os.path.join(temp_dir, "temp", f"temp_clipped_{year}_region{region_number}.tif")
            
            with rasterio.open(temp_clipped_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_img)
            
            utm_clipped_path = os.path.join(temp_dir, "temp", f"utm_clipped_{year}_region{region_number}.tif")
            utm_path, utm_crs = convert_to_utm(temp_clipped_path, utm_clipped_path, selected_polygon)
            
            with rasterio.open(utm_path) as src_utm:
                clipped_img = src_utm.read()
                clipped_meta = src_utm.meta.copy()
            
            if year == st.session_state.before_year:
                st.session_state.clipped_img = clipped_img
                st.session_state.clipped_meta = clipped_meta
                st.session_state.region_number = region_number
                st.session_state.year = year
            else:
                st.session_state.clipped_img_2024 = clipped_img
                st.session_state.clipped_meta_2024 = clipped_meta
                st.session_state.region_number_2024 = region_number
                st.session_state.year_2024 = year
            
            if clipped_img.shape[1] < 300 or clipped_img.shape[2] < 300:
                status_placeholder.error(f"تصویر برش‌خورده بسیار کوچک است ({clipped_img.shape[1]}x{clipped_img.shape[2]} پیکسل).")
                return False
            
            rgb_bands = [3, 2, 1]
            
            if clipped_img.shape[0] >= 3:
                rgb = np.zeros((clipped_img.shape[1], clipped_img.shape[2], 3), dtype=np.float32)
                
                for i, band in enumerate(rgb_bands):
                    if band < clipped_img.shape[0]:
                        band_data = clipped_img[band]
                        min_val = np.percentile(band_data, 2)
                        max_val = np.percentile(band_data, 98)
                        rgb[:, :, i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb)
                ax.set_title(f"تصویر برش‌خورده سنتینل-2 سال {year}")
                ax.axis('off')
                st.pyplot(fig)
            
            progress_placeholder.progress(25)
            status_placeholder.success("مرحله 1 از 4: برش با موفقیت انجام شد")
        
        # Step 2: Patching
        status_placeholder.info("مرحله 2 از 4: ایجاد تکه‌های تصویر...")
        
        img_for_patching = np.moveaxis(clipped_img, 0, -1)
        
        patch_size = 224
        patches = patchify(img_for_patching, (patch_size, patch_size, clipped_img.shape[0]), step=patch_size)
        patches = patches.squeeze()
        
        base_dir = os.path.dirname(image_path)
        output_folder = os.path.join(base_dir, f"patches_{year}_region{region_number}")
        os.makedirs(output_folder, exist_ok=True)
        
        num_patches = patches.shape[0] * patches.shape[1]
        saved_paths = []
        
        patch_progress = st.progress(0)
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j]
                patch_normalized = normalized(patch)
                patch_for_saving = np.moveaxis(patch_normalized, -1, 0)
                
                patch_name = f"region{region_number}_{i}_{j}.tif"
                output_file_path = os.path.join(output_folder, patch_name)
                saved_paths.append(output_file_path)
                
                with rasterio.open(
                    output_file_path, 'w', driver='GTiff',
                    height=patch_size, width=patch_size,
                    count=patch_for_saving.shape[0], dtype='float64',
                    crs=clipped_meta.get('crs'),
                    transform=clipped_meta.get('transform')
                ) as dst:
                    for band_idx in range(patch_for_saving.shape[0]):
                        band_data = patch_for_saving[band_idx].reshape(patch_size, patch_size)
                        dst.write(band_data, band_idx+1)
                
                patch_count = i * patches.shape[1] + j + 1
                patch_progress.progress(patch_count / num_patches)
        
        if year == st.session_state.before_year:
            st.session_state.saved_patches_paths = saved_paths
            st.session_state.patches_shape = patches.shape
            st.session_state.patches_info = {
                'year': year, 'region_number': region_number,
                'output_folder': output_folder, 'patch_size': patch_size
            }
        else:
            st.session_state.saved_patches_paths_2024 = saved_paths
            st.session_state.patches_shape_2024 = patches.shape
            st.session_state.patches_info_2024 = {
                'year': year, 'region_number': region_number,
                'output_folder': output_folder, 'patch_size': patch_size
            }
        
        num_samples = min(6, num_patches)
        if num_samples > 0:
            st.subheader("نمونه تکه‌های تصویر")
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
            if num_samples == 1:
                axes = [axes]
            
            for idx, ax in enumerate(axes):
                if idx < num_patches:
                    i, j = idx // patches.shape[1], idx % patches.shape[1]
                    patch = patches[i, j]
                    
                    rgb_bands = [3, 2, 1]
                    patch_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                    for b, band in enumerate(rgb_bands):
                        if band < patch.shape[-1]:
                            band_data = patch[:, :, band]
                            min_val = np.percentile(band_data, 2)
                            max_val = np.percentile(band_data, 98)
                            patch_rgb[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                    
                    ax.imshow(patch_rgb)
                    ax.set_title(f"تکه {i}_{j}")
                    ax.axis('off')
            
            st.pyplot(fig)
        
        progress_placeholder.progress(50)
        status_placeholder.success(f"مرحله 2 از 4: {num_patches} تکه با موفقیت ایجاد شد")
        
        # Step 3: Classification
        status_placeholder.info("مرحله 3 از 4: طبقه‌بندی تکه‌ها...")
        
        if not st.session_state.model_loaded:
            with st.spinner("بارگذاری مدل..."):
                model, device = load_model(model_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                else:
                    status_placeholder.error("بارگذاری مدل ناموفق بود.")
                    return False
        
        if year == st.session_state.before_year:
            patches_info = st.session_state.patches_info
            saved_paths = st.session_state.saved_patches_paths
            patches_shape = st.session_state.patches_shape
        else:
            patches_info = st.session_state.patches_info_2024
            saved_paths = st.session_state.saved_patches_paths_2024
            patches_shape = st.session_state.patches_shape_2024
        
        base_dir = os.path.dirname(image_path)
        classified_folder = os.path.join(base_dir, f"classified_{year}_region{region_number}")
        os.makedirs(classified_folder, exist_ok=True)
        
        classified_results = []
        classified_paths = []
        
        classify_progress = st.progress(0)
        
        total_patches = len(saved_paths)
        for idx, patch_path in enumerate(saved_paths):
            try:
                filename = os.path.basename(patch_path)
                i_j_part = filename.split('_')[-2:]
                i = int(i_j_part[0])
                j = int(i_j_part[1].split('.')[0])
                
                with rasterio.open(patch_path) as src:
                    patch = src.read()
                    patch_meta = src.meta.copy()
                
                rgb_bands = [3, 2, 1]
                if patch.shape[0] >= 3:
                    rgb_patch = np.zeros((patch.shape[1], patch.shape[2], 3), dtype=np.float32)
                    
                    for b, band in enumerate(rgb_bands):
                        if band < patch.shape[0]:
                            band_data = patch[band]
                            min_val = np.percentile(band_data, 2)
                            max_val = np.percentile(band_data, 98)
                            rgb_patch[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                
                img_tensor = torch.tensor(patch, dtype=torch.float32)
                img_tensor = img_tensor.unsqueeze(0)
                
                with torch.inference_mode():
                    prediction = st.session_state.model(img_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                pred_np = prediction.squeeze().numpy()
                binary_mask = (pred_np > 0.5).astype(np.uint8) * 255
                
                output_filename = f"classified_region{patches_info['region_number']}_{i}_{j}.tif"
                output_path = os.path.join(classified_folder, output_filename)
                classified_paths.append(output_path)
                
                patch_meta.update({'count': 1, 'dtype': 'uint8'})
                
                with rasterio.open(output_path, 'w', **patch_meta) as dst:
                    dst.write(binary_mask.reshape(1, binary_mask.shape[0], binary_mask.shape[1]))
                
                if len(classified_results) < 6:
                    classified_results.append({
                        'path': output_path, 'i': i, 'j': j,
                        'mask': binary_mask, 'rgb_original': rgb_patch
                    })
                
                classify_progress.progress((idx + 1) / total_patches)
                
            except Exception as e:
                st.error(f"خطا در پردازش تکه {patch_path}: {str(e)}")
        
        if year == st.session_state.before_year:
            st.session_state.classified_paths = classified_paths
            st.session_state.classified_shape = patches_shape
        else:
            st.session_state.classified_paths_2024 = classified_paths
            st.session_state.classified_shape_2024 = patches_shape
        
        if classified_results:
            st.subheader("نمونه نتایج طبقه‌بندی")
            
            num_samples = len(classified_results)
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
            
            for idx, result in enumerate(classified_results):
                axes[0, idx].imshow(result['rgb_original'])
                axes[0, idx].set_title(f"اصلی {result['i']}_{result['j']}")
                axes[0, idx].axis('off')
            
            for idx, result in enumerate(classified_results):
                axes[1, idx].imshow(result['mask'], cmap='gray')
                axes[1, idx].set_title(f"طبقه‌بندی شده {result['i']}_{result['j']}")
                axes[1, idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        progress_placeholder.progress(75)
        status_placeholder.success(f"مرحله 3 از 4: {total_patches} تکه با موفقیت طبقه‌بندی شد")
        
        # Step 4: Reconstruction
        status_placeholder.info("مرحله 4 از 4: بازسازی تصویر کامل...")
        
        if year == st.session_state.before_year:
            patches_info = st.session_state.patches_info
            classified_paths = st.session_state.classified_paths
            patches_shape = st.session_state.classified_shape
            clipped_meta = st.session_state.clipped_meta
        else:
            patches_info = st.session_state.patches_info_2024
            classified_paths = st.session_state.classified_paths_2024
            patches_shape = st.session_state.classified_shape_2024
            clipped_meta = st.session_state.clipped_meta_2024
        
        patches = []
        patch_indices = []
        
        for path in classified_paths:
            filename = os.path.basename(path)
            parts = filename.split('_')
            i_j_part = parts[-2:]
            i = int(i_j_part[0])
            j = int(i_j_part[1].split('.')[0])
            
            with rasterio.open(path) as src:
                patch = src.read(1)
                patches.append(patch)
                patch_indices.append((i, j))
        
        i_vals = [idx[0] for idx in patch_indices]
        j_vals = [idx[1] for idx in patch_indices]
        max_i = max(i_vals) + 1
        max_j = max(j_vals) + 1
        
        patch_size = patches_info['patch_size']
        grid = np.zeros((max_i, max_j, patch_size, patch_size), dtype=np.uint8)
        
        for (i, j), patch in zip(patch_indices, patches):
            grid[i, j] = patch
        
        reconstructed_image = unpatchify(grid, (max_i * patch_size, max_j * patch_size))
        
        base_dir = os.path.dirname(image_path)
        output_filename = f"reconstructed_classification_{year}_region{region_number}.tif"
        output_path = os.path.join(base_dir, output_filename)
        
        out_meta = clipped_meta.copy()
        out_meta.update({
            'count': 1,
            'height': reconstructed_image.shape[0],
            'width': reconstructed_image.shape[1],
            'dtype': 'uint8'
        })
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(reconstructed_image, 1)
        
        if year == st.session_state.before_year:
            st.session_state.reconstructed_before_path = output_path
            st.session_state.reconstructed_before_image = reconstructed_image
        else:
            st.session_state.reconstructed_after_path = output_path
            st.session_state.reconstructed_after_image = reconstructed_image
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(reconstructed_image, cmap='gray')
        ax.set_title(f"تصویر طبقه‌بندی بازسازی شده ({year})")
        ax.axis('off')
        st.pyplot(fig)
        
        with open(output_path, "rb") as file:
            st.download_button(
                label=f"دانلود طبقه‌بندی بازسازی شده ({year})",
                data=file,
                file_name=output_filename,
                mime="image/tiff"
            )
        
        progress_placeholder.progress(100)
        status_placeholder.success(f"تمام مراحل پردازش برای {year} با موفقیت انجام شد!")
        
        return True
        
    except Exception as e:
        status_placeholder.error(f"خطا در پردازش: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False
                
