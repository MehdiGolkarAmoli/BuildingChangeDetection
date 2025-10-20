# برنامه تشخیص تغییرات ساختمان با رابط کاربری فارسی کامل
# Complete Persian UI Building Change Detection Application

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

# Now import streamlit
import streamlit as st

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="سیستم تشخیص تغییرات ساختمان",
    page_icon="🏗️"
)

# Custom CSS for Persian font and modern styling
st.markdown("""
<style>
    @import url('https://v1.fontapi.ir/css/BNazanin');
    
    * {
        font-family: 'B Nazanin', 'Tahoma', sans-serif !important;
        direction: rtl;
        text-align: right;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        margin: 20px auto;
    }
    
    h1 {
        color: #2c3e50;
        font-size: 2.8rem !important;
        font-weight: bold;
        text-align: center;
        border-bottom: 4px solid #667eea;
        padding-bottom: 15px;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #34495e;
        font-size: 2rem !important;
        font-weight: bold;
        margin: 25px 0 15px 0;
    }
    
    h3 {
        color: #546e7a;
        font-size: 1.5rem !important;
        margin: 20px 0 10px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-right: 5px solid #2196f3;
        border-radius: 10px;
        padding: 18px;
        color: #1565c0;
        font-size: 1.1rem;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-right: 5px solid #4caf50;
        border-radius: 10px;
        padding: 18px;
        color: #2e7d32;
        font-size: 1.1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-right: 5px solid #ff9800;
        border-radius: 10px;
        padding: 18px;
        color: #e65100;
        font-size: 1.1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-right: 5px solid #f44336;
        border-radius: 10px;
        padding: 18px;
        color: #c62828;
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.25);
        border-radius: 10px;
        color: white;
        font-weight: bold;
        padding: 12px 25px;
        font-size: 1.2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .time-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        border: 3px solid #667eea;
    }
    
    .time-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .region-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-right: 5px solid #667eea;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    .stSelectbox label, .stSlider label {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🏗️ سیستم تشخیص تغییرات ساختمان با تصاویر ماهواره‌ای</h1>", unsafe_allow_html=True)

# Now import other streamlit-related packages
import folium
from streamlit_folium import folium_static, st_folium
import geemap
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Function to download model from Google Drive
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """دانلود فایل از گوگل درایو"""
    try:
        correct_file_id = "1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov"
        
        st.info(f"🔽 در حال دانلود مدل از گوگل درایو (شناسه فایل: {correct_file_id})...")
        
        try:
            import gdown
        except ImportError:
            st.info("📦 در حال نصب کتابخانه gdown...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        download_methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id
        ]
        
        for i, method in enumerate(download_methods):
            try:
                st.info(f"🔄 تلاش روش {i+1} از 3...")
                
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    file_size = os.path.getsize(local_filename)
                    
                    try:
                        with open(local_filename, 'rb') as f:
                            header = f.read(10)
                            if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK'):
                                st.success(f"✅ مدل با موفقیت دانلود شد! حجم: {file_size / (1024*1024):.1f} مگابایت")
                                return local_filename
                            else:
                                st.warning(f"⚠️ روش {i+1}: فایل دانلود شده معتبر نیست")
                                if os.path.exists(local_filename):
                                    os.remove(local_filename)
                    except Exception as e:
                        st.warning(f"⚠️ روش {i+1}: خطا در بررسی فایل: {e}")
                        if os.path.exists(local_filename):
                            os.remove(local_filename)
                else:
                    st.warning(f"⚠️ روش {i+1}: فایل دانلود شده خالی است")
                    if os.path.exists(local_filename):
                        os.remove(local_filename)
                        
            except Exception as e:
                st.warning(f"⚠️ روش {i+1} با خطا مواجه شد: {str(e)}")
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                continue
        
        st.info("🔄 در حال تلاش با روش دستی...")
        return manual_download_fallback(correct_file_id, local_filename)
            
    except Exception as e:
        st.error(f"❌ خطا در تابع دانلود: {str(e)}")
        return None

def manual_download_fallback(file_id, local_filename):
    """روش دستی دانلود با requests"""
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
                st.info(f"🔄 روش دستی {i+1} از 3...")
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
                                        status_text.text(f"دانلود شده: {downloaded_size / (1024*1024):.1f} از {total_size / (1024*1024):.1f} مگابایت")
                        
                        if total_size > 0:
                            progress_bar.empty()
                            status_text.empty()
                        
                        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                            file_size = os.path.getsize(local_filename)
                            st.success(f"✅ دانلود دستی موفق! حجم: {file_size / (1024*1024):.1f} مگابایت")
                            return local_filename
                    else:
                        st.warning(f"⚠️ روش {i+1} بجای فایل، HTML برگرداند")
                        
            except Exception as e:
                st.warning(f"⚠️ روش دستی {i+1} با خطا مواجه شد: {e}")
                continue
        
        st.error("❌ تمام روش‌های دانلود خودکار با شکست مواجه شدند")
        
        st.markdown(f"""
        **📋 راهنمای دانلود دستی:**
        
        1. این لینک را در مرورگر باز کنید: 
           https://drive.google.com/file/d/{file_id}/view
        
        2. دکمه دانلود را کلیک کنید
        
        3. فایل را با نام `{local_filename}` ذخیره کنید
        
        4. از بخش زیر فایل را آپلود کنید
        """)
        
        uploaded_file = st.file_uploader(
            f"📤 فایل مدل را آپلود کنید ({local_filename}):",
            type=['pt', 'pth'],
            help="فایل را از گوگل درایو دانلود کرده و اینجا آپلود کنید"
        )
        
        if uploaded_file is not None:
            with open(local_filename, 'wb') as f:
                f.write(uploaded_file.read())
            
            file_size = os.path.getsize(local_filename)
            st.success(f"✅ مدل با موفقیت آپلود شد! حجم: {file_size / (1024*1024):.1f} مگابایت")
            return local_filename
        
        return None
        
    except Exception as e:
        st.error(f"❌ خطا در دانلود دستی: {e}")
        return None

# Model loading section
gdrive_model_url = "https://drive.google.com/file/d/1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov/view?usp=drive_link"
model_path = "best_model_version_Unet++_v02_e7.pt"

if not os.path.exists(model_path):
    st.info("📂 مدل در سیستم یافت نشد. در حال دانلود از گوگل درایو...")
    
    downloaded_model_path = download_model_from_gdrive(gdrive_model_url, model_path)
    
    if downloaded_model_path is None:
        st.error("❌ دانلود خودکار ناموفق بود. لطفا از گزینه دانلود دستی استفاده کنید")
        st.stop()
else:
    st.success("✅ مدل در سیستم یافت شد!")

# Verify the model file
if os.path.exists(model_path):
    try:
        file_size = os.path.getsize(model_path)
        st.info(f"📊 حجم فایل مدل: {file_size / (1024*1024):.1f} مگابایت")
        
        with open(model_path, 'rb') as f:
            header = f.read(10)
            if not (header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK')):
                st.error("❌ فایل مدل خراب یا نامعتبر است")
                os.remove(model_path)
                st.stop()
            else:
                st.success("✅ فایل مدل معتبر است!")
                
    except Exception as e:
        st.error(f"❌ خطا در بررسی فایل مدل: {e}")
        st.stop()

# Install GEES2Downloader
try:
    from geeS2downloader.geeS2downloader import GEES2Downloader
    st.sidebar.success("✅ GEES2Downloader نصب شده است")
except ImportError:
    st.sidebar.info("📦 در حال نصب GEES2Downloader...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/cordmaur/GEES2Downloader.git"
        ])
        
        from geeS2downloader.geeS2downloader import GEES2Downloader
        st.sidebar.success("✅ GEES2Downloader با موفقیت نصب شد!")
    except Exception as e:
        st.sidebar.error(f"❌ خطا در نصب GEES2Downloader: {str(e)}")

# Initialize Earth Engine
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize()
        return True, "✅ Earth Engine آماده است"
    except Exception as e:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if not base64_key:
                return False, "❌ کلید احراز هویت Earth Engine یافت نشد"
            
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
            
            return True, "✅ احراز هویت Earth Engine موفق بود!"
        except Exception as auth_error:
            return False, f"❌ خطا در احراز هویت: {str(auth_error)}"

ee_initialized, ee_message = initialize_earth_engine()
if ee_initialized:
    st.sidebar.success(ee_message)
else:
    st.sidebar.error(ee_message)
    st.error("❌ احراز هویت Earth Engine الزامی است")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ انتخاب منطقه", 
    "📅 تصویر قبل", 
    "📅 تصویر بعد", 
    "🔍 تشخیص تغییرات"
])

# Session state initialization
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

# Sentinel-2 bands
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

# Helper functions
def download_sentinel2_with_gees2(year, polygon, start_month, end_month, cloud_cover_limit=10):
    """دانلود تصاویر Sentinel-2"""
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
        status_placeholder.info(f"📏 مساحت منطقه: ~{area_sq_km:.2f} کیلومتر مربع")
        
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, f"sentinel2_{year}_{start_month:02d}_{end_month:02d}_median.tif")
        
        geojson = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
        ee_geometry = ee.Geometry.Polygon(geojson['coordinates'])
        
        status_placeholder.info(f"🔍 جستجوی تصاویر از {start_date} تا {end_date}...")
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(ee_geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_limit)))
        
        count = collection.size().getInfo()
        status_placeholder.info(f"✅ {count} تصویر با ابر کمتر از {cloud_cover_limit}٪ یافت شد")
        
        if count == 0:
            status_placeholder.warning(f"⚠️ تصویری یافت نشد. افزایش محدوده ابر...")
            
            higher_limit = min(cloud_cover_limit * 2, 100)
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(ee_geometry)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', higher_limit)))
            
            count = collection.size().getInfo()
            
            if count == 0:
                status_placeholder.error("❌ هیچ تصویری یافت نشد")
                return None
        
        status_placeholder.info(f"🔄 ایجاد تصویر میانه از {count} تصویر...")
        median_image = collection.median().select(S2_BANDS)
        
        def progress_callback(progress):
            progress_placeholder.progress(progress)
        
        bands_dir = os.path.join(temp_dir, "bands")
        os.makedirs(bands_dir, exist_ok=True)
        
        status_placeholder.info("⬇️ در حال دانلود باندها...")
        band_files = []
        region = ee_geometry.bounds().getInfo()['coordinates']
        
        for i, band in enumerate(S2_BANDS):
            try:
                status_placeholder.info(f"⬇️ دانلود باند {band} ({i+1}/{len(S2_BANDS)})...")
                
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
                    status_placeholder.error(f"❌ خطا در دانلود باند {band}")
            except Exception as e:
                status_placeholder.error(f"❌ خطا در دانلود باند {band}: {str(e)}")
        
        if len(band_files) == len(S2_BANDS):
            status_placeholder.info("🔄 ایجاد فایل چندباندی...")
            
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            
            meta.update(count=len(band_files))
            
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            status_placeholder.success("✅ فایل چندباندی با موفقیت ایجاد شد")
            return output_file
        else:
            status_placeholder.error(f"❌ فقط {len(band_files)}/{len(S2_BANDS)} باند دانلود شد")
            return None
        
    except Exception as e:
        status_placeholder.error(f"❌ خطا در دانلود: {str(e)}")
        return None

def normalized(img):
    """نرمال‌سازی تصویر"""
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm

def get_utm_zone(longitude):
    """تعیین منطقه UTM"""
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    """تعیین کد EPSG برای منطقه UTM"""
    zone_number = get_utm_zone(longitude)
    
    if latitude >= 0:
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"

def convert_to_utm(src_path, dst_path, polygon=None):
    """تبدیل سیستم مختصات به UTM"""
    with rasterio.open(src_path) as src:
        if polygon:
            centroid = polygon.centroid
            lon, lat = centroid.x, centroid.y
            dst_crs = get_utm_epsg(lon, lat)
            st.info(f"🗺️ منطقه UTM تشخیص داده شد: {get_utm_zone(lon)} ({dst_crs})")
        else:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            dst_crs = get_utm_epsg(center_lon, center_lat)
            st.info(f"🗺️ منطقه UTM تشخیص داده شد: {get_utm_zone(center_lon)} ({dst_crs})")
        
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src
