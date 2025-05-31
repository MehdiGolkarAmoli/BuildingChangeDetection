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
st.set_page_config(layout="wide", page_title="Satellite Image Analysis Tool")

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
    
    Parameters:
    gdrive_url (str): Google Drive sharing URL
    local_filename (str): Local filename to save the downloaded file
    
    Returns:
    str: Path to the downloaded file or None if download failed
    """
    try:
        # FIXED: Use the correct file ID from your URL
        # URL: https://drive.google.com/file/d/1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov/view?usp=drive_link
        correct_file_id = "1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov"
        
        st.info(f"Downloading model from Google Drive (File ID: {correct_file_id})...")
        
        # Use gdown library which is more reliable for Google Drive downloads
        try:
            import gdown
        except ImportError:
            st.info("Installing gdown library for reliable Google Drive downloads...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        # Try multiple download approaches
        download_methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id  # Sometimes gdown works with just the file ID
        ]
        
        for i, method in enumerate(download_methods):
            try:
                st.info(f"Trying download method {i+1}/3...")
                
                # Use gdown with fuzzy matching
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                
                # Verify the downloaded file
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    file_size = os.path.getsize(local_filename)
                    
                    # Verify it's a valid PyTorch file
                    try:
                        with open(local_filename, 'rb') as f:
                            header = f.read(10)
                            if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK'):
                                st.success(f"Model downloaded successfully! Size: {file_size / (1024*1024):.1f} MB")
                                return local_filename
                            else:
                                st.warning(f"Method {i+1}: Downloaded file doesn't appear to be a valid PyTorch model. Header: {header}")
                                if os.path.exists(local_filename):
                                    os.remove(local_filename)
                    except Exception as e:
                        st.warning(f"Method {i+1}: Error verifying downloaded file: {e}")
                        if os.path.exists(local_filename):
                            os.remove(local_filename)
                else:
                    st.warning(f"Method {i+1}: Downloaded file is empty or doesn't exist")
                    if os.path.exists(local_filename):
                        os.remove(local_filename)
                        
            except Exception as e:
                st.warning(f"Download method {i+1} failed: {str(e)}")
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                continue
        
        # If all gdown methods failed, try manual requests approach
        st.info("All gdown methods failed. Trying manual download approach...")
        return manual_download_fallback(correct_file_id, local_filename)
            
    except Exception as e:
        st.error(f"Error in download function: {str(e)}")
        return None

def manual_download_fallback(file_id, local_filename):
    """
    Fallback manual download method using requests
    """
    try:
        import requests
        
        # Try different download URLs
        urls_to_try = [
            f"https://drive.google.com/uc?export=download&id={file_id}",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
        ]
        
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        for i, url in enumerate(urls_to_try):
            try:
                st.info(f"Trying manual method {i+1}/3...")
                response = session.get(url, headers=headers, stream=True, timeout=30)
                
                if response.status_code == 200:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/html' not in content_type:
                        # Create progress bar
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded_size = 0
                        
                        if total_size > 0:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        # Download the file
                        with open(local_filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    
                                    if total_size > 0:
                                        progress = downloaded_size / total_size
                                        progress_bar.progress(progress)
                                        status_text.text(f"Downloaded: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        
                        if total_size > 0:
                            progress_bar.empty()
                            status_text.empty()
                        
                        # Verify file
                        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                            file_size = os.path.getsize(local_filename)
                            st.success(f"Manual download successful! Size: {file_size / (1024*1024):.1f} MB")
                            return local_filename
                    else:
                        st.warning(f"Method {i+1} returned HTML instead of file")
                        
            except Exception as e:
                st.warning(f"Manual method {i+1} failed: {e}")
                continue
        
        # All methods failed - provide manual instructions
        st.error("All automatic download methods failed. Please download manually:")
        
        st.info("**Manual Download Instructions:**")
        st.markdown(f"""
        1. **Open this link in a new browser tab:** 
           https://drive.google.com/file/d/{file_id}/view
        
        2. **If you see a permission error:**
           - The file owner needs to change sharing to "Anyone with the link can view"
           - Or you may need to request access
        
        3. **Click the Download button** (usually in the top-right corner)
        
        4. **Save the file as:** `{local_filename}`
        
        5. **Upload it using the file uploader below**
        """)
        
        # File uploader as backup
        uploaded_file = st.file_uploader(
            f"Upload the model file ({local_filename}) after manual download:",
            type=['pt', 'pth'],
            help="Download the file manually from Google Drive and upload it here"
        )
        
        if uploaded_file is not None:
            with open(local_filename, 'wb') as f:
                f.write(uploaded_file.read())
            
            file_size = os.path.getsize(local_filename)
            st.success(f"Model uploaded successfully! Size: {file_size / (1024*1024):.1f} MB")
            return local_filename
        
        return None
        
    except Exception as e:
        st.error(f"Manual download fallback failed: {e}")
        return None

# CORRECTED: Updated model loading section with the correct URL and file ID
gdrive_model_url = "https://drive.google.com/file/d/1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov/view?usp=drive_link"
model_path = "best_model_version_Unet++_v02_e7.pt"  # Updated filename to match your preference

# Download model if it doesn't exist locally
if not os.path.exists(model_path):
    st.info("Model not found locally. Downloading from Google Drive...")
    
    # Try downloading with the corrected URL
    downloaded_model_path = download_model_from_gdrive(gdrive_model_url, model_path)
    
    if downloaded_model_path is None:
        st.error("Automatic download failed. Please use the manual download option above.")
        st.stop()
else:
    st.success("Model found locally!")

# Verify the model file before proceeding
if os.path.exists(model_path):
    try:
        file_size = os.path.getsize(model_path)
        st.info(f"Model file size: {file_size / (1024*1024):.1f} MB")
        
        # Try to load just the header to verify it's a valid PyTorch file
        with open(model_path, 'rb') as f:
            header = f.read(10)
            if not (header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK')):
                st.error("The model file appears to be corrupted or invalid.")
                st.error(f"File header: {header}")
                st.info("Please re-download the model file.")
                
                # Show file content preview for debugging
                try:
                    with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(200)
                        st.code(content, language='text')
                except:
                    pass
                
                os.remove(model_path)
                st.stop()
            else:
                st.success("Model file appears to be valid!")
                
    except Exception as e:
        st.error(f"Error verifying model file: {e}")
        st.stop()

# Install GEES2Downloader if not already installed
try:
    from geeS2downloader.geeS2downloader import GEES2Downloader
    st.sidebar.success("GEES2Downloader is already installed.")
except ImportError:
    st.sidebar.info("Installing GEES2Downloader...")
    try:
        # Install directly from GitHub
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/cordmaur/GEES2Downloader.git"
        ])
        
        # Import the module
        from geeS2downloader.geeS2downloader import GEES2Downloader
        st.sidebar.success("GEES2Downloader installed successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to install GEES2Downloader: {str(e)}")
        st.sidebar.info("Please manually install GEES2Downloader")

# Initialize Earth Engine with service account from environment variable
@st.cache_resource
def initialize_earth_engine():
    try:
        # Check if Earth Engine is already initialized
        ee.Initialize()
        return True, "Earth Engine already initialized"
    except Exception as e:
        try:
            # Try service account authentication from environment variable
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if not base64_key:
                return False, "Earth Engine service account key not found in environment variables."
            
            # Decode the base64 string
            key_json = base64.b64decode(base64_key).decode()
            key_data = json.loads(key_json)
            
            # Create a temporary file for the key
            key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            with open(key_file.name, 'w') as f:
                json.dump(key_data, f)
            
            # Initialize Earth Engine with the service account credentials
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'],
                key_file.name
            )
            ee.Initialize(credentials)
            
            # Clean up the temporary file
            os.unlink(key_file.name)
            
            return True, "Successfully authenticated with Earth Engine!"
        except Exception as auth_error:
            return False, f"Authentication failed: {str(auth_error)}"

# Initialize Earth Engine
ee_initialized, ee_message = initialize_earth_engine()
if ee_initialized:
    st.sidebar.success(ee_message)
else:
    st.sidebar.error(ee_message)
    st.error("Earth Engine authentication is required to use this application.")
    st.info("""
    Please set the GOOGLE_EARTH_ENGINE_KEY_BASE64 environment variable with your service account key.
    
    1. Create a service account in Google Cloud Console
    2. Generate a JSON key for the service account
    3. Convert the JSON key to base64:
       ```python
       import base64
       with open('your-key.json', 'r') as f:
           print(base64.b64encode(f.read().encode()).decode())
       ```
    4. Set this as an environment variable in your Posit Cloud environment
    """)
    st.stop()

# Create tabs for different pages
tab1, tab2, tab3, tab4  = st.tabs(["Region Selection", "Before Image Analysis", "After Image Analysis", "Change Detection"])

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

# Define Sentinel-2 bands to use
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S2_NAMES = ['Aerosols', 'Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 
           'Red Edge 3', 'NIR', 'Red Edge 4', 'Water Vapor', 'SWIR1', 'SWIR2']

# [Rest of your code continues unchanged from here...]
# The rest of your functions and code remain exactly the same

# Function to download Sentinel-2 imagery using GEES2Downloader
def download_sentinel2_with_gees2(year, polygon, start_month, end_month, cloud_cover_limit=10):
    """
    Download Sentinel-2 Level-2A imagery for a specific year and region using GEES2Downloader.
    
    Parameters:
    year (str): Year to download imagery for (e.g., "2021")
    polygon (shapely.geometry.Polygon): Region of interest
    start_month (int): Start month (1-12)
    end_month (int): End month (1-12)
    cloud_cover_limit (int): Maximum cloud cover percentage
    
    Returns:
    str: Path to the downloaded GeoTIFF file or None if no suitable images found
    """
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    try:
        # Define date ranges for the specified year and months
        start_date = f"{year}-{start_month:02d}-01"
        
        # Calculate end date based on month (handle different month lengths)
        if end_month in [4, 6, 9, 11]:  # 30 days
            end_day = 30
        elif end_month == 2:  # February
            end_day = 29 if int(year) % 4 == 0 else 28  # Simple leap year check
        else:  # 31 days
            end_day = 31
            
        end_date = f"{year}-{end_month:02d}-{end_day}"
        
        # Display the region area in square kilometers
        area_sq_km = polygon.area * 111 * 111  # Approximate conversion from degrees to km²
        status_placeholder.info(f"Selected region area: ~{area_sq_km:.2f} km². Searching for Sentinel-2 images...")
        
        # Create a temporary directory for downloading
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, f"sentinel2_{year}_{start_month:02d}_{end_month:02d}_median.tif")
        
        # Convert polygon to GeoJSON format
        geojson = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
        
        # Initialize Earth Engine geometry
        ee_geometry = ee.Geometry.Polygon(geojson['coordinates'])
        
        # Create Earth Engine image collection
        status_placeholder.info(f"Creating Earth Engine image collection for {start_date} to {end_date} with cloud cover < {cloud_cover_limit}%...")
        
        # Filter Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(ee_geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_limit)))
        
        # Get the count of images
        count = collection.size().getInfo()
        status_placeholder.info(f"Found {count} Sentinel-2 images with cloud cover < {cloud_cover_limit}%")
        
        if count == 0:
            status_placeholder.warning(f"No Sentinel-2 images found for {year} ({start_month:02d}-{end_month:02d}) with cloud cover < {cloud_cover_limit}%")
            
            # Try with higher cloud cover
            higher_limit = min(cloud_cover_limit * 2, 100)
            status_placeholder.info(f"Trying with higher cloud cover limit ({higher_limit}%)...")
            
            # Create a new collection with higher cloud cover limit
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(ee_geometry)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', higher_limit)))
            
            count = collection.size().getInfo()
            status_placeholder.info(f"Found {count} Sentinel-2 images with cloud cover < {higher_limit}%")
            
            if count == 0:
                status_placeholder.error(f"No Sentinel-2 images found for {year} ({start_month:02d}-{end_month:02d}) even with higher cloud cover limit.")
                return None
        
        # Create median composite
        status_placeholder.info(f"Creating median composite from {count} images...")
        median_image = collection.median().select(S2_BANDS)
        
        # Set up progress callback
        def progress_callback(progress):
            progress_placeholder.progress(progress)
        
        # Create a directory for individual band downloads
        bands_dir = os.path.join(temp_dir, "bands")
        os.makedirs(bands_dir, exist_ok=True)
        
        # Manually download each band using Earth Engine's getDownloadURL
        status_placeholder.info("Downloading bands individually...")
        band_files = []
        
        # Get the region bounds for download
        region = ee_geometry.bounds().getInfo()['coordinates']
        
        for i, band in enumerate(S2_BANDS):
            try:
                status_placeholder.info(f"Downloading band {band} ({i+1}/{len(S2_BANDS)})...")
                
                # Create a filename for this band
                band_file = os.path.join(bands_dir, f"{band}.tif")
                
                # Get the download URL for this specific band
                url = median_image.select(band).getDownloadURL({
                    'scale': 10,  # 10m resolution
                    'region': region,
                    'format': 'GEO_TIFF',
                    'bands': [band]
                })
                
                # Download the file
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(band_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    band_files.append(band_file)
                    progress_callback((i + 1) / len(S2_BANDS))
                else:
                    status_placeholder.error(f"Failed to download band {band}: HTTP status {response.status_code}")
            except Exception as e:
                status_placeholder.error(f"Error downloading band {band}: {str(e)}")
        
        # Check if we have downloaded all bands
        if len(band_files) == len(S2_BANDS):
            status_placeholder.info("All bands downloaded. Creating multiband GeoTIFF...")
            
            # Read the first band to get metadata
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            
            # Update metadata for multiband output
            meta.update(count=len(band_files))
            
            # Create the output file
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            status_placeholder.success("Successfully created multiband GeoTIFF")
            return output_file
        else:
            status_placeholder.error(f"Only downloaded {len(band_files)}/{len(S2_BANDS)} bands")
            return None
        
    except Exception as e:
        status_placeholder.error(f"Error downloading Sentinel-2 data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Helper function to normalize image data
def normalized(img):
    """
    Normalize image data to range [0, 1]
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Normalized image
    """
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm

# Function to determine UTM zone from longitude
def get_utm_zone(longitude):
    """
    Determine the UTM zone for a given longitude
    
    Parameters:
    longitude (float): Longitude in decimal degrees
    
    Returns:
    int: UTM zone number
    """
    return math.floor((longitude + 180) / 6) + 1

# Function to determine UTM EPSG code from longitude and latitude
def get_utm_epsg(longitude, latitude):
    """
    Determine the EPSG code for UTM zone based on longitude and latitude
    
    Parameters:
    longitude (float): Longitude in decimal degrees
    latitude (float): Latitude in decimal degrees
    
    Returns:
    str: EPSG code for the appropriate UTM zone
    """
    zone_number = get_utm_zone(longitude)
    
    # Northern hemisphere if latitude >= 0, Southern hemisphere if latitude < 0
    if latitude >= 0:
        # Northern hemisphere EPSG: 326xx where xx is the UTM zone
        return f"EPSG:326{zone_number:02d}"
    else:
        # Southern hemisphere EPSG: 327xx where xx is the UTM zone
        return f"EPSG:327{zone_number:02d}"

# Function to convert coordinate system from WGS-84 to appropriate UTM Zone
def convert_to_utm(src_path, dst_path, polygon=None):
    """
    Reproject a GeoTIFF from WGS-84 to the appropriate UTM zone
    
    Parameters:
    src_path (str): Path to source GeoTIFF in WGS-84
    dst_path (str): Path to save reprojected GeoTIFF
    polygon (shapely.geometry.Polygon, optional): Polygon to determine UTM zone
    
    Returns:
    str: Path to the reprojected file and the EPSG code used
    """
    with rasterio.open(src_path) as src:
        # If polygon is provided, use its centroid to determine UTM zone
        if polygon:
            centroid = polygon.centroid
            lon, lat = centroid.x, centroid.y
            dst_crs = get_utm_epsg(lon, lat)
            st.info(f"Automatically detected UTM zone: {get_utm_zone(lon)} ({dst_crs})")
        else:
            # Use the center of the image to determine UTM zone
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            dst_crs = get_utm_epsg(center_lon, center_lat)
            st.info(f"Automatically detected UTM zone: {get_utm_zone(center_lon)} ({dst_crs})")
        
        # Calculate the transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Update the metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create the output file
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            # Reproject each band
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

# Function to apply morphological erosion
def apply_erosion(image, kernel_size):
    """
    Apply morphological erosion to a binary image
    
    Parameters:
    image (numpy.ndarray): Binary image to erode
    kernel_size (int): Size of the structuring element
    
    Returns:
    numpy.ndarray: Eroded binary image
    """
    # Create a structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Apply erosion
    eroded_image = ndimage.binary_erosion(image, structure=kernel).astype(image.dtype)
    
    return eroded_image

@st.cache_resource
def load_model(model_path):
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b7',
            encoder_weights='imagenet',
            in_channels=12,  # 12 bands for Sentinel-2
            classes=1,
            decoder_attention_type='scse'
        ).to(device)

        # Try loading the state_dict directly
        loaded_object = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            # It's a checkpoint dictionary
            model.load_state_dict(loaded_object['model_state_dict'])
            st.info("Model loaded from checkpoint dictionary.")
        elif isinstance(loaded_object, dict):
            # It's a raw state_dict
            model.load_state_dict(loaded_object)
            st.info("Model loaded directly from state_dict.")
        else:
            # This case might occur if torch.save(model, PATH) was used
            # and loaded_object is the model itself.
            # However, smp.UnetPlusPlus is already instantiated above.
            # This path is less likely for .pt files meant for state_dict loading.
            st.error("Loaded model file is not a recognized state_dict or checkpoint format.")
            st.session_state.model_loaded = False
            return None, None

        model.eval()
        st.session_state.model_loaded = True # Indicate model is loaded
        st.success("Model loaded successfully!") # Add success message
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False # Indicate model loading failed
        return None, None

# [Rest of your code continues unchanged from here...]
# Function to process image with better error handling for non-overlapping geometries
def process_image(image_path, year, selected_polygon, region_number):
    """
    Complete image processing pipeline: clip, patch, classify, and reconstruct
    
    Parameters:
    image_path (str): Path to the Sentinel-2 image
    year (str): Year of the image (e.g., "2021")
    selected_polygon (shapely.geometry.Polygon): Selected region polygon
    region_number (int): Region number
    
    Returns:
    bool: True if processing was successful, False otherwise
    """
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # 1. CLIPPING STEP
        status_placeholder.info("Step 1/4: Clipping image...")
        progress_placeholder.progress(0)
        
        # Open the raster file
        with rasterio.open(image_path) as src:
            # Check if the polygon overlaps with the raster bounds
            raster_bounds = box(*src.bounds)
            polygon_shapely = selected_polygon
            
            if not raster_bounds.intersects(polygon_shapely):
                status_placeholder.error("Error: The selected region doesn't overlap with the downloaded Sentinel-2 image.")
                st.error(f"Selected region bounds: {polygon_shapely.bounds}")
                st.error(f"Raster bounds: {src.bounds}")
                
                # Display the issue visually
                fig, ax = plt.subplots(figsize=(10, 10))
                # Plot raster bounds
                x, y = raster_bounds.exterior.xy
                ax.plot(x, y, color='red', linewidth=2, label='Raster Bounds')
                # Plot selected polygon
                x, y = polygon_shapely.exterior.xy
                ax.plot(x, y, color='blue', linewidth=2, label='Selected Region')
                ax.set_title("Geometry Issue: No Overlap")
                ax.legend()
                st.pyplot(fig)
                
                return False
            
            # Convert polygon to GeoJSON format for rasterio
            geoms = [mapping(selected_polygon)]
            
            try:
                # Perform the clipping
                clipped_img, clipped_transform = mask(src, geoms, crop=True)
                
                # Check if the clipped image has valid data
                if clipped_img.size == 0 or np.all(clipped_img == 0):
                    status_placeholder.error("Error: Clipping resulted in an empty image. The region may not overlap with valid data.")
                    return False
                
            except ValueError as e:
                if "Input shapes do not overlap raster" in str(e):
                    status_placeholder.error("Error: The selected region doesn't overlap with valid data in the image.")
                    st.error("This can happen if the region is outside the image bounds or in an area with no data.")
                    
                    # Try to visualize the issue
                    fig, ax = plt.subplots(figsize=(10, 10))
                    # Show a simple RGB preview of the raster if possible
                    try:
                        preview = src.read([4, 3, 2]) # Try to read RGB bands
                        preview = np.dstack([preview[i, :, :] for i in range(3)])
                        preview = (preview - preview.min()) / (preview.max() - preview.min())
                        ax.imshow(preview)
                        ax.set_title("Raster Overview (RGB)")
                    except:
                        ax.set_title("Could not display raster overview")
                    
                    st.pyplot(fig)
                    return False
                else:
                    raise
            
            # Store the clipped image metadata
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "height": clipped_img.shape[1],
                "width": clipped_img.shape[2],
                "transform": clipped_transform
            })
            
            # Save the clipped image to a temporary file with original CRS
            temp_dir = os.path.dirname(image_path)
            os.makedirs(os.path.join(temp_dir, "temp"), exist_ok=True)
            temp_clipped_path = os.path.join(temp_dir, "temp", f"temp_clipped_{year}_region{region_number}.tif")
            
            with rasterio.open(temp_clipped_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_img)
            
            # Convert to appropriate UTM Zone based on the polygon's location
            utm_clipped_path = os.path.join(temp_dir, "temp", f"utm_clipped_{year}_region{region_number}.tif")
            utm_path, utm_crs = convert_to_utm(temp_clipped_path, utm_clipped_path, selected_polygon)
            
            # Read back the UTM version
            with rasterio.open(utm_path) as src_utm:
                clipped_img = src_utm.read()
                clipped_meta = src_utm.meta.copy()
            
            # Store clipped image in session state based on year
            if year == st.session_state.before_year:
                st.session_state.clipped_img = clipped_img
                st.session_state.clipped_meta = clipped_meta
                st.session_state.region_number = region_number
                st.session_state.year = year
            else: # after year
                st.session_state.clipped_img_2024 = clipped_img
                st.session_state.clipped_meta_2024 = clipped_meta
                st.session_state.region_number_2024 = region_number
                st.session_state.year_2024 = year
            
            # Check if the clipped image is large enough for patching
            if clipped_img.shape[1] < 300 or clipped_img.shape[2] < 300:
                status_placeholder.error(f"The clipped image is too small ({clipped_img.shape[1]}x{clipped_img.shape[2]} pixels). Please select a larger area (minimum 300x300 pixels).")
                return False
            
            # Display the clipped image
            rgb_bands = [3, 2, 1] # 0-indexed for bands 4, 3, 2
            
            if clipped_img.shape[0] >= 3:
                rgb = np.zeros((clipped_img.shape[1], clipped_img.shape[2], 3), dtype=np.float32)
                
                # Normalize and assign bands
                for i, band in enumerate(rgb_bands):
                    if band < clipped_img.shape[0]:
                        band_data = clipped_img[band]
                        # Simple contrast stretch for visualization
                        min_val = np.percentile(band_data, 2)
                        max_val = np.percentile(band_data, 98)
                        rgb[:, :, i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                
                # Display the RGB image
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb)
                ax.set_title(f"Clipped Sentinel-2 {year} Image ({clipped_meta['crs']})")
                ax.axis('off')
                st.pyplot(fig)
            
            progress_placeholder.progress(25)
            status_placeholder.success("Step 1/4: Clipping completed successfully")
        
        # 2. PATCHING STEP
        status_placeholder.info("Step 2/4: Creating patches...")
        
        # Prepare the image for patching
        img_for_patching = np.moveaxis(clipped_img, 0, -1) # Change to H x W x C format for patchify
        
        # Create patches
        patch_size = 224
        patches = patchify(img_for_patching, (patch_size, patch_size, clipped_img.shape[0]), step=patch_size)
        patches = patches.squeeze()
        
        # Create output directory
        base_dir = os.path.dirname(image_path)
        output_folder = os.path.join(base_dir, f"patches_{year}_region{region_number}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Save patches
        num_patches = patches.shape[0] * patches.shape[1]
        saved_paths = []
        
        # Create a sub-progress bar for patches
        patch_progress = st.progress(0)
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j] # Get the individual patch
                
                # Normalize the patch
                patch_normalized = normalized(patch)
                
                # Convert to rasterio format (C x H x W)
                patch_for_saving = np.moveaxis(patch_normalized, -1, 0)
                
                # Create filename
                patch_name = f"region{region_number}_{i}_{j}.tif"
                output_file_path = os.path.join(output_folder, patch_name)
                saved_paths.append(output_file_path)
                
                # Save the patch as GeoTIFF with the detected UTM CRS
                # Fix: Properly reshape the data when writing each band
                with rasterio.open(
                    output_file_path,
                    'w',
                    driver='GTiff',
                    height=patch_size,
                    width=patch_size,
                    count=patch_for_saving.shape[0],
                    dtype='float64',
                    crs=clipped_meta.get('crs'),
                    transform=clipped_meta.get('transform')
                ) as dst:
                    # Write each band separately with proper reshaping
                    for band_idx in range(patch_for_saving.shape[0]):
                        band_data = patch_for_saving[band_idx].reshape(patch_size, patch_size)
                        dst.write(band_data, band_idx+1)
                
                # Update patch progress
                patch_count = i * patches.shape[1] + j + 1
                patch_progress.progress(patch_count / num_patches)
        
        # Store the saved paths in session state for classification
        if year == st.session_state.before_year:
            st.session_state.saved_patches_paths = saved_paths
            st.session_state.patches_shape = patches.shape
            st.session_state.patches_info = {
                'year': year,
                'region_number': region_number,
                'output_folder': output_folder,
                'patch_size': patch_size
            }
        else: # after year
            st.session_state.saved_patches_paths_2024 = saved_paths
            st.session_state.patches_shape_2024 = patches.shape
            st.session_state.patches_info_2024 = {
                'year': year,
                'region_number': region_number,
                'output_folder': output_folder,
                'patch_size': patch_size
            }
        
        # Display sample patches
        num_samples = min(6, num_patches)
        if num_samples > 0:
            st.subheader("Sample Patches")
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
            if num_samples == 1:
                axes = [axes] # Make it iterable if only one subplot
            
            for idx, ax in enumerate(axes):
                if idx < num_patches:
                    i, j = idx // patches.shape[1], idx % patches.shape[1]
                    patch = patches[i, j]
                    
                    # For visualization, use the same RGB bands as the main image
                    rgb_bands = [3, 2, 1] # Adjust based on your data
                    patch_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                    for b, band in enumerate(rgb_bands):
                        if band < patch.shape[-1]:
                            band_data = patch[:, :, band]
                            min_val = np.percentile(band_data, 2)
                            max_val = np.percentile(band_data, 98)
                            patch_rgb[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                    
                    ax.imshow(patch_rgb)
                    ax.set_title(f"Patch {i}_{j}")
                    ax.axis('off')
            
            st.pyplot(fig)
        
        progress_placeholder.progress(50)
        status_placeholder.success(f"Step 2/4: Created {num_patches} patches successfully")
        
        # 3. CLASSIFICATION STEP
        status_placeholder.info("Step 3/4: Classifying patches...")
        
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                model, device = load_model(model_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                else:
                    status_placeholder.error("Failed to load model. Please check the model path.")
                    return False
        
        # Get saved paths based on year
        if year == st.session_state.before_year:
            patches_info = st.session_state.patches_info
            saved_paths = st.session_state.saved_patches_paths
            patches_shape = st.session_state.patches_shape
        else: # after year
            patches_info = st.session_state.patches_info_2024
            saved_paths = st.session_state.saved_patches_paths_2024
            patches_shape = st.session_state.patches_shape_2024
        
        # Create output directory for classified images
        base_dir = os.path.dirname(image_path)
        classified_folder = os.path.join(base_dir, f"classified_{year}_region{region_number}")
        os.makedirs(classified_folder, exist_ok=True)
        
        # Track classified images and original images for display
        classified_results = []
        classified_paths = []
        
        # Create a sub-progress bar for classification
        classify_progress = st.progress(0)
        
        # Process each patch
        total_patches = len(saved_paths)
        for idx, patch_path in enumerate(saved_paths):
            try:
                # Extract i, j from filename
                filename = os.path.basename(patch_path)
                i_j_part = filename.split('_')[-2:]
                i = int(i_j_part[0])
                j = int(i_j_part[1].split('.')[0])
                
                # Read the patch
                with rasterio.open(patch_path) as src:
                    patch = src.read() # This will be in (C, H, W) format
                    patch_meta = src.meta.copy()
                
                # Create RGB version of original patch for display
                rgb_bands = [3, 2, 1] # Adjust based on your data
                if patch.shape[0] >= 3:
                    rgb_patch = np.zeros((patch.shape[1], patch.shape[2], 3), dtype=np.float32)
                    
                    # Normalize and assign bands
                    for b, band in enumerate(rgb_bands):
                        if band < patch.shape[0]:
                            band_data = patch[band]
                            # Simple contrast stretch for visualization
                            min_val = np.percentile(band_data, 2)
                            max_val = np.percentile(band_data, 98)
                            rgb_patch[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                
                # Convert to torch tensor
                img_tensor = torch.tensor(patch, dtype=torch.float32)
                img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
                
                # Perform classification
                with torch.inference_mode():
                    prediction = st.session_state.model(img_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                # Convert prediction to numpy
                pred_np = prediction.squeeze().numpy()
                
                # Create binary mask (threshold at 0.5)
                binary_mask = (pred_np > 0.5).astype(np.uint8) * 255
                
                # Save classified result
                output_filename = f"classified_region{patches_info['region_number']}_{i}_{j}.tif"
                output_path = os.path.join(classified_folder, output_filename)
                classified_paths.append(output_path)
                
                # Save as GeoTIFF with the same CRS as the patch
                patch_meta.update({
                    'count': 1,
                    'dtype': 'uint8'
                })
                
                with rasterio.open(
                    output_path,
                    'w',
                    **patch_meta
                ) as dst:
                    dst.write(binary_mask.reshape(1, binary_mask.shape[0], binary_mask.shape[1]))
                
                # Add to display list (limit to 6)
                if len(classified_results) < 6:
                    classified_results.append({
                        'path': output_path,
                        'i': i,
                        'j': j,
                        'mask': binary_mask,
                        'rgb_original': rgb_patch
                    })
                
                # Update classification progress
                classify_progress.progress((idx + 1) / total_patches)
                
            except Exception as e:
                st.error(f"Error processing patch {patch_path}: {str(e)}")
        
        # Store classified paths in session state
        if year == st.session_state.before_year:
            st.session_state.classified_paths = classified_paths
            st.session_state.classified_shape = patches_shape
        else: # after year
            st.session_state.classified_paths_2024 = classified_paths
            st.session_state.classified_shape_2024 = patches_shape
        
        # Display sample classified images with originals
        if classified_results:
            st.subheader("Sample Classification Results")
            
            # Create a figure with subplots - 2 rows: originals and masks
            num_samples = len(classified_results)
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
            
            # First row: Original Sentinel-2 RGB images
            for idx, result in enumerate(classified_results):
                axes[0, idx].imshow(result['rgb_original'])
                axes[0, idx].set_title(f"Original {result['i']}_{result['j']}")
                axes[0, idx].axis('off')
            
            # Second row: Classification masks
            for idx, result in enumerate(classified_results):
                axes[1, idx].imshow(result['mask'], cmap='gray')
                axes[1, idx].set_title(f"Classified {result['i']}_{result['j']}")
                axes[1, idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        progress_placeholder.progress(75)
        status_placeholder.success(f"Step 3/4: Classified {total_patches} patches successfully")
        
        # 4. RECONSTRUCTION STEP
        status_placeholder.info("Step 4/4: Reconstructing full classification image...")
        
        # Get classified paths based on year
        if year == st.session_state.before_year:
            patches_info = st.session_state.patches_info
            classified_paths = st.session_state.classified_paths
            patches_shape = st.session_state.classified_shape
            clipped_meta = st.session_state.clipped_meta
        else: # after year
            patches_info = st.session_state.patches_info_2024
            classified_paths = st.session_state.classified_paths_2024
            patches_shape = st.session_state.classified_shape_2024
            clipped_meta = st.session_state.clipped_meta_2024
        
        # Load all classified patches
        patches = []
        patch_indices = []
        
        for path in classified_paths:
            # Extract i, j from filename
            filename = os.path.basename(path)
            parts = filename.split('_')
            i_j_part = parts[-2:]
            i = int(i_j_part[0])
            j = int(i_j_part[1].split('.')[0])
            
            # Read the patch
            with rasterio.open(path) as src:
                patch = src.read(1) # Read first band
                patches.append(patch)
                patch_indices.append((i, j))
        
        # Determine grid dimensions
        i_vals = [idx[0] for idx in patch_indices]
        j_vals = [idx[1] for idx in patch_indices]
        max_i = max(i_vals) + 1
        max_j = max(j_vals) + 1
        
        # Create empty grid to hold patches
        patch_size = patches_info['patch_size']
        grid = np.zeros((max_i, max_j, patch_size, patch_size), dtype=np.uint8)
        
        # Fill the grid with patches
        for (i, j), patch in zip(patch_indices, patches):
            grid[i, j] = patch
        
        # Reconstruct the full image using unpatchify
        reconstructed_image = unpatchify(grid, (max_i * patch_size, max_j * patch_size))
        
        # Create output filename
        base_dir = os.path.dirname(image_path)
        output_filename = f"reconstructed_classification_{year}_region{region_number}.tif"
        output_path = os.path.join(base_dir, output_filename)
        
        # Save as GeoTIFF with the appropriate UTM CRS
        # Use the metadata from the clipped image but update dimensions
        out_meta = clipped_meta.copy()
        out_meta.update({
            'count': 1,
            'height': reconstructed_image.shape[0],
            'width': reconstructed_image.shape[1],
            'dtype': 'uint8'
        })
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(reconstructed_image, 1)
        
        # Store the reconstructed image path in session state for change detection
        if year == st.session_state.before_year:
            st.session_state.reconstructed_before_path = output_path
            st.session_state.reconstructed_before_image = reconstructed_image
        else: # after year
            st.session_state.reconstructed_after_path = output_path
            st.session_state.reconstructed_after_image = reconstructed_image
        
        # Display the reconstructed image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(reconstructed_image, cmap='gray')
        ax.set_title(f"Reconstructed Classification Image ({year})")
        ax.axis('off')
        st.pyplot(fig)
        
        # Provide download link
        with open(output_path, "rb") as file:
            st.download_button(
                label=f"Download Reconstructed Classification ({year})",
                data=file,
                file_name=output_filename,
                mime="image/tiff"
            )
        
        progress_placeholder.progress(100)
        status_placeholder.success(f"All processing steps completed successfully for {year}!")
        
        return True
        
    except Exception as e:
        status_placeholder.error(f"Error in processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# First tab - Region Selection
with tab1:
    st.header("Select Region of Interest and Time Periods")
    
    # Add warning about region size
    st.warning("Note: For optimal results, select regions smaller than 40 sq km. Larger areas will be processed using tiling, which may take longer.")
    
    # Create a folium map centered at a default location
    m = folium.Map(location=[35.6892, 51.3890], zoom_start=10) # Default to Tehran
    
    # Add drawing tools to the map
    draw = folium.plugins.Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    m.add_child(draw)
    
    # Use st_folium to capture the drawn shapes
    map_data = st_folium(m, width=800, height=500)
    
    # Process the drawn shapes from map_data
    if map_data is not None and 'last_active_drawing' in map_data and map_data['last_active_drawing'] is not None:
        drawn_shape = map_data['last_active_drawing']
        if 'geometry' in drawn_shape:
            geometry = drawn_shape['geometry']
            
            if geometry['type'] == 'Polygon':
                # Extract coordinates from the GeoJSON
                coords = geometry['coordinates'][0] # First element contains the exterior ring
                polygon = Polygon(coords)
                
                # Store in session state
                st.session_state.last_drawn_polygon = polygon
                
                # Display the UTM zone for this polygon
                centroid = polygon.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                
                # Calculate approximate area in square kilometers
                area_sq_km = polygon.area * 111 * 111  # Approximate conversion from degrees to km²
                
                st.success(f"Shape captured in UTM Zone {utm_zone} ({utm_epsg})! Area: ~{area_sq_km:.2f} km². Click 'Save Selected Region' to save it.")
                
                # Warn if area is large
                if area_sq_km > 40:
                    st.warning(f"Selected area is large ({area_sq_km:.2f} km²). Processing will use tiling to handle the download size limit.")
    
    # Add a button to save the drawn polygons
    if st.button("Save Selected Region"):
        if 'last_drawn_polygon' in st.session_state:
            # Check if this polygon is already saved
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success(f"Region saved! Total regions: {len(st.session_state.drawn_polygons)}")
            else:
                st.info("This polygon is already saved.")
        else:
            st.warning("Please draw a polygon on the map first")
    
    # For demonstration purposes - keep the manual entry option
    with st.expander("Manually Enter Polygon Coordinates (For Testing)"):
        col1, col2 = st.columns(2)
        with col1:
            lat_input = st.text_input("Latitude coordinates (comma separated)", "35.68, 35.70, 35.69, 35.68")
        with col2:
            lon_input = st.text_input("Longitude coordinates (comma separated)", "51.38, 51.39, 51.40, 51.38")
        
        if st.button("Add Test Polygon"):
            try:
                lats = [float(x.strip()) for x in lat_input.split(",")]
                lons = [float(x.strip()) for x in lon_input.split(",")]
                if len(lats) == len(lons) and len(lats) >= 3:
                    coords = list(zip(lons, lats)) # GeoJSON format is [lon, lat]
                    test_polygon = Polygon(coords)
                    st.session_state.last_drawn_polygon = test_polygon
                    
                    # Display the UTM zone for this polygon
                    centroid = test_polygon.centroid
                    utm_zone = get_utm_zone(centroid.x)
                    utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                    
                    # Calculate approximate area
                    area_sq_km = test_polygon.area * 111 * 111  # Approximate conversion from degrees to km²
                    
                    st.success(f"Test polygon created in UTM Zone {utm_zone} ({utm_epsg})! Area: ~{area_sq_km:.2f} km². Click 'Save Selected Region' to save it.")
                    
                    # Warn if area is large
                    if area_sq_km > 40:
                        st.warning(f"Selected area is large ({area_sq_km:.2f} km²). Processing will use tiling to handle the download size limit.")
                else:
                    st.error("Please provide at least 3 coordinate pairs")
            except ValueError:
                st.error("Invalid coordinates. Please enter numeric values.")

    # Display saved regions with delete options
    if st.session_state.drawn_polygons:
        st.subheader("Saved Regions")
        
        # Display each region with a delete button
        for i, poly in enumerate(st.session_state.drawn_polygons):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                # Display region information
                st.write(f"Region {i+1}: {poly.wkt[:60]}...")
            
            with col2:
                # Display the UTM zone for this polygon
                centroid = poly.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                st.write(f"UTM Zone: {utm_zone} ({utm_epsg})")
            
            with col3:
                # Display area
                area_sq_km = poly.area * 111 * 111  # Approximate conversion from degrees to km²
                st.write(f"Area: ~{area_sq_km:.2f} km²")
                
            with col4:
                # Add a delete button for each region
                if st.button("Delete", key=f"delete_region_{i}"):
                    st.session_state.drawn_polygons.pop(i)
                    st.rerun()
    
    # Time period selection section
    st.subheader("Select Time Periods for Analysis", divider="blue")
    
    # Create a beautiful card-like container for time selection
    time_selection_container = st.container()
    with time_selection_container:
        st.markdown("""
        <style>
        .time-selection-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .time-selection-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #1E88E5;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state variables for time selection if they don't exist
        if 'before_year' not in st.session_state:
            st.session_state.before_year = "2021"
        if 'after_year' not in st.session_state:
            st.session_state.after_year = "2024"
        if 'start_month' not in st.session_state:
            st.session_state.start_month = 5  # May
        if 'end_month' not in st.session_state:
            st.session_state.end_month = 6  # June
        
        # Month selection (shared between before and after)
        st.markdown('<div class="time-selection-card">', unsafe_allow_html=True)
        st.markdown('<div class="time-selection-header">Select Months (for both images)</div>', unsafe_allow_html=True)
        
        months = ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"]
        
        # Create two columns for start and end month selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_month = st.selectbox(
                "Start Month",
                options=range(1, 13),
                format_func=lambda x: months[x-1],
                index=st.session_state.start_month - 1,
                key="start_month_select"
            )
            st.session_state.start_month = start_month
        
        with col2:
            # Ensure end month is not before start month
            valid_end_months = range(start_month, 13)
            end_month = st.selectbox(
                "End Month",
                options=valid_end_months,
                format_func=lambda x: months[x-1],
                index=min(st.session_state.end_month - start_month, len(valid_end_months) - 1),
                key="end_month_select"
            )
            st.session_state.end_month = end_month
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Year selection for "Before" and "After" images
        col1, col2 = st.columns(2)
        
        with col1:
           # st.markdown('<div class="time-selection-card">', unsafe_allow_html=True)
            st.markdown('<div class="time-selection-header">Before Image Year</div>', unsafe_allow_html=True)
            
            # Year dropdown for "Before" image
            before_year = st.selectbox(
                "Select Year",
                options=["2019", "2020", "2021", "2022", "2023", "2024"],
                index=2,  # Default to 2021
                key="before_year_select"
            )
            st.session_state.before_year = before_year
            
            # Display the selected date range
            before_start_date = f"{before_year}-{start_month:02d}-01"
            before_end_date = f"{before_year}-{end_month:02d}-{30 if end_month in [4, 6, 9, 11] else 31 if end_month != 2 else 29 if int(before_year) % 4 == 0 else 28}"
            st.info(f"Selected period: {before_start_date} to {before_end_date}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            #st.markdown('<div class="time-selection-card">', unsafe_allow_html=True)
            st.markdown('<div class="time-selection-header">After Image Year</div>', unsafe_allow_html=True)
            
            # Year dropdown for "After" image
            after_year = st.selectbox(
                "Select Year",
                options=["2019", "2020", "2021", "2022", "2023", "2024"],
                index=5,  # Default to 2024
                key="after_year_select"
            )
            st.session_state.after_year = after_year
            
            # Display the selected date range
            after_start_date = f"{after_year}-{start_month:02d}-01"
            after_end_date = f"{after_year}-{end_month:02d}-{30 if end_month in [4, 6, 9, 11] else 31 if end_month != 2 else 29 if int(after_year) % 4 == 0 else 28}"
            st.info(f"Selected period: {after_start_date} to {after_end_date}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Validate time periods
        if int(before_year) > int(after_year):
            st.warning("⚠️ The 'Before' year is later than the 'After' year. This may lead to unexpected results.")

with tab2:
    st.header("Before Image Analysis")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at: {model_path}. Please update the path to your model file.")
    
    # Automated processing section
    st.subheader("Automated Image Processing")
    
    if len(st.session_state.drawn_polygons) > 0:
        # Let user select which polygon to use for processing
        polygon_index = st.selectbox(
            "Select region to process",
            range(len(st.session_state.drawn_polygons)),
            format_func=lambda i: f"Region {i+1}",
            key="polygon_selector_before"
        )
        
        selected_polygon = st.session_state.drawn_polygons[polygon_index]
        region_number = polygon_index + 1 # For naming files
        
        # Calculate and display area
        area_sq_km = selected_polygon.area * 111 * 111  # Approximate conversion from degrees to km²
        st.info(f"Selected region area: ~{area_sq_km:.2f} km²")
        
        # Warn if area is large
        if area_sq_km > 40:
            st.warning(f"Selected area is large ({area_sq_km:.2f} km²). Processing will use tiling to handle the download size limit, which may take longer.")
        
        # Display selected time period
        year = st.session_state.before_year
        start_month = st.session_state.start_month
        end_month = st.session_state.end_month
        months = ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"]
        
        st.info(f"Selected time period: {months[start_month-1]} to {months[end_month-1]} {year}")
        
        # Cloud cover slider
        cloud_cover = st.slider("Maximum Cloud Cover (%)", 0, 100, 15, key="cloud_cover_before")
        
        # Process button
        if st.button(f"Download and Process {year} Image ({months[start_month-1]}-{months[end_month-1]} Median)", key="process_before"):
            st.info(f"Starting download of Sentinel-2 median composite for {months[start_month-1]}-{months[end_month-1]} {year}...")
            
            # Download Sentinel-2 image using GEES2Downloader
            sentinel_before_path = download_sentinel2_with_gees2(year, selected_polygon, start_month, end_month, cloud_cover)
            
            if sentinel_before_path:
                # Call the processing function
                success = process_image(
                    image_path=sentinel_before_path,
                    year=year,
                    selected_polygon=selected_polygon,
                    region_number=region_number
                )
                
                if success:
                    st.success(f"✅ {year} image processing complete! You can now proceed to the Change Detection tab.")
                else:
                    st.error(f"❌ There was an error processing the {year} image. Please check the error messages above.")
            else:
                st.error("There is no proper image for download. Try increasing the cloud cover percentage or selecting a different region or time period.")
        
        # Show current status
        if 'reconstructed_before_image' in st.session_state and st.session_state.reconstructed_before_image is not None:
            st.success("✅ 'Before' image has already been processed successfully.")
            
            # Show the reconstructed image
            st.subheader("Current 'Before' Classification Result")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(st.session_state.reconstructed_before_image, cmap='gray')
            ax.set_title(f"{year} Building Classification")
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.info("No regions have been selected yet. Please go to the Region Selection tab and draw a polygon.")

# Third tab - Sentinel-2 "After" Image Analysis
with tab3:
    st.header("After Image Analysis")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at: {model_path}. Please update the path to your model file.")
    
    # Automated processing section
    st.subheader("Automated Image Processing")
    
    if len(st.session_state.drawn_polygons) > 0:
        # Let user select which polygon to use for processing
        polygon_index = st.selectbox(
            "Select region to process",
            range(len(st.session_state.drawn_polygons)),
            format_func=lambda i: f"Region {i+1}",
            key="polygon_selector_after"
        )
        
        selected_polygon = st.session_state.drawn_polygons[polygon_index]
        region_number = polygon_index + 1 # For naming files
        
        # Calculate and display area
        area_sq_km = selected_polygon.area * 111 * 111  # Approximate conversion from degrees to km²
        st.info(f"Selected region area: ~{area_sq_km:.2f} km²")
        
        # Warn if area is large
        if area_sq_km > 40:
            st.warning(f"Selected area is large ({area_sq_km:.2f} km²). Processing will use tiling to handle the download size limit, which may take longer.")
        
        # Display selected time period
        year = st.session_state.after_year
        start_month = st.session_state.start_month
        end_month = st.session_state.end_month
        months = ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"]
        
        st.info(f"Selected time period: {months[start_month-1]} to {months[end_month-1]} {year}")
        
        # Cloud cover slider
        cloud_cover = st.slider("Maximum Cloud Cover (%)", 0, 100, 15, key="cloud_cover_after")
        
        # Process button
        if st.button(f"Download and Process {year} Image ({months[start_month-1]}-{months[end_month-1]} Median)", key="process_after"):
            st.info(f"Starting download of Sentinel-2 median composite for {months[start_month-1]}-{months[end_month-1]} {year}...")
            
            # Download Sentinel-2 image using GEES2Downloader
            sentinel_after_path = download_sentinel2_with_gees2(year, selected_polygon, start_month, end_month, cloud_cover)
            
            if sentinel_after_path:
                # Call the processing function
                success = process_image(
                    image_path=sentinel_after_path,
                    year=year,
                    selected_polygon=selected_polygon,
                    region_number=region_number
                )
                
                if success:
                    st.success(f"✅ {year} image processing complete! You can now proceed to the Change Detection tab.")
                else:
                    st.error(f"❌ There was an error processing the {year} image. Please check the error messages above.")
            else:
                st.error("There is no proper image for download. Try increasing the cloud cover percentage or selecting a different region or time period.")
        
        # Show current status
        if 'reconstructed_after_image' in st.session_state and st.session_state.reconstructed_after_image is not None:
            st.success("✅ 'After' image has already been processed successfully.")
            
            # Show the reconstructed image
            st.subheader("Current 'After' Classification Result")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(st.session_state.reconstructed_after_image, cmap='gray')
            ax.set_title(f"{year} Building Classification")
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.info("No regions have been selected yet. Please go to the Region Selection tab and draw a polygon.")

with tab4: # This line is commented out as the code below is the content of tab4
    st.header("Building Change Detection")
    
    # Import required libraries (ensure these are available in your environment)
    import tempfile
    import os
    import time
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask  # Not explicitly used in this snippet, but kept from original
    from shapely.geometry import mapping, box  # Not explicitly used in this snippet, but kept from original
    import io
    from PIL import Image
    import folium
    from folium import plugins
    # from streamlit_folium import st_folium # No longer needed for display
    import streamlit.components.v1 as components  # For st.components.v1.html
    import json  # Not explicitly used in this snippet, but kept from original
    import geopandas as gpd
    import base64
    import numpy as np  # Ensure numpy is imported
    import matplotlib.pyplot as plt  # Ensure matplotlib is imported
    
    # Example definition for apply_erosion if not already defined in your main script:
    # You should have this function defined or imported appropriately.
    # For demonstration, a simple placeholder:
    def apply_erosion(mask, kernel_size_val):
        # Replace with your actual cv2.erode or skimage.morphology.erosion implementation
        # This is just a placeholder to make the code runnable
        st.warning(f"apply_erosion function called with kernel_size {kernel_size_val}. Implement actual erosion.")
        return mask  # Placeholder returns original mask
    
    
    # 1) Retrieve the processed classification arrays
    before_year = st.session_state.get("before_year", "2021")
    after_year = st.session_state.get("after_year", "2024")
    
    # Ensure both exist
    if (
        "reconstructed_before_image" not in st.session_state or
        "reconstructed_after_image" not in st.session_state
    ):
        st.warning("Please process both the Before and After images first (tabs 2 & 3).")
        st.stop()
    
    img_before = st.session_state.reconstructed_before_image
    img_after = st.session_state.reconstructed_after_image
    
    # 2) Dimension check
    if img_before.shape != img_after.shape:
        st.error("The Before/After images have different shapes.")
        st.info(f"{before_year}: {img_before.shape}, {after_year}: {img_after.shape}")
        st.stop()
    
    # 3) Compute raw change mask (new buildings)
    binary_before = (img_before > 0).astype(np.uint8)
    binary_after = (img_after > 0).astype(np.uint8)
    raw_mask = ((binary_after == 1) & (binary_before == 0)).astype(np.uint8) * 255
    st.session_state.change_detection_result = raw_mask
    
    # 4) Display raw results side by side
    st.subheader("Raw Change Detection")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(binary_before, cmap="gray")
    axs[0].set_title(f"{before_year} Classification")
    axs[0].axis("off")
    axs[1].imshow(binary_after, cmap="gray")
    axs[1].set_title(f"{after_year} Classification")
    axs[1].axis("off")
    axs[2].imshow(raw_mask, cmap="hot")
    axs[2].set_title("New Buildings")
    axs[2].axis("off")
    st.pyplot(fig)
    
    # 5) Erosion UI
    st.subheader("Refine with Morphological Erosion")
    
    kernel = st.selectbox(
        "Kernel size",
        [2, 3, 4, 5, 7, 9],
        index=0,
        key="tab4_erosion_kernel_size"
    )
    
    if st.button("Apply Erosion", key="tab4_apply_erosion_btn"):
        eroded = apply_erosion(raw_mask, kernel)  # Ensure apply_erosion is defined
        st.session_state.eroded_result = eroded
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.imshow(raw_mask, cmap="hot")
        ax1.set_title("Original Mask")
        ax1.axis("off")
        ax2.imshow(eroded, cmap="hot")
        ax2.set_title(f"Eroded (k={kernel})")
        ax2.axis("off")
        st.pyplot(fig)
    
    # 6) Interactive Map only once we have eroded_result
    if "eroded_result" in st.session_state:
        st.subheader("Interactive Map with Transparency Controls")
        st.info("Use the layer control panel on the right to toggle layers on/off and adjust transparency with the sliders. Click the fullscreen button to view the map in fullscreen mode.")
    
        try:
            if ('region_number' in st.session_state and
                'drawn_polygons' in st.session_state and
                st.session_state.region_number <= len(st.session_state.drawn_polygons)):
                selected_polygon = st.session_state.drawn_polygons[st.session_state.region_number - 1]
                centroid = selected_polygon.centroid
                center = [centroid.y, centroid.x]
            else:
                center = [35.6892, 51.3890]
                selected_polygon = None
    
            temp_dir = tempfile.gettempdir()
            has_sentinel_data = (
                'clipped_img' in st.session_state and
                'clipped_img_2024' in st.session_state and
                'clipped_meta' in st.session_state
            )
    
            if 'clipped_meta' in st.session_state:
                utm_transform = st.session_state.clipped_meta['transform']
                utm_crs = st.session_state.clipped_meta['crs']
                utm_height = st.session_state.clipped_img.shape[1]
                utm_width = st.session_state.clipped_img.shape[2]
                if selected_polygon:
                    bounds = selected_polygon.bounds
                else:
                    bounds = None
            else:
                if selected_polygon:
                    bounds = selected_polygon.bounds
                else:
                    bounds = None
                utm_crs = None
                utm_transform = None
                utm_height = binary_before.shape[0]
                utm_width = binary_before.shape[1]
    
            before_class_wgs84_path = None
            after_class_wgs84_path = None
            change_mask_wgs84_path = None
            before_rgb_wgs84_path = None
            after_rgb_wgs84_path = None
    
            target_transform = None
            target_width = None
            target_height = None
            target_bounds = None
    
            if utm_crs is not None and utm_transform is not None:
                dst_crs = 'EPSG:4326'
    
                before_class_utm_path = os.path.join(temp_dir, f"before_class_utm_{before_year}_{time.time()}.tif")
                with rasterio.open(
                    before_class_utm_path, 'w', driver='GTiff',
                    height=binary_before.shape[0], width=binary_before.shape[1],
                    count=1, dtype=binary_before.dtype, crs=utm_crs, transform=utm_transform
                ) as dst:
                    dst.write(binary_before, 1)
    
                before_class_wgs84_path = os.path.join(temp_dir, f"before_class_wgs84_{before_year}_{time.time()}.tif")
                with rasterio.open(before_class_utm_path) as src:
                    dst_transform_calc, dst_width_calc, dst_height_calc = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds)
                    target_transform = dst_transform_calc
                    target_width = dst_width_calc
                    target_height = dst_height_calc
                    dst_kwargs = src.meta.copy()
                    dst_kwargs.update({
                        'crs': dst_crs, 'transform': target_transform,
                        'width': target_width, 'height': target_height
                    })
                    with rasterio.open(before_class_wgs84_path, 'w', **dst_kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=target_transform, dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
                        target_bounds = dst.bounds
    
                after_class_utm_path = os.path.join(temp_dir, f"after_class_utm_{after_year}_{time.time()}.tif")
                with rasterio.open(
                    after_class_utm_path, 'w', driver='GTiff',
                    height=binary_after.shape[0], width=binary_after.shape[1],
                    count=1, dtype=binary_after.dtype, crs=utm_crs, transform=utm_transform
                ) as dst:
                    dst.write(binary_after, 1)
    
                after_class_wgs84_path = os.path.join(temp_dir, f"after_class_wgs84_{after_year}_{time.time()}.tif")
                with rasterio.open(after_class_utm_path) as src:
                    dst_kwargs = src.meta.copy()
                    dst_kwargs.update({
                        'crs': dst_crs, 'transform': target_transform,
                        'width': target_width, 'height': target_height
                    })
                    with rasterio.open(after_class_wgs84_path, 'w', **dst_kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=target_transform, dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
    
                if "eroded_result" in st.session_state:
                    change_mask_utm_path = os.path.join(temp_dir, f"change_mask_utm_{time.time()}.tif")
                    with rasterio.open(
                        change_mask_utm_path, 'w', driver='GTiff',
                        height=st.session_state.eroded_result.shape[0], width=st.session_state.eroded_result.shape[1],
                        count=1, dtype=st.session_state.eroded_result.dtype, crs=utm_crs, transform=utm_transform
                    ) as dst:
                        dst.write(st.session_state.eroded_result, 1)
    
                    change_mask_wgs84_path = os.path.join(temp_dir, f"change_mask_wgs84_{time.time()}.tif")
                    with rasterio.open(change_mask_utm_path) as src:
                        dst_kwargs = src.meta.copy()
                        dst_kwargs.update({
                            'crs': dst_crs, 'transform': target_transform,
                            'width': target_width, 'height': target_height
                        })
                        with rasterio.open(change_mask_wgs84_path, 'w', **dst_kwargs) as dst:
                            reproject(
                                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                                src_transform=src.transform, src_crs=src.crs,
                                dst_transform=target_transform, dst_crs=dst_crs,
                                resampling=Resampling.nearest
                            )
    
                if has_sentinel_data and target_bounds is not None:
                    before_sentinel_utm_path = os.path.join(temp_dir, f"before_sentinel_utm_{before_year}_{time.time()}.tif")
                    before_bands = st.session_state.clipped_img[:4, :, :]
                    with rasterio.open(
                        before_sentinel_utm_path, 'w', driver='GTiff',
                        height=before_bands.shape[1], width=before_bands.shape[2],
                        count=4, dtype=before_bands.dtype, crs=utm_crs, transform=utm_transform
                    ) as dst:
                        for i in range(4):
                            dst.write(before_bands[i], i+1)
    
                    before_sentinel_wgs84_path = os.path.join(temp_dir, f"before_sentinel_wgs84_{before_year}_{time.time()}.tif")
                    with rasterio.open(before_sentinel_utm_path) as src:
                        dst_kwargs = src.meta.copy()
                        dst_kwargs.update({
                            'crs': 'EPSG:4326', 'transform': target_transform,
                            'width': target_width, 'height': target_height
                        })
                        with rasterio.open(before_sentinel_wgs84_path, 'w', **dst_kwargs) as dst:
                            for i in range(1, 5):
                                reproject(
                                    source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                                    src_transform=src.transform, src_crs=src.crs,
                                    dst_transform=target_transform, dst_crs='EPSG:4326',
                                    resampling=Resampling.bilinear
                                )
    
                    after_sentinel_utm_path = os.path.join(temp_dir, f"after_sentinel_utm_{after_year}_{time.time()}.tif")
                    after_bands = st.session_state.clipped_img_2024[:4, :, :]
                    with rasterio.open(
                        after_sentinel_utm_path, 'w', driver='GTiff',
                        height=after_bands.shape[1], width=after_bands.shape[2],
                        count=4, dtype=after_bands.dtype, crs=utm_crs, transform=utm_transform
                    ) as dst:
                        for i in range(4):
                            dst.write(after_bands[i], i+1)
    
                    after_sentinel_wgs84_path = os.path.join(temp_dir, f"after_sentinel_wgs84_{after_year}_{time.time()}.tif")
                    with rasterio.open(after_sentinel_utm_path) as src:
                        dst_kwargs = src.meta.copy()
                        dst_kwargs.update({
                            'crs': 'EPSG:4326', 'transform': target_transform,
                            'width': target_width, 'height': target_height
                        })
                        with rasterio.open(after_sentinel_wgs84_path, 'w', **dst_kwargs) as dst:
                            for i in range(1, 5):
                                reproject(
                                    source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                                    src_transform=src.transform, src_crs=src.crs,
                                    dst_transform=target_transform, dst_crs='EPSG:4326',
                                    resampling=Resampling.bilinear
                                )
    
                    before_rgb_wgs84_path = os.path.join(temp_dir, f"before_rgb_wgs84_{before_year}_{time.time()}.tif")
                    with rasterio.open(before_sentinel_wgs84_path) as src:
                        profile = src.profile.copy()
                        profile.update(count=3, dtype='uint8')
                        with rasterio.open(before_rgb_wgs84_path, 'w', **profile) as dst:
                            rgb_data = np.zeros((3, src.height, src.width), dtype=np.uint8)
                            for i, band_idx in enumerate([3, 2, 1]):
                                band_data = src.read(band_idx)
                                min_val = np.percentile(band_data[band_data > 0], 2) if np.any(band_data > 0) else 0
                                max_val = np.percentile(band_data[band_data > 0], 98) if np.any(band_data > 0) else 1
                                if max_val > min_val:
                                    rgb_data[i] = np.clip((band_data - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                                else:
                                    rgb_data[i] = np.zeros_like(band_data, dtype=np.uint8)
                            dst.write(rgb_data)
    
                    after_rgb_wgs84_path = os.path.join(temp_dir, f"after_rgb_wgs84_{after_year}_{time.time()}.tif")
                    with rasterio.open(after_sentinel_wgs84_path) as src:
                        profile = src.profile.copy()
                        profile.update(count=3, dtype='uint8')
                        with rasterio.open(after_rgb_wgs84_path, 'w', **profile) as dst:
                            rgb_data = np.zeros((3, src.height, src.width), dtype=np.uint8)
                            for i, band_idx in enumerate([3, 2, 1]):
                                band_data = src.read(band_idx)
                                min_val = np.percentile(band_data[band_data > 0], 2) if np.any(band_data > 0) else 0
                                max_val = np.percentile(band_data[band_data > 0], 98) if np.any(band_data > 0) else 1
                                if max_val > min_val:
                                    rgb_data[i] = np.clip((band_data - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                                else:
                                    rgb_data[i] = np.zeros_like(band_data, dtype=np.uint8)
                            dst.write(rgb_data)
    
                st.subheader("Download Reprojected Data")
                st.write("The following files have been reprojected from UTM to WGS84 coordinate system:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if before_class_wgs84_path:
                        with open(before_class_wgs84_path, "rb") as file:
                            st.download_button(
                                label=f"Download {before_year} Classification", data=file,
                                file_name=f"before_classification_{before_year}_wgs84.tif", mime="image/tiff",
                                key=f"download_before_class_{before_year}"
                            )
                with col2:
                    if after_class_wgs84_path:
                        with open(after_class_wgs84_path, "rb") as file:
                            st.download_button(
                                label=f"Download {after_year} Classification", data=file,
                                file_name=f"after_classification_{after_year}_wgs84.tif", mime="image/tiff",
                                key=f"download_after_class_{after_year}"
                            )
                with col3:
                    if change_mask_wgs84_path:
                        with open(change_mask_wgs84_path, "rb") as file:
                            st.download_button(
                                label="Download Change Mask", data=file,
                                file_name=f"change_mask_{before_year}_{after_year}_wgs84.tif", mime="image/tiff",
                                key="download_change_mask"
                            )
            else:
                st.warning("No UTM coordinate information found. Files are not properly georeferenced.")
                st.subheader("Download Data")
                st.write("Note: These files are not georeferenced as UTM coordinate information was not available.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    img = Image.fromarray((binary_before * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label=f"Download {before_year} Classification", data=buf.getvalue(),
                        file_name=f"before_classification_{before_year}.png", mime="image/png",
                        key=f"download_before_class_png_{before_year}"
                    )
                with col2:
                    img = Image.fromarray((binary_after * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label=f"Download {after_year} Classification", data=buf.getvalue(),
                        file_name=f"after_classification_{after_year}.png", mime="image/png",
                        key=f"download_after_class_png_{after_year}"
                    )
                with col3:
                    img = Image.fromarray(st.session_state.eroded_result)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="Download Change Mask", data=buf.getvalue(),
                        file_name=f"change_mask_{before_year}_{after_year}.png", mime="image/png",
                        key="download_change_mask_png"
                    )
    
            def raster_to_folium_overlay(raster_path, colormap='viridis', opacity=0.7, is_binary=False):
                with rasterio.open(raster_path) as src:
                    data = src.read(1)
                    bounds = src.bounds
                    bounds_latlon = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
                    if is_binary:
                        rgba_array = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
                        mask_val = data == 1
                        if colormap == 'Greens':
                            rgba_array[mask_val, 0:3] = [0, 255, 0]  # R, G, B
                        elif colormap == 'Reds':
                            rgba_array[mask_val, 0:3] = [255, 0, 0]  # R, G, B
                        rgba_array[mask_val, 3] = 180  # Alpha for non-transparent parts
                        pil_img = Image.fromarray(rgba_array, 'RGBA')
                    elif colormap == 'hot' and data.max() > 1:  # Assuming change mask 0-255
                        import matplotlib.cm as cm  # Moved import here
                        data_norm = data / 255.0
                        cmap_hot = cm.get_cmap('hot')  # Use get_cmap
                        rgba_array = cmap_hot(data_norm)
                        rgba_array[data == 0, 3] = 0    # Fully transparent for 0
                        rgba_array[data > 0, 3] = 0.8  # Semi-transparent for changes
                        rgba_array = (rgba_array * 255).astype(np.uint8)
                        pil_img = Image.fromarray(rgba_array, 'RGBA')
                    else:  # For RGB Sentinel or other single band with general colormap
                        if src.count == 3:  # RGB
                            rgb_data_src = src.read([1, 2, 3])
                            # Ensure data is in range 0-255 if not already
                            # This part might need adjustment based on input RGB GeoTIFF bit depth
                            if rgb_data_src.dtype != np.uint8:
                                # Simple scaling if not uint8, assumes it's scaled appropriately before saving
                                rgb_data_src = np.clip(rgb_data_src, 0, 255).astype(np.uint8) if np.issubdtype(rgb_data_src.dtype, np.integer) else (rgb_data_src / rgb_data_src.max() * 255).astype(np.uint8)
    
                            img_array_rgb = np.transpose(rgb_data_src, (1, 2, 0))
                            pil_img = Image.fromarray(img_array_rgb)
                        else:  # Single band, general colormap (e.g., viridis if not specified)
                            import matplotlib.cm as cm  # Moved import here
                            data_min, data_max = np.nanmin(data), np.nanmax(data)
                            if data_max > data_min:
                                data_norm = (data - data_min) / (data_max - data_min)
                            else:
                                data_norm = np.zeros_like(data)
                            cmap_viridis = cm.get_cmap(colormap)  # Use get_cmap
                            img_array_cmap = cmap_viridis(data_norm)
                            img_array_cmap = (img_array_cmap[:, :, :3] * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_array_cmap)
    
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format='PNG')
                    img_str = base64.b64encode(img_buffer.getvalue()).decode()
                    return f"data:image/png;base64,{img_str}", bounds_latlon
    
            # Create the folium map
            m = folium.Map(location=center, zoom_start=15, tiles=None)
            plugins.Fullscreen(
                position='topleft', title='Expand to fullscreen',
                title_cancel='Exit fullscreen', force_separate_button=True
            ).add_to(m)
    
            # Add base layers
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='Google Satellite', overlay=False, control=True).add_to(m)
            folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google Maps', name='Google Maps', overlay=False, control=True).add_to(m)
            folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', overlay=False, control=True, show=True).add_to(m)  # Default OSM shown
    
            # Store layer information for transparency controls
            layer_info = []
    
            if utm_crs is not None and utm_transform is not None:
                if selected_polygon and 'region_number' in st.session_state:
                    gdf = gpd.GeoDataFrame(geometry=[selected_polygon], crs="EPSG:4326")
                    folium.GeoJson(
                        gdf.to_json(), name="Selected Region",
                        style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 2, 'fillOpacity': 0}
                    ).add_to(m)
    
                if has_sentinel_data and before_rgb_wgs84_path and after_rgb_wgs84_path:
                    try:
                        img_data_before_rgb, bounds_before_rgb = raster_to_folium_overlay(before_rgb_wgs84_path, opacity=0.8)
                        before_sentinel_layer = folium.raster_layers.ImageOverlay(
                            image=img_data_before_rgb, bounds=bounds_before_rgb, 
                            opacity=0.8, name=f"Before Sentinel-2 ({before_year})"
                        )
                        before_sentinel_layer.add_to(m)
                        layer_info.append({
                            'name': f"Before Sentinel-2 ({before_year})",
                            'id': f'before_sentinel_{before_year}',
                            'default_opacity': 0.8
                        })
                        
                        img_data_after_rgb, bounds_after_rgb = raster_to_folium_overlay(after_rgb_wgs84_path, opacity=0.8)
                        after_sentinel_layer = folium.raster_layers.ImageOverlay(
                            image=img_data_after_rgb, bounds=bounds_after_rgb, 
                            opacity=0.8, name=f"After Sentinel-2 ({after_year})"
                        )
                        after_sentinel_layer.add_to(m)
                        layer_info.append({
                            'name': f"After Sentinel-2 ({after_year})",
                            'id': f'after_sentinel_{after_year}',
                            'default_opacity': 0.8
                        })
                    except Exception as e:
                        st.warning(f"Could not add Sentinel-2 RGB layers: {str(e)}")
    
                if before_class_wgs84_path:
                    try:
                        img_data_before_class, bounds_before_class = raster_to_folium_overlay(before_class_wgs84_path, colormap='Greens', opacity=0.7, is_binary=True)
                        before_class_layer = folium.raster_layers.ImageOverlay(
                            image=img_data_before_class, bounds=bounds_before_class, 
                            opacity=0.7, name=f"Before Classification ({before_year})"
                        )
                        before_class_layer.add_to(m)
                        layer_info.append({
                            'name': f"Before Classification ({before_year})",
                            'id': f'before_class_{before_year}',
                            'default_opacity': 0.7
                        })
                    except Exception as e:
                        st.warning(f"Could not add before classification layer: {str(e)}")
    
                if after_class_wgs84_path:
                    try:
                        img_data_after_class, bounds_after_class = raster_to_folium_overlay(after_class_wgs84_path, colormap='Reds', opacity=0.7, is_binary=True)
                        after_class_layer = folium.raster_layers.ImageOverlay(
                            image=img_data_after_class, bounds=bounds_after_class, 
                            opacity=0.7, name=f"After Classification ({after_year})"
                        )
                        after_class_layer.add_to(m)
                        layer_info.append({
                            'name': f"After Classification ({after_year})",
                            'id': f'after_class_{after_year}',
                            'default_opacity': 0.7
                        })
                    except Exception as e:
                        st.warning(f"Could not add after classification layer: {str(e)}")
    
                if change_mask_wgs84_path:
                    try:
                        img_data_change, bounds_change = raster_to_folium_overlay(change_mask_wgs84_path, colormap='hot', opacity=0.7)
                        change_mask_layer = folium.raster_layers.ImageOverlay(
                            image=img_data_change, bounds=bounds_change, 
                            opacity=0.7, name=f"Change Detection Mask ({before_year}-{after_year})"
                        )
                        change_mask_layer.add_to(m)
                        layer_info.append({
                            'name': f"Change Detection Mask ({before_year}-{after_year})",
                            'id': f'change_mask_{before_year}_{after_year}',
                            'default_opacity': 0.7
                        })
                    except Exception as e:
                        st.warning(f"Could not add change detection mask layer: {str(e)}")
    
                if target_bounds:
                    m.fit_bounds([[target_bounds.bottom, target_bounds.left], [target_bounds.top, target_bounds.right]])
            else:
                st.warning("Cannot display non-georeferenced data in the interactive map.")
    
            # Add the default layer control
            folium.LayerControl().add_to(m)
    
            # Get the map HTML
            map_html = m.get_root().render()
            
            # Create custom transparency control panel HTML and JavaScript
            transparency_control_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    .transparency-panel {{
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                        font-family: Arial, sans-serif;
                        font-size: 12px;
                        max-width: 280px;
                        z-index: 1000;
                        max-height: 400px;
                        overflow-y: auto;
                    }}
                    
                    .transparency-panel h4 {{
                        margin: 0 0 15px 0;
                        color: #333;
                        font-size: 14px;
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 8px;
                    }}
                    
                    .layer-control {{
                        margin-bottom: 15px;
                        padding: 8px;
                        border: 1px solid #e0e0e0;
                        border-radius: 4px;
                        background: #f9f9f9;
                    }}
                    
                    .layer-header {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 8px;
                    }}
                    
                    .layer-checkbox {{
                        margin-right: 8px;
                    }}
                    
                    .layer-name {{
                        font-weight: bold;
                        color: #333;
                        font-size: 11px;
                        flex: 1;
                    }}
                    
                    .transparency-slider {{
                        width: 100%;
                        margin-top: 5px;
                    }}
                    
                    .opacity-value {{
                        font-size: 10px;
                        color: #666;
                        text-align: center;
                        margin-top: 2px;
                    }}
                    
                    .slider-container {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }}
                    
                    .slider-label {{
                        font-size: 10px;
                        color: #666;
                        min-width: 60px;
                    }}
                </style>
            </head>
            <body>
                {map_html}
                
                <div class="transparency-panel" id="transparencyPanel">
                    <h4>🎛️ Layer Controls</h4>
                    <div id="layerControls">
                        <!-- Layer controls will be populated by JavaScript -->
                    </div>
                </div>
                
                <script>
                    // Layer information
                    const layerInfo = {json.dumps(layer_info)};
                    
                    // Wait for map to be fully loaded
                    setTimeout(function() {{
                        initializeTransparencyControls();
                    }}, 1000);
                    
                    function initializeTransparencyControls() {{
                        const controlsContainer = document.getElementById('layerControls');
                        
                        layerInfo.forEach(function(layer, index) {{
                            const layerDiv = document.createElement('div');
                            layerDiv.className = 'layer-control';
                            layerDiv.innerHTML = `
                                <div class="layer-header">
                                    <input type="checkbox" class="layer-checkbox" id="checkbox_${{layer.id}}" checked onchange="toggleLayer('${{layer.id}}', this.checked)">
                                    <label class="layer-name" for="checkbox_${{layer.id}}">${{layer.name}}</label>
                                </div>
                                <div class="slider-container">
                                    <span class="slider-label">Transparency:</span>
                                    <input type="range" class="transparency-slider" id="slider_${{layer.id}}" 
                                           min="0" max="100" value="${{Math.round(layer.default_opacity * 100)}}"
                                           oninput="updateOpacity('${{layer.id}}', this.value)">
                                </div>
                                <div class="opacity-value" id="value_${{layer.id}}">${{Math.round(layer.default_opacity * 100)}}%</div>
                            `;
                            controlsContainer.appendChild(layerDiv);
                        }});
                    }}
                    
                    function toggleLayer(layerId, isVisible) {{
                        // Find the layer in the map
                        if (window[window.map_var] && window[window.map_var].eachLayer) {{
                            window[window.map_var].eachLayer(function(layer) {{
                                if (layer.options && layer.options.name) {{
                                    const layerName = layer.options.name;
                                    const matchingLayerInfo = layerInfo.find(info => info.id === layerId);
                                    if (matchingLayerInfo && layerName === matchingLayerInfo.name) {{
                                        if (isVisible) {{
                                            if (!window[window.map_var].hasLayer(layer)) {{
                                                window[window.map_var].addLayer(layer);
                                            }}
                                        }} else {{
                                            if (window[window.map_var].hasLayer(layer)) {{
                                                window[window.map_var].removeLayer(layer);
                                            }}
                                        }}
                                    }}
                                }}
                            }});
                        }}
                    }}
                    
                    function updateOpacity(layerId, opacityValue) {{
                        const opacity = opacityValue / 100;
                        document.getElementById('value_' + layerId).textContent = opacityValue + '%';
                        
                        // Find and update the layer opacity
                        if (window[window.map_var] && window[window.map_var].eachLayer) {{
                            window[window.map_var].eachLayer(function(layer) {{
                                if (layer.options && layer.options.name) {{
                                    const layerName = layer.options.name;
                                    const matchingLayerInfo = layerInfo.find(info => info.id === layerId);
                                    if (matchingLayerInfo && layerName === matchingLayerInfo.name) {{
                                        if (layer.setOpacity) {{
                                            layer.setOpacity(opacity);
                                        }}
                                    }}
                                }}
                            }});
                        }}
                    }}
                    
                    // Find the map variable name
                    window.map_var = null;
                    for (let prop in window) {{
                        if (window[prop] && typeof window[prop] === 'object' && window[prop]._container) {{
                            window.map_var = prop;
                            break;
                        }}
                    }}
                </script>
            </body>
            </html>
            """
    
            # Display the enhanced map with transparency controls
            components.html(transparency_control_html, height=600)
    
            st.info("""
            **Enhanced Interactive Map Usage:**
            - 🎛️ **Layer Controls Panel**: Use the panel on the right to control each layer individually
            - ✅ **Toggle Layers**: Check/uncheck boxes to show/hide layers
            - 🎚️ **Transparency Sliders**: Adjust the transparency of each layer from 0% to 100%
            - 🔍 **Fullscreen**: Click the fullscreen button (top-left) for better viewing
            - 🗺️ **Base Maps**: Switch between Google Satellite, Google Maps, and OpenStreetMap
            - 🎨 **Color Coding**: 
              - Before classification in **green**
              - After classification in **red** 
              - Change detection mask in **hot colors** (red/yellow)
            - 📍 All layers are perfectly aligned and georeferenced
            """)
    
        except Exception as e:
            st.error(f"Error creating interactive map: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("After applying erosion, the interactive map will appear here.")
