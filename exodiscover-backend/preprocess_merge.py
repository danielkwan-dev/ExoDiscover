import pandas as pd
import numpy as np

def load_nasa_csv(filepath):
    """Load NASA CSV files that have comment headers"""
    try:
        # Try reading with comment character for NASA files
        return pd.read_csv(filepath, comment='#')
    except:
        try:
            # If that fails, try reading without header and find data start
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find where data starts (first line that doesn't start with # and has multiple columns)
            data_start = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith('#') and len(line.split(',')) > 1:
                    data_start = i
                    break
            
            # Read with skiprows
            return pd.read_csv(filepath, skiprows=data_start, comment='#')
        except:
            # Last resort: read manually
            print(f"Warning: Could not read {filepath} with standard methods")
            return pd.DataFrame()

def merge_exoplanet_datasets():
    """Simple script to merge Kepler, K2, and TESS datasets including planet names"""
    
    print("🪐 Loading and merging exoplanet datasets...")
    
    # Load datasets with proper NASA formatting
    kepler_df = load_nasa_csv('nasa_data/kepler.csv')
    k2_df = load_nasa_csv('nasa_data/k2.csv') 
    tess_df = load_nasa_csv('nasa_data/tess.csv')
    
    print(f"Loaded - Kepler: {len(kepler_df)} rows, {len(kepler_df.columns)} cols")
    print(f"Loaded - K2: {len(k2_df)} rows, {len(k2_df.columns)} cols") 
    print(f"Loaded - TESS: {len(tess_df)} rows, {len(tess_df.columns)} cols")
    
    # Debug: Show available columns
    print("\n🔍 Kepler columns:", list(kepler_df.columns) if len(kepler_df.columns) > 0 else "No columns found")
    print("🔍 K2 columns:", list(k2_df.columns) if len(k2_df.columns) > 0 else "No columns found")
    print("🔍 TESS columns:", list(tess_df.columns) if len(tess_df.columns) > 0 else "No columns found")
    
    # Initialize empty DataFrames
    kepler_clean = pd.DataFrame()
    k2_clean = pd.DataFrame()
    tess_clean = pd.DataFrame()
    
    # Process Kepler data - only if we have data
    if len(kepler_df) > 0:
        # Planet names for Kepler
        if 'kepler_name' in kepler_df.columns:
            kepler_clean['planet_name'] = kepler_df['kepler_name']
        elif 'kepoi_name' in kepler_df.columns:
            kepler_clean['planet_name'] = kepler_df['kepoi_name']
        elif 'kepid' in kepler_df.columns:
            kepler_clean['planet_name'] = 'KIC ' + kepler_df['kepid'].astype(str)
        
        # Scientific features
        if 'koi_period' in kepler_df.columns:
            kepler_clean['orbital_period'] = kepler_df['koi_period']
        if 'koi_prad' in kepler_df.columns:
            kepler_clean['planet_radius'] = kepler_df['koi_prad']
        if 'koi_depth' in kepler_df.columns:
            kepler_clean['transit_depth'] = kepler_df['koi_depth']
        if 'koi_duration' in kepler_df.columns:
            kepler_clean['transit_duration'] = kepler_df['koi_duration']
        if 'koi_teq' in kepler_df.columns:
            kepler_clean['eq_temperature'] = kepler_df['koi_teq']
        if 'koi_insol' in kepler_df.columns:
            kepler_clean['insolation_flux'] = kepler_df['koi_insol']
        if 'koi_steff' in kepler_df.columns:
            kepler_clean['stellar_temp'] = kepler_df['koi_steff']
        if 'koi_srad' in kepler_df.columns:
            kepler_clean['stellar_radius'] = kepler_df['koi_srad']
        if 'koi_slogg' in kepler_df.columns:
            kepler_clean['stellar_gravity'] = kepler_df['koi_slogg']
        
        # Add Kepler target and source
        if 'koi_disposition' in kepler_df.columns:
            disposition_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
            kepler_clean['disposition'] = kepler_df['koi_disposition'].map(disposition_map)
        kepler_clean['source'] = 'Kepler'
    
    # Process K2 data
    if len(k2_df) > 0:
        # Planet names for K2
        if 'pl_name' in k2_df.columns:
            k2_clean['planet_name'] = k2_df['pl_name']
        elif 'hostname' in k2_df.columns:
            k2_clean['planet_name'] = k2_df['hostname'] + ' b'  # Common exoplanet naming
        
        # Scientific features
        if 'pl_orbper' in k2_df.columns:
            k2_clean['orbital_period'] = k2_df['pl_orbper']
        if 'pl_rade' in k2_df.columns:
            k2_clean['planet_radius'] = k2_df['pl_rade']
        if 'pl_eqt' in k2_df.columns:
            k2_clean['eq_temperature'] = k2_df['pl_eqt']
        if 'pl_insol' in k2_df.columns:
            k2_clean['insolation_flux'] = k2_df['pl_insol']
        if 'st_teff' in k2_df.columns:
            k2_clean['stellar_temp'] = k2_df['st_teff']
        if 'st_rad' in k2_df.columns:
            k2_clean['stellar_radius'] = k2_df['st_rad']
        if 'st_logg' in k2_df.columns:
            k2_clean['stellar_gravity'] = k2_df['st_logg']
        
        # Add K2 target and source
        if 'disposition' in k2_df.columns:
            disposition_map = {'Confirmed': 2, 'Candidate': 1, 'False Positive': 0}
            k2_clean['disposition'] = k2_df['disposition'].map(disposition_map)
        k2_clean['source'] = 'K2'
    
    # Process TESS data
    if len(tess_df) > 0:
        # Planet names for TESS
        if 'toi' in tess_df.columns:
            tess_clean['planet_name'] = 'TOI ' + tess_df['toi'].astype(str)
        elif 'tid' in tess_df.columns:
            tess_clean['planet_name'] = 'TIC ' + tess_df['tid'].astype(str)
        
        # Scientific features
        if 'pl_orbper' in tess_df.columns:
            tess_clean['orbital_period'] = tess_df['pl_orbper']
        if 'pl_rade' in tess_df.columns:
            tess_clean['planet_radius'] = tess_df['pl_rade']
        if 'pl_trandep' in tess_df.columns:
            tess_clean['transit_depth'] = tess_df['pl_trandep']
        if 'pl_trandurh' in tess_df.columns:
            tess_clean['transit_duration'] = tess_df['pl_trandurh']
        if 'pl_eqt' in tess_df.columns:
            tess_clean['eq_temperature'] = tess_df['pl_eqt']
        if 'pl_insol' in tess_df.columns:
            tess_clean['insolation_flux'] = tess_df['pl_insol']
        if 'st_teff' in tess_df.columns:
            tess_clean['stellar_temp'] = tess_df['st_teff']
        if 'st_rad' in tess_df.columns:
            tess_clean['stellar_radius'] = tess_df['st_rad']
        if 'st_logg' in tess_df.columns:
            tess_clean['stellar_gravity'] = tess_df['st_logg']
        
        # Add TESS target and source
        if 'tfopwg_disp' in tess_df.columns:
            disposition_map = {'CP': 2, 'KP': 1, 'PC': 1, 'FP': 0}
            tess_clean['disposition'] = tess_df['tfopwg_disp'].map(disposition_map)
        tess_clean['source'] = 'TESS'
    
    # Merge all datasets
    all_dfs = []
    if len(kepler_clean) > 0:
        all_dfs.append(kepler_clean)
    if len(k2_clean) > 0:
        all_dfs.append(k2_clean) 
    if len(tess_clean) > 0:
        all_dfs.append(tess_clean)
    
    if not all_dfs:
        print("❌ No data could be loaded from any dataset!")
        return None
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"✅ Merged dataset size: {len(merged_df)}")
    print(f"📊 Sources: {merged_df['source'].value_counts().to_dict()}")
    
    # Basic info
    print("\n📈 Dataset Info:")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Columns: {list(merged_df.columns)}")
    
    if 'disposition' in merged_df.columns:
        print(f"Target distribution: {merged_df['disposition'].value_counts().sort_index()}")
    
    if 'planet_name' in merged_df.columns:
        print(f"Sample planet names: {merged_df['planet_name'].head(5).tolist()}")
    
    # Save merged dataset
    merged_df.to_csv('merged_exoplanets.csv', index=False)
    print(f"💾 Saved to 'merged_exoplanets.csv'")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_exoplanet_datasets()