import pandas as pd
import numpy as np
from io import StringIO

# -----------------------------
# Mock CSV data for testing
# -----------------------------
kepler_csv = """# Sample Kepler data
kepid,kepoi_name,koi_period,koi_prad,koi_depth,koi_duration,koi_teq,koi_insol,koi_steff,koi_srad,koi_slogg,koi_disposition
1001,KIC1001,365.25,1.1,0.01,10,300,200,5500,1,4.5,CONFIRMED
1002,KIC1002,10.5,2.3,0.02,5,600,400,6000,1.1,4.4,CANDIDATE
"""

k2_csv = """# Sample K2 data
pl_name,pl_orbper,pl_rade,pl_eqt,pl_insol,st_teff,st_rad,st_logg,disposition
K2-1,2.5,1.2,400,500,5000,1.0,4.3,Confirmed
K2-2,3.0,1.8,600,700,5200,1.1,4.4,Candidate
"""

tess_csv = """# Sample TESS data
toi,pl_orbper,pl_rade,pl_trandep,pl_trandurh,pl_eqt,pl_insol,st_teff,st_rad,st_logg,tfopwg_disp
1,1.5,1.0,0.01,2,350,300,5400,0.9,4.5,CP
2,2.0,1.5,0.02,3,450,400,5500,1.0,4.4,KP
"""

# -----------------------------
# Load CSV from string (simulate files)
# -----------------------------
kepler_df = pd.read_csv(StringIO(kepler_csv), comment='#')
k2_df = pd.read_csv(StringIO(k2_csv), comment='#')
tess_df = pd.read_csv(StringIO(tess_csv), comment='#')

# -----------------------------
# Merge datasets (simplified)
# -----------------------------
def merge_test_datasets():
    kepler_clean = pd.DataFrame()
    k2_clean = pd.DataFrame()
    tess_clean = pd.DataFrame()

    # Kepler
    if 'kepoi_name' in kepler_df.columns:
        kepler_clean['planet_name'] = kepler_df['kepoi_name']
    kepler_clean['orbital_period'] = kepler_df['koi_period']
    kepler_clean['planet_radius'] = kepler_df['koi_prad']
    kepler_clean['source'] = 'Kepler'

    # K2
    if 'pl_name' in k2_df.columns:
        k2_clean['planet_name'] = k2_df['pl_name']
    k2_clean['orbital_period'] = k2_df['pl_orbper']
    k2_clean['planet_radius'] = k2_df['pl_rade']
    k2_clean['source'] = 'K2'

    # TESS
    if 'toi' in tess_df.columns:
        tess_clean['planet_name'] = 'TOI ' + tess_df['toi'].astype(str)
    tess_clean['orbital_period'] = tess_df['pl_orbper']
    tess_clean['planet_radius'] = tess_df['pl_rade']
    tess_clean['source'] = 'TESS'

    # Merge all datasets
    merged_df = pd.concat([kepler_clean, k2_clean, tess_clean], ignore_index=True)

    # -----------------------------
    # Basic validation
    # -----------------------------
    print("✅ Merged dataset size:", len(merged_df))
    print("Sources:", merged_df['source'].value_counts().to_dict())

    # Check for duplicate planet names
    duplicates = merged_df['planet_name'][merged_df['planet_name'].duplicated()]
    if len(duplicates) > 0:
        print("⚠️ Duplicate planet names found:", duplicates.tolist())
    else:
        print("No duplicate planet names found.")

    # Print sample planets
    print("\nSample planets:")
    print(merged_df[['planet_name', 'orbital_period', 'planet_radius', 'source']].head(5))

    return merged_df

# -----------------------------
# Run test merge
# -----------------------------
if __name__ == "__main__":
    merged = merge_test_datasets()
    # Optionally save to CSV
    merged.to_csv("test_merged_exoplanets.csv", index=False)
    print("\n💾 Saved merged dataset to 'test_merged_exoplanets.csv'")
