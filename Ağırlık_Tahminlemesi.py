# Veri_Bilimine_Giriş_Proje
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import joblib
import logging
import warnings
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, LeaveOneOut, StratifiedKFold, ParameterSampler, ParameterGrid, cross_val_score
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import time
import sys, random
import re

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

random.seed(42)
np.random.seed(42)

# ── Simple terminal progress bar (no external dependency) ──────────────
class ProgressBar:
    def __init__(self, total, desc='', width=45):
        self.total = max(total, 1)
        self.desc  = desc
        self.width = width
        self.n     = 0
        self._lock = __import__('threading').Lock()
        self._render()

    def _render(self):
        filled = int(self.width * self.n / self.total)
        bar    = '#' * filled + '.' * (self.width - filled)
        pct    = 100 * self.n / self.total
        sys.stdout.write(f"\r  {self.desc}: [{bar}] {self.n}/{self.total}  %{pct:.0f}")
        sys.stdout.flush()

    def update(self, n=1):
        with self._lock:
            self.n = min(self.n + n, self.total)
            self._render()

    def close(self):
        filled = '#' * self.width
        sys.stdout.write(f"\r  {self.desc}: [{filled}] {self.total}/{self.total}  %100 DONE\n")
        sys.stdout.flush()

# ── Basic setup ──────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

VERI_DIR      = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'VERİLER')
ARA_DEGER_DIR = os.path.join(VERI_DIR, 'ARA DEĞER VERİLERİ')
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'MODEL_OUTPUTS', 'GB_OUTPUTS_2647')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'GLOBAL_CACHE_2647')
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)

_fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "load_est_2647.log"), encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_sh = logging.StreamHandler()
_sh.setLevel(logging.WARNING)
_sh.setFormatter(logging.Formatter("⚠️ %(levelname)s: %(message)s"))

_root = logging.getLogger()
_root.handlers.clear()
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURABLE HYPERPARAMETERS ⚙️
# (You can enter the results from hyperparameter tests here)
# ══════════════════════════════════════════════════════════════════════════════

# ── Synthetic Data Settings ──
SENTETIK_ORNEK_SAYISI = 5000   # Number of synthetic samples to add to original data
SENTETIK_NOISE_X      = 0.025 # Tolerance added to sensor measurements
SENTETIK_NOISE_Y      = 0.020 # Tolerance added to weight target

# ── Window (Duration) Settings ──
T_PENCERE_BASLANGIC   = 250   # Analysis start second (5th minute)
T_PENCERE_BITIS       = 1500  # Analysis end second (30th minute)
MIN_SEGMENT_SATIR     = 100
MAX_SLOPE_SURE        = 1800

# ── Gradient Boosting (GB) Model Settings ──
GB_N_ESTIMATORS       = 450
GB_MAX_DEPTH          = 2
GB_LEARNING_RATE      = 0.03
GB_SUBSAMPLE          = 0.8
GB_MIN_SAMPLES_SPLIT  = 5
GB_MIN_SAMPLES_LEAF   = 5
GB_MAX_FEATURES       = 'sqrt'

# ── Random Forest (RF) Model Settings ──
RF_N_ESTIMATORS       = 50
RF_MAX_DEPTH          = 5
RF_MIN_SAMPLES_SPLIT  = 5
RF_MIN_SAMPLES_LEAF   = 5
RF_MAX_FEATURES       = 'sqrt'
RF_BOOTSTRAP          = True

# ── Ensemble Weight Ratios ──
ENS_W_GB              = 0.60
ENS_W_RF              = 0.40

OZELLIK_ISIMLERI      = ["p_norm", "p_std", "v_mean", "energy_total",
                         "t3_slope", "t3_start", "t3_max", "t3_range",
                         "thermal_idx", "current_mean", "current_std", "current_rms",
                         "p_volatilite", "i_volatilite", "enerji_zaman_ratio",
                         "t3_dalgalanma", "t3_ramp_short", "isil_kapasite",
                         # NEW: DAQ additional features
                         "daq_energy_kwh", "daq_flowmeter", "daq_current", "daq_cosfi",
                         # NEW: MAGNUM additional features
                         "ntc2_mean", "ntc2_slope", "cycle_duration",
                         # NEW: Physical interaction features
                         "energy_per_flow", "power_per_current", "ntc_ratio_23"]

# Feature filter for model training
KULLANILACAK_OZELLIKLER = ["p_norm", "t3_slope", "t3_start", "thermal_idx", 
                           "ntc2_mean", "current_rms", "ntc_ratio_23", "t3_range"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 2647 INTERMEDIATE VALUE TESTS (DYNAMIC FILE DISCOVERY)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ⚖️  KNOWN ACTUAL WEIGHT VALUES (grams)
#
# Two key formats are supported:
#   "<WEIGHT>"        → applies to ALL runs of that weight  (e.g. "7KG": 6760)
#   "<WEIGHT>_RUN<N>" → overrides for a specific run only   (e.g. "7KG_RUN2": 6775)
#
# Resolution order (most specific wins):
#   1. "<WEIGHT>_RUN<N>"  exact key
#   2. "<WEIGHT>"          generic key
#   3. Nearest value from WeightDetection.xlsx  (if file is present)
#   4. Nominal KG × 1000  (automatic fallback)
#
# You only need to fill in values you actually have measured.
# Any weight/run combination not listed here will use the fallback.
# ══════════════════════════════════════════════════════════════════════════════
BILINEN_AGIRLIKLAR = {
    # ── RUN1 / Generic (applies to every run unless a per-run key overrides) ──
    "0.5KG":    534,
    "1KG":     1092,
    "1.5KG":   1486,
    "2KG":     2000,
    "2.5KG":   2538,
    "3KG":     3096,
    "3.5KG":   3362,
    "4KG":     4012,
    "4.5KG":   4392,
    "5KG":     5002,
    "5.5KG":   5506,
    "6KG":     6000,
    "6.5KG":   6502,
    "7KG":     6760,
    "7.5KG":   7510,
    "8KG":     8026,
    "8.5KG":   8512,
    "9KG":     8780,
    "9.5KG":   9524,
    "10KG":   10002,
    "10.5KG": 10432,
    "11KG":   11006,
    "11.5KG": 11708,
    "12KG":   12054,
    "12.5KG": 12504,
    "13KG":   13005,
    "13.5KG": 13520,
    "14KG":   13825,
    "14.5KG": 14488,

    # ── RUN2 ─────────────────────────────────────────────────────────────────
    "0.5KG_RUN2":    538,
    "1KG_RUN2":     1140,
    "1.5KG_RUN2":   1530,
    "2KG_RUN2":     2086,
    "2.5KG_RUN2":   2502,
    "3KG_RUN2":     3008,
    "3.5KG_RUN2":   3504,
    "4KG_RUN2":     4018,
    "4.5KG_RUN2":   4504,
    "5KG_RUN2":     5080,
    "5.5KG_RUN2":   5518,
    "6KG_RUN2":     6022,
    "6.5KG_RUN2":   6504,
    "7KG_RUN2":     6975,
    "7.5KG_RUN2":   7500,
    "8KG_RUN2":     8005,
    "8.5KG_RUN2":   8738,
    "9KG_RUN2":     8922,
    "9.5KG_RUN2":   9416,
    "10KG_RUN2":   10000,
    "10.5KG_RUN2": 10502,
    "11KG_RUN2":   11000,
    "11.5KG_RUN2": 11508,
    #"12KG_RUN2":   12000,
    "12.5KG_RUN2": 12600,
    "13KG_RUN2":   13065,
    "13.5KG_RUN2": 13565,
    "14KG_RUN2":   14085,

    # ── RUN3 ─────────────────────────────────────────────────────────────────
    # TODO: Ölçülen gerçek ağırlıkları (gram) buraya girin
    "0.6KG_RUN3":    660,
    "1.1KG_RUN3":    1124,
    "1.7KG_RUN3":    1690,
    #"2KG_RUN3":      0,
    "2.5KG_RUN3":    2478,
    "2.9KG_RUN3":    2940,
    "3.6KG_RUN3":    3648,
    #"4KG_RUN3":      0,
    "4.4KG_RUN3":    4404,
    #"5KG_RUN3":      0,
    #"5.5KG_RUN3":    0,
    "6.4KG_RUN3":    6375,
    "6.6KG_RUN3":    6614,
    #"7KG_RUN3":      0,
    "7.7KG_RUN3":    7720,
    #"8KG_RUN3":      0,
    #"8.5KG_RUN3":    0,
    #"9KG_RUN3":      0,
    #"9.5KG_RUN3":    0,
    #"10KG_RUN3":     0,
    "10.5KG_RUN3":   10550,
    #"11KG_RUN3":     0,
    #"11.5KG_RUN3":   0,
    #"12KG_RUN3":     0,
    "12.8KG_RUN3":   12820,
    #"13KG_RUN3":     0,
    #"13.5KG_RUN3":   0,
    #"14KG_RUN3":     0,
    #"14.5KG_RUN3":   0,
}

# ── File discovery regexes ────────────────────────────────────────────────────
# Supported formats (case-insensitive):
#   TD2647_1.5KG_RUN2_DAQ.xlsx
#   TD2647_1.5KG_RUN2_DAQ.csv
_DAQ_RE = re.compile(
    r'^TD2647_([\d\.]+(?:KG)?)_RUN(\d+)_DAQ\.(xlsx?|csv)$',
    re.IGNORECASE
)
_MAGNUM_RE = re.compile(
    r'^TD2647_([\d\.]+(?:KG)?)_RUN(\d+)_MAGNUM\.(xlsx?|csv)$',
    re.IGNORECASE
)

def _ara_deger_testleri_kesfet(klasor: str) -> list:
    """
    Scans ARA_DEGER_DIR and automatically discovers all DAQ/MAGNUM pairs.
    Adding any new TD2647_<WEIGHT>_RUN<N>_DAQ + _MAGNUM file pair to the
    folder is all that is needed — no code changes required.
    """
    testler = []

    # --- Cache spinned data from WeightDetection Excel ---
    spinned_listesi = []
    wd_path = os.path.join(klasor, 'TD2647_WeightDetection.xlsx')
    if os.path.exists(wd_path):
        try:
            df_w = pd.read_excel(wd_path, sheet_name='Table tests', header=None)
            for i in range(25):
                row = df_w.iloc[i].astype(str).str.strip().str.lower()
                if 'spinned' in row.values:
                    spinned_col = (row == 'spinned').idxmax()
                    for v in df_w.iloc[i+1:, spinned_col]:
                        try:
                            v_clean = str(v).strip().replace(',', '.')
                            spinned_listesi.append(int(float(v_clean)))
                        except:
                            pass
                    break
        except Exception as e:
            logger.warning(f"Could not read WeightDetection: {e}")

    try:
        tum_dosyalar = os.listdir(klasor)
    except FileNotFoundError:
        logger.error(f"Intermediate value directory not found: {klasor}")
        return testler

    # Build a case-insensitive lookup: normalised_name → original_name
    dosya_harita = {d.lower(): d for d in tum_dosyalar}

    # ── Collect every DAQ file and look for its MAGNUM counterpart ──────────
    for dosya in sorted(tum_dosyalar):
        m = _DAQ_RE.match(dosya)
        if not m:
            continue

        ham_tip = m.group(1).upper()   # e.g. '1.5KG'  or  '9.5' (no KG)
        run_no  = int(m.group(2))
        uzanti  = m.group(3)            # 'xlsx' / 'csv'

        # Normalise: ensure KG suffix is always present
        tip = ham_tip if ham_tip.endswith('KG') else ham_tip + 'KG'

        # Build expected MAGNUM stem (same weight / run / extension variants)
        # Use the ORIGINAL capitalisation of the DAQ stem, replace _DAQ with _MAGNUM
        daq_stem   = dosya[:-(len(uzanti) + 1)]  # strip extension
        magnum_stem = re.sub(r'_DAQ$', '_MAGNUM', daq_stem, flags=re.IGNORECASE)

        magnum_dosya = None
        for ext in ('xlsx', 'csv', 'XLSX', 'CSV'):
            aday_lower = (magnum_stem + '.' + ext).lower()
            if aday_lower in dosya_harita:
                magnum_dosya = dosya_harita[aday_lower]   # use real capitalisation
                break

        if not magnum_dosya:
            logger.warning(f"MAGNUM pair not found for: {dosya}  →  skipping")
            continue

        # ── Determine actual weight ──────────────────────────────────────────
        try:
            nominal_g = int(float(tip.replace('KG', '')) * 1000)
        except ValueError:
            nominal_g = 0

        # Priority 1a: per-run specific key  → e.g. "7KG_RUN2"
        run_key     = f"{tip}_RUN{run_no}"
        agirlik_g   = BILINEN_AGIRLIKLAR.get(run_key)

        # Priority 1b: generic weight key    → e.g. "7KG"
        if agirlik_g is None:
            agirlik_g = BILINEN_AGIRLIKLAR.get(tip)

        # Priority 2: nearest spinned value from WeightDetection Excel
        if agirlik_g is None and nominal_g > 0 and spinned_listesi:
            agirlik_g = min(spinned_listesi, key=lambda x: abs(x - nominal_g))

        # Priority 3: fall back to nominal (KG × 1000)
        if agirlik_g is None:
            agirlik_g = nominal_g
            logger.info(f"{tip} RUN{run_no}: not in dictionary — using nominal {nominal_g}g")

        # Log which key was actually used (helpful for debugging)
        used_key = run_key if BILINEN_AGIRLIKLAR.get(run_key) is not None else tip
        logger.debug(f"{tip} RUN{run_no}: {agirlik_g}g  (key: '{used_key}')")

        testler.append({
            "m":         "TD2647",
            "tip":       tip,
            "run":       run_no,
            "agirlik_g": agirlik_g,
            "daq":       dosya,
            "magnum":    magnum_dosya,
        })

    if not testler:
        logger.error(f"No DAQ/MAGNUM pairs found: {klasor}")
    else:
        run_counts = {}
        for t in testler:
            run_counts[t['run']] = run_counts.get(t['run'], 0) + 1
        run_summary = ", ".join(f"RUN{r}: {c} pair(s)" for r, c in sorted(run_counts.items()))
        logger.info(f"{len(testler)} test pairs discovered [{run_summary}] → {klasor}")
        print(f"\n  📂 {len(testler)} file pair(s) detected: {run_summary}")

    # Sort: weight ascending, then run number ascending
    testler = sorted(testler, key=lambda x: (x['agirlik_g'], x['run']))
    return testler

ARA_DEGER_TESTLER = _ara_deger_testleri_kesfet(ARA_DEGER_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA READING FUNCTIONS (REINFORCED)
# ══════════════════════════════════════════════════════════════════════════════

def saniyeye_cevir(zaman_obj):
    try:
        if pd.isnull(zaman_obj): return None
        s = str(zaman_obj).strip()
        if ' ' in s: s = s.split(' ')[-1]
        p = s.split(':')
        if len(p) == 3: return int(p[0])*3600 + int(p[1])*60 + int(p[2])
        if len(p) == 2: return int(p[0])*60 + int(p[1])
        return float(s)
    except Exception as exc:
        return None

def find_val_col(cols, keyword, forbidden='TIME'):
    candidates = [c for c in cols if keyword.upper() in str(c).upper() and forbidden.upper() not in str(c).upper()]
    return candidates[-1] if candidates else None

def safe_float(val):
    """Converts text, comma-separated, or dirty numbers cleanly to float."""
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    v_str = str(val).replace(',', '.')
    v_str = re.sub(r'[^\d\.-]', '', v_str) # Remove everything except digits, minus and dot
    try:
        return float(v_str)
    except:
        return np.nan

def magnum_oku(magnum_path: str):
    df = None
    try:
        if magnum_path.lower().endswith('.csv'):
            encodings = ['utf-8', 'cp1254', 'latin1']
            header_idx = -1
            sep = ';'
            for enc in encodings:
                try:
                    with open(magnum_path, 'r', encoding=enc) as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines[:50]):
                        line_upper = line.upper()
                        if 'CURRENTAMPLITUDE' in line_upper or 'NTC' in line_upper:
                            header_idx = i
                            if ';' in line: sep = ';'
                            elif ',' in line: sep = ','
                            elif '\t' in line: sep = '\t'
                            break
                    if header_idx != -1:
                        df = pd.read_csv(magnum_path, sep=sep, header=header_idx, encoding=enc, on_bad_lines='skip', low_memory=False)
                        break 
                except: pass
        else:
            # NEW: Smart Header Finder in Excel File
            df_raw = pd.read_excel(magnum_path, header=None, nrows=50)
            header_idx = 23 # Classic default
            for i, row in df_raw.iterrows():
                row_str = " ".join([str(x).upper() for x in row])
                if 'CURRENTAMPLITUDE' in row_str or 'NTC' in row_str:
                    header_idx = i
                    break
            df = pd.read_excel(magnum_path, header=header_idx)
            
        if df is None:
            logger.error(f"magnum_oku: File could not be read or header not found → {magnum_path}")
            return None, None
            
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as exc:
        logger.error(f"magnum_oku: File reading error → {magnum_path}  ({exc})")
        return None, None

    # --- FLEXIBLE (SMART) COLUMN FINDING ---
    # Time column (System/Recording Time should be excluded)
    sn_col = next((c for c in df.columns if 'TOTAL MILLISECOND' in c.upper()), None)
    if not sn_col:
        sn_col = next((c for c in df.columns if 'MILLISECOND' in c.upper()), None)
    if not sn_col:
        sn_col = next((c for c in df.columns if 'TIME' in c.upper() and 'SYSTEM' not in c.upper() and 'RECORDING' not in c.upper()), None)

    t3_col = next((c for c in df.columns if 'NTC3' in c.upper() or 'NTC 3' in c.upper()), None)
    cur_col = next((c for c in df.columns if 'CURRENTAMPLITUDE' in c.upper()), None)

    # NEW: NTC2 and cycle_duration columns (optional)
    ntc2_col  = next((c for c in df.columns if 'NTC2' in c.upper() or 'NTC 2' in c.upper()), None)
    cycle_col = next((c for c in df.columns if 'cycle_duration' in str(c).lower()), None)

    if not t3_col or not cur_col or not sn_col:
        logger.error(f"magnum_oku: Expected columns not found! (Sn:{sn_col}, T3:{t3_col}, Cur:{cur_col}) → {magnum_path}")
        return None, None

    # Force-convert all columns to numeric using safe_float
    df['sn']    = df[sn_col].apply(safe_float) / 1000
    df[t3_col]  = df[t3_col].apply(safe_float) / 10
    df[cur_col] = df[cur_col].apply(safe_float)
    if ntc2_col:
        df[ntc2_col] = df[ntc2_col].apply(safe_float) / 10
    if cycle_col:
        df[cycle_col] = df[cycle_col].apply(safe_float)
    
    df = df.dropna(subset=['sn', t3_col, cur_col]).reset_index(drop=True)

    if df.empty or len(df) < 10:
        logger.error(f"magnum_oku: Data invalid or could not be converted to numeric (empty after dropna) → {magnum_path}")
        return None, None

    t0  = df['sn'].iloc[0]
    seg = df[(df['sn'] >= t0 + T_PENCERE_BASLANGIC) & (df['sn'] < t0 + T_PENCERE_BITIS)]
    
    # Magnum files may have different lengths; if the file is too short
    # seg may remain empty. In that case, we take from 1/4 to the end.
    if seg.empty or len(seg) < 10:
        seg = df.iloc[len(df)//4 : len(df)]

    current_mean = float(seg[cur_col].mean())
    t3_start_v = float(df[t3_col].iloc[0])
    
    # To prevent the cooling phase in long experiments from distorting the slope,
    # we calculate t3_slope within the 'seg' period (e.g., first 30 min) instead of the entire file.
    t3_end_calc   = float(seg[t3_col].iloc[-1])
    toplam_dk_calc  = (float(seg['sn'].iloc[-1]) - float(df['sn'].iloc[0])) / 60.0
    
    if toplam_dk_calc < 1.0:
        toplam_dk_calc = 1.0
        
    t3_slope_v = (t3_end_calc - t3_start_v) / toplam_dk_calc
    if abs(t3_slope_v) < 0.01: 
        t3_slope_v = 0.1
    # Convert negative or very low slope to positive (may be setup/cooling period)
    if t3_slope_v < 0.05:
        t3_slope_v = 0.1

    current_std  = float(seg[cur_col].std()) if len(seg) > 1 else 0.0
    current_rms  = float(np.sqrt((seg[cur_col] ** 2).mean()))
    t3_max       = float(df[t3_col].max())
    t3_range     = float(df[t3_col].max() - df[t3_col].min())
    
    # NEW: Fast ramp-up slope from first 10 rows (2nd minute)
    seg_short = seg.iloc[:min(10, len(seg))]
    if len(seg_short) > 1:
        t3_ramp_short = (float(seg_short[t3_col].iloc[-1]) - t3_start_v) / 2.0
    else:
        t3_ramp_short = 0.0

    # ── NEW: NTC2 statistics ──
    if ntc2_col and ntc2_col in seg.columns:
        ntc2_vals = seg[ntc2_col].dropna()
        ntc2_mean_v  = float(ntc2_vals.mean()) if len(ntc2_vals) > 0 else 0.0
        ntc2_start_v = float(df[ntc2_col].dropna().iloc[0]) if len(df[ntc2_col].dropna()) > 0 else ntc2_mean_v
        ntc2_end_v   = float(ntc2_vals.iloc[-1]) if len(ntc2_vals) > 0 else ntc2_start_v
        ntc2_slope_v = (ntc2_end_v - ntc2_start_v) / max(toplam_dk_calc, 1.0)
    else:
        ntc2_mean_v  = 0.0
        ntc2_slope_v = 0.0

    # ── NEW: cycle_duration statistics ──
    if cycle_col and cycle_col in seg.columns:
        cycle_vals = seg[cycle_col].dropna()
        cycle_dur_mean = float(cycle_vals.mean()) if len(cycle_vals) > 0 else 0.0
    else:
        cycle_dur_mean = 0.0

    return t3_slope_v, current_mean, t3_start_v, current_std, current_rms, t3_max, t3_range, t3_ramp_short, ntc2_mean_v, ntc2_slope_v, cycle_dur_mean

def veri_dogru_mu(feat: np.ndarray, agirlik_g: float, is_first_pass=False) -> tuple:
    """
    Detect extreme and erroneous data entries.
    is_first_pass=True → Soft checks (wide ranges on first pass)
    Returns: (valid: bool, errors: list)
    """
    validasyon_hatalari = []
    
    if feat is None:
        return False, ["Feature None"]
    
    # First pass: wide checks (allow most data through)
    # Second pass: strict checks (anomaly detection)
    
    if is_first_pass:
        # [0] p_norm (power) — wide range
        if feat[0] < 50 or feat[0] > 5000:
            validasyon_hatalari.append(f"Power out of range: {feat[0]:.0f}W")
        
        # [4] t3_slope (thermal slope) — negative or 0 not expected
        if feat[4] < -10 or feat[4] > 200:
            validasyon_hatalari.append(f"T3 slope invalid: {feat[4]:.2f}°C/min")
        
        # [9] current_mean — expected: 0.1-100A (wide)
        if feat[9] < 0 or feat[9] > 100:
            validasyon_hatalari.append(f"Current invalid: {feat[9]:.1f}A")
        
        # Weight — expected: 100-20000g
        if agirlik_g < 100 or agirlik_g > 20000:
            validasyon_hatalari.append(f"Weight out of range: {agirlik_g:.0f}g")
    
    else:
        # Second pass: strict anomaly checks
        # [0] p_norm (power) 
        if feat[0] < 100 or feat[0] > 3000:
            validasyon_hatalari.append(f"Power suspicious: {feat[0]:.0f}W")
        
        # [4] t3_slope — should be positive or not too small
        if feat[4] <= 0.01 or feat[4] > 150:
            validasyon_hatalari.append(f"T3 slope suspicious: {feat[4]:.2f}°C/min")
    
    # NaN/Inf check in both passes
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        validasyon_hatalari.append("NaN/Inf value detected!")
    
    return len(validasyon_hatalari) == 0, validasyon_hatalari

def veri_ayikla_2647(s: dict):
    tag = f"2647 {s['tip']} R{s['run']}"
    daq_path = os.path.join(ARA_DEGER_DIR, s['daq'])
    
    if not os.path.exists(daq_path):
        logger.warning(f"{tag}: DAQ not found → {daq_path}")
        return None, None

    try:
        df_daq = pd.read_excel(daq_path)
        df_daq.columns = [str(c).strip() for c in df_daq.columns]
        cols_daq = list(df_daq.columns)

        p_col = find_val_col(cols_daq, 'POWER') or find_val_col(cols_daq, 'GÜÇ')
        v_col = find_val_col(cols_daq, 'VOLTAGE') or find_val_col(cols_daq, 'GERİLİM')
        if not p_col or not v_col: return None, None

        # NEW: Find additional DAQ columns (optional)
        energy_kwh_col = find_val_col(cols_daq, 'ENERGY')
        flow_col       = find_val_col(cols_daq, 'FLOWMETER')
        cur_daq_col    = find_val_col(cols_daq, 'CURRENT')
        cosfi_col      = find_val_col(cols_daq, 'COSfi')

        p_idx    = cols_daq.index(p_col)
        t_search = [c for i, c in enumerate(cols_daq) if 'TIME' in c.upper() and i <= p_idx]
        t_col    = t_search[-1] if t_search else cols_daq[0]

        df_daq['mutlak_sn'] = df_daq[t_col].apply(saniyeye_cevir)
        
        # FIX: Handle comma decimals in DAQ files
        df_daq[p_col] = df_daq[p_col].apply(safe_float)
        df_daq[v_col] = df_daq[v_col].apply(safe_float)
        
        for _col in [energy_kwh_col, flow_col, cur_daq_col, cosfi_col]:
            if _col:
                df_daq[_col] = df_daq[_col].apply(safe_float)
                
        df_daq = df_daq.dropna(subset=['mutlak_sn', p_col]).reset_index(drop=True)

        t0      = df_daq['mutlak_sn'].iloc[0]
        segment = df_daq[(df_daq['mutlak_sn'] >= t0 + T_PENCERE_BASLANGIC) &
                         (df_daq['mutlak_sn'] <  t0 + T_PENCERE_BITIS)]
        if segment.empty or len(segment) < 5:
            segment = df_daq.iloc[len(df_daq)//4 : len(df_daq)]

        v_mean_raw = segment[v_col].mean()
        v_real  = v_mean_raw * (10 if v_mean_raw < 100 else 1)
        p_norm  = segment[p_col].mean() * (230 / max(v_real, 1))
        p_std   = float(segment[p_col].std()) if len(segment) > 1 else 0.0
        v_mean  = float(v_real)

        # Total energy: power × duration (Wh)
        sure_saat = (segment['mutlak_sn'].iloc[-1] - segment['mutlak_sn'].iloc[0]) / 3600.0
        energy_total = p_norm * max(sure_saat, 0.001)

        # ── NEW: DAQ Additional Features ──
        daq_energy_kwh     = float(segment[energy_kwh_col].iloc[-1]) if energy_kwh_col and energy_kwh_col in segment.columns else 0.0
        daq_flowmeter_mean = float(segment[flow_col].mean()) if flow_col and flow_col in segment.columns else 0.0
        daq_current_mean   = float(segment[cur_daq_col].mean()) if cur_daq_col and cur_daq_col in segment.columns else 0.0
        daq_cosfi_mean     = float(segment[cosfi_col].mean()) if cosfi_col and cosfi_col in segment.columns else 0.0

        magnum_path = os.path.join(ARA_DEGER_DIR, s['magnum'])
        res_mag = magnum_oku(magnum_path)
        if res_mag[0] is None: return None, None
        
        t3_slope, current_mean, t3_start, current_std, current_rms, t3_max, t3_range, t3_ramp_short, ntc2_mean, ntc2_slope, cycle_dur_mean = res_mag
        
        # Thermal index calculation
        thermal_idx = p_norm / max(t3_slope, 0.1)
        thermal_idx = np.clip(thermal_idx, 1, 10000)

        # ─── DERIVED FEATURES ───
        p_volatilite       = p_std / max(p_norm, 1)
        i_volatilite       = current_std / max(current_mean, 1)
        enerji_zaman_ratio = p_norm / max(energy_total, 0.01)
        t3_dalgalanma      = t3_range / max(t3_max, 1)
        isil_kapasite      = t3_range / max(t3_slope, 0.01)

        # ─── NEW: PHYSICAL INTERACTION FEATURES ───
        energy_per_flow   = daq_energy_kwh / max(daq_flowmeter_mean, 0.01)
        power_per_current = p_norm / max(daq_current_mean, 0.01)
        ntc_ratio_23      = ntc2_mean / max(t3_max, 1.0)

        # 28 features — must match OZELLIK_ISIMLERI one-to-one
        feat = np.array([
            p_norm, p_std, v_mean, energy_total,
            t3_slope, t3_start, t3_max, t3_range,
            thermal_idx, current_mean, current_std, current_rms,
            p_volatilite, i_volatilite, enerji_zaman_ratio,
            t3_dalgalanma, t3_ramp_short, isil_kapasite,
            # DAQ additional
            daq_energy_kwh, daq_flowmeter_mean, daq_current_mean, daq_cosfi_mean,
            # MAGNUM additional
            ntc2_mean, ntc2_slope, cycle_dur_mean,
            # Interaction
            energy_per_flow, power_per_current, ntc_ratio_23
        ], dtype=float)

        # ── NaN / Inf diagnosis: log which feature is problematic, then patch ──
        bad_idx = [i for i, v in enumerate(feat) if not np.isfinite(v)]
        if bad_idx:
            bad_names = [(OZELLIK_ISIMLERI[i], feat[i]) for i in bad_idx]
            logger.info(
                f"{tag}: NaN/Inf in features → "
                + ", ".join(f"{n}={v}" for n, v in bad_names)
            )
            # Check if any *used* feature is bad
            used_bad = [OZELLIK_ISIMLERI[i] for i in bad_idx
                        if OZELLIK_ISIMLERI[i] in KULLANILACAK_OZELLIKLER]
            if used_bad:
                logger.info(f"{tag}: [FILTERED] Corrupted DAQ sensor data detected (missing {', '.join(used_bad)}). Test excluded to protect model.")
                return None, None
            # Otherwise patch unused bad features with 0 and continue
            feat[bad_idx] = 0.0
            logger.info(f"{tag}: patched unused NaN/Inf features with 0, continuing")

        # Data validation (first pass: soft checks)
        is_valid, errors = veri_dogru_mu(feat, float(s['agirlik_g']), is_first_pass=True)
        if not is_valid:
            logger.warning(f"{tag} data validation error: {', '.join(errors)}")
            return None, None
        
        return feat, float(s['agirlik_g'])
    except Exception as exc:
        logger.error(f"{tag} error: {exc}")
        return None, None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING & TESTING
# ══════════════════════════════════════════════════════════════════════════════

def mod_secim_al():
    if not sys.stdin.isatty(): return 1, 1
    print("\n  " + "=" * 70)
    print("  HOW WOULD YOU LIKE TO PROCEED?")
    print("  [1] Normal Training & Test (Use Current Settings and Indices)")
    print("  [2] Detailed Hyperparameter Test (Finds the optimal settings)")
    print("  " + "=" * 70)
    sec = input("  Your choice [1/2] (ENTER=1): ").strip()
    if sec == "2":
        print("\n  " + "-" * 70)
        print("  HYPERPARAMETER TEST TYPE:")
        print("  [1] EXHAUSTIVE TEST (Tries all possibilities incl. duration & synthetic - TAKES LONG)")
        print("  [2] QUICK TEST (ML test only for current settings above - SHORT DURATION)")
        print("  " + "-" * 70)
        alt_sec = input("  Your choice [1/2] (ENTER=1): ").strip()
        return 2, (2 if alt_sec == "2" else 1)
    return 1, 1

def eval_param(param, model_class, X, y, cv_list, n_jobs_model, syn_sayi, syn_nx, syn_ny):
    scores = []
    rng = np.random.default_rng(42)
    for tr, te in cv_list:
        X_tr = X[tr].copy(); y_tr = y[tr].copy()
        
        if syn_sayi > 0:
            idx_syn = rng.choice(len(X_tr), syn_sayi, replace=True)
            noise_m = 1 + rng.normal(0, syn_nx, (syn_sayi, X_tr.shape[1]))
            X_syn   = X_tr[idx_syn] * noise_m
            y_syn   = y_tr[idx_syn] * (1 + rng.normal(0, syn_ny, syn_sayi))
            X_trE   = np.vstack([X_tr, X_syn])
            y_trE   = np.concatenate([y_tr, y_syn])
        else:
            X_trE, y_trE = X_tr, y_tr
            
        kwargs = {"random_state": 42}
        if n_jobs_model: kwargs["n_jobs"] = n_jobs_model
        
        model = model_class(**param, **kwargs)
        model.fit(X_trE, y_trE)
        
        pred = model.predict(X[te])
        scores.append(np.abs((y[te] - pred) / y[te])[0] * 100)
    return param, -np.mean(scores)

def barli_arama(model_class, param_list, X, y, cv_list, desc, n_jobs_model, syn_sayi, syn_nx, syn_ny):
    bar = ProgressBar(len(param_list), desc=desc)
    best_score = float('-inf')
    best_params = None
    
    parallel = Parallel(n_jobs=-1, return_as='generator')
    gen = parallel(delayed(eval_param)(p, model_class, X, y, cv_list, n_jobs_model, syn_sayi, syn_nx, syn_ny) for p in param_list)
    
    for param, score in gen:
        if score > best_score:
            best_score = score
            best_params = param
        bar.update()
    bar.close()
    return best_params, best_score

def hiperparametre_test_yap(test_tipi):
    global T_PENCERE_BASLANGIC, T_PENCERE_BITIS, SENTETIK_ORNEK_SAYISI, SENTETIK_NOISE_X, SENTETIK_NOISE_Y
    
    if test_tipi == 1:
        sure_ayarlari = [
            {"bas": 300, "bit": 1800},
            {"bas": 250, "bit": 1500},
            {"bas": 350, "bit": 2000}
        ]
        
        sentetik_ayarlari = [
            {"sayi": 800, "nx": 0.005, "ny": 0.003},
            {"sayi": 400, "nx": 0.003, "ny": 0.002},
            {"sayi": 1200, "nx": 0.008, "ny": 0.005}
        ]
        test_ad = "EXHAUSTIVE HYPERPARAMETER OPTIMIZATION"
    else:
        sure_ayarlari = [
            {"bas": T_PENCERE_BASLANGIC, "bit": T_PENCERE_BITIS}
        ]
        sentetik_ayarlari = [
            {"sayi": SENTETIK_ORNEK_SAYISI, "nx": SENTETIK_NOISE_X, "ny": SENTETIK_NOISE_Y}
        ]
        test_ad = "QUICK (FIXED DATA) HYPERPARAMETER OPTIMIZATION"
    
    gb_space = {
        "n_estimators": [100, 200, 300, 500, 750, 1000],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
        "subsample": [0.55, 0.65, 0.75, 0.80, 0.90, 1.00],
        "min_samples_split": [2, 3, 4, 5, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": ["sqrt", "log2", None]
    }
    rf_space = {
        "n_estimators": [100, 200, 300, 500, 750, 1000],
        "max_depth": [3, 4, 5, 6, 7, 8, 10, None],
        "min_samples_split": [2, 3, 4, 5, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
        "bootstrap": [True, False]
    }

    genel_en_iyi_mape = float('inf')
    best_rapor = {}

    toplam_senaryo = len(sure_ayarlari) * len(sentetik_ayarlari)
    mevcut_senaryo = 0
    
    print("\n" + "="*80)
    print(f"  {test_ad} STARTING...")
    print(f"  Total Main Scenarios to Process (Duration x Synthetic) : {toplam_senaryo}")
    print("="*80)

    for sur in sure_ayarlari:
        T_PENCERE_BASLANGIC = sur["bas"]
        T_PENCERE_BITIS     = sur["bit"]
        
        features_list, weights_list = [], []
        print(f"\n  >> [WINDOW: {sur['bas']} - {sur['bit']}] Reading second range...")
        for s in ARA_DEGER_TESTLER:
            f, w = veri_ayikla_2647(s)
            features_list.append(f)
            weights_list.append(w)
            
        secilen_idx = [OZELLIK_ISIMLERI.index(col) for col in KULLANILACAK_OZELLIKLER]
        filtered_list = [f[secilen_idx] if f is not None else None for f in features_list]
            
        valid = [i for i, f in enumerate(filtered_list) if f is not None]
        if len(valid) < 5: continue
        X = np.array([filtered_list[i] for i in valid])
        y = np.array([weights_list[i] for i in valid])
        
        # NEW: 5-Fold CV (instead of LOO) — 20x faster!
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Bin y values (categorize for stratified split)
        y_bins = np.digitize(y, bins=np.percentile(y, [20, 40, 60, 80]))
        cv_splits = list(cv.split(X, y_bins))
        
        for sen in sentetik_ayarlari:
            mevcut_senaryo += 1
            print(f"\n  --- Scenario {mevcut_senaryo}/{toplam_senaryo} ---")
            print(f"  Window: {sur['bas']}-{sur['bit']} | Synthetic: {sen['sayi']} (NX:{sen['nx']}, NY:{sen['ny']})")

            # 1. GB Wide Search
            gb_rs_list = list(ParameterSampler(gb_space, n_iter=300, random_state=42))
            bp_gb, bscore_gb = barli_arama(GradientBoostingRegressor, gb_rs_list, X, y, cv_splits, 
                                           "[STEP 1/5] GB Wide Search ", None,
                                           syn_sayi=sen['sayi'], syn_nx=sen['nx'], syn_ny=sen['ny'])
            
            # 2. GB Fine Tuning
            gb_dar = {
                "n_estimators": sorted({max(50, bp_gb["n_estimators"]-150), bp_gb["n_estimators"], bp_gb["n_estimators"]+150}),
                "max_depth": sorted({max(1, bp_gb["max_depth"]-1), bp_gb["max_depth"], min(8, bp_gb["max_depth"]+1)}),
                "learning_rate": sorted({round(bp_gb["learning_rate"]*0.5, 4), bp_gb["learning_rate"], round(bp_gb["learning_rate"]*2.0, 4)}),
                "subsample": sorted({round(max(0.5, bp_gb["subsample"]-0.1),2), bp_gb["subsample"], round(min(1.0, bp_gb["subsample"]+0.1),2)}),
                "min_samples_split": [bp_gb["min_samples_split"]],
                "min_samples_leaf": [bp_gb["min_samples_leaf"]],
                "max_features": [bp_gb["max_features"]]
            }
            gb_gs_list = list(ParameterGrid(gb_dar))
            bp_gb_final, bscore_gb_final = barli_arama(GradientBoostingRegressor, gb_gs_list, X, y, cv_splits, 
                                           "[STEP 2/5] GB Fine Tuning ", None,
                                           syn_sayi=sen['sayi'], syn_nx=sen['nx'], syn_ny=sen['ny'])

            # 3. RF Wide Search
            rf_rs_list = list(ParameterSampler(rf_space, n_iter=300, random_state=42))
            bp_rf, bscore_rf = barli_arama(RandomForestRegressor, rf_rs_list, X, y, cv_splits, 
                                           "[STEP 3/5] RF Wide Search ", 1,
                                           syn_sayi=sen['sayi'], syn_nx=sen['nx'], syn_ny=sen['ny'])

            # 4. RF Fine Tuning
            md_rf = [None] if bp_rf["max_depth"] is None else sorted({max(2, bp_rf["max_depth"]-1), bp_rf["max_depth"], bp_rf["max_depth"]+1})
            rf_dar = {
                "n_estimators": sorted({max(50, bp_rf["n_estimators"]-150), bp_rf["n_estimators"], bp_rf["n_estimators"]+150}),
                "max_depth": md_rf,
                "min_samples_split": [bp_rf["min_samples_split"]],
                "min_samples_leaf": [bp_rf["min_samples_leaf"]],
                "max_features": [bp_rf["max_features"]],
                "bootstrap": [bp_rf["bootstrap"]]
            }
            rf_gs_list = list(ParameterGrid(rf_dar))
            bp_rf_final, bscore_rf_final = barli_arama(RandomForestRegressor, rf_gs_list, X, y, cv_splits, 
                                           "[STEP 4/5] RF Fine Tuning ", 1,
                                           syn_sayi=sen['sayi'], syn_nx=sen['nx'], syn_ny=sen['ny'])

            # 5. Ensemble Test
            bar5 = ProgressBar(15, desc="[STEP 5/5] Ensemble Calc  ")
            y_gb_loo = np.zeros(len(X))
            y_rf_loo = np.zeros(len(X))
            rng = np.random.default_rng(42)
            for tr, te in cv_splits:
                X_tr = X[tr].copy(); y_tr = y[tr].copy()
                if sen['sayi'] > 0:
                    idx_syn = rng.choice(len(X_tr), sen['sayi'], replace=True)
                    noise_m = 1 + rng.normal(0, sen['nx'], (sen['sayi'], X_tr.shape[1]))
                    X_syn   = X_tr[idx_syn] * noise_m
                    y_syn   = y_tr[idx_syn] * (1 + rng.normal(0, sen['ny'], sen['sayi']))
                    X_trE   = np.vstack([X_tr, X_syn])
                    y_trE   = np.concatenate([y_tr, y_syn])
                else:
                    X_trE, y_trE = X_tr, y_tr
                
                g = GradientBoostingRegressor(**bp_gb_final, random_state=42).fit(X_trE, y_trE)
                r = RandomForestRegressor(**bp_rf_final, random_state=42, n_jobs=-1).fit(X_trE, y_trE)
                y_gb_loo[te] = g.predict(X[te])
                y_rf_loo[te] = r.predict(X[te])
            
            b_w = (0.7, 0.3)
            b_m = float('inf')
            for wx in range(30, 101, 5):
                wg = wx / 100.0
                wr = round(1.0 - wg, 2)
                ens = wg * y_gb_loo + wr * y_rf_loo
                mape = float(np.mean(np.abs((ens - y) / y)) * 100)
                if mape < b_m:
                    b_m = mape
                    b_w = (wg, wr)
                bar5.update()
            bar5.close()
            
            print(f"  > This Scenario Error Rate: %{b_m:.3f}")
            
            if b_m < genel_en_iyi_mape:
                genel_en_iyi_mape = b_m
                best_rapor = {
                    "sur": sur, "sen": sen,
                    "bp_gb": bp_gb_final, "bp_rf": bp_rf_final, "bw": b_w
                }
                
    # === RESULT REPORTING ===
    bp_gb_final = best_rapor['bp_gb']
    bp_rf_final = best_rapor['bp_rf']
    en_iyi_syn = best_rapor['sen']
    en_iyi_sur = best_rapor['sur']
    bw = best_rapor['bw']
    
    print("\n\n" + "#" * 80)
    print("  🏆 COMPREHENSIVE TEST COMPLETED - BEST VALUES (COPY-PASTE) 🏆")
    print(f"  EXPECTED ERROR RATE WITH THESE VALUES: %{genel_en_iyi_mape:.3f}")
    print("#" * 80 + "\n")
    print("      # ── Synthetic Data Settings ──")
    print(f"      SENTETIK_ORNEK_SAYISI = {en_iyi_syn['sayi']}")
    print(f"      SENTETIK_NOISE_X      = {en_iyi_syn['nx']}")
    print(f"      SENTETIK_NOISE_Y      = {en_iyi_syn['ny']}\n")
    print("      # ── Window (Duration) Settings ──")
    print(f"      T_PENCERE_BASLANGIC   = {en_iyi_sur['bas']}")
    print(f"      T_PENCERE_BITIS       = {en_iyi_sur['bit']}")
    print("      MIN_SEGMENT_SATIR     = 100")
    print("      MAX_SLOPE_SURE        = 1800\n")
    print("      # ── Gradient Boosting (GB) Model Settings ──")
    print(f"      GB_N_ESTIMATORS       = {bp_gb_final['n_estimators']}")
    print(f"      GB_MAX_DEPTH          = {bp_gb_final['max_depth']}")
    print(f"      GB_LEARNING_RATE      = {bp_gb_final['learning_rate']}")
    print(f"      GB_SUBSAMPLE          = {bp_gb_final['subsample']}")
    print(f"      GB_MIN_SAMPLES_SPLIT  = {bp_gb_final['min_samples_split']}")
    print(f"      GB_MIN_SAMPLES_LEAF   = {bp_gb_final['min_samples_leaf']}")
    v_feat_gb = f"'{bp_gb_final['max_features']}'" if isinstance(bp_gb_final['max_features'], str) else bp_gb_final['max_features']
    print(f"      GB_MAX_FEATURES       = {v_feat_gb}\n")
    print("      # ── Random Forest (RF) Model Settings ──")
    print(f"      RF_N_ESTIMATORS       = {bp_rf_final['n_estimators']}")
    print(f"      RF_MAX_DEPTH          = {bp_rf_final['max_depth']}")
    print(f"      RF_MIN_SAMPLES_SPLIT  = {bp_rf_final['min_samples_split']}")
    print(f"      RF_MIN_SAMPLES_LEAF   = {bp_rf_final['min_samples_leaf']}")
    v_feat_rf = f"'{bp_rf_final['max_features']}'" if isinstance(bp_rf_final['max_features'], str) else bp_rf_final['max_features']
    print(f"      RF_MAX_FEATURES       = {v_feat_rf}")
    print(f"      RF_BOOTSTRAP          = {bp_rf_final['bootstrap']}\n")
    print("      # ── Ensemble Weight Ratios ──")
    print(f"      ENS_WEIGHT_GB         = {bw[0]:.2f}")
    print(f"      ENS_WEIGHT_RF         = {bw[1]:.2f}")
    print("\n" + "#" * 80 + "\n")


def secim_al():
    if not sys.stdin.isatty(): return set(range(len(ARA_DEGER_TESTLER)))

    print("\n  2647 Test List:")
    for i, s in enumerate(ARA_DEGER_TESTLER):
        print(f"  [{i:>2}] {s['tip']:<8}  Run{s['run']}  ({s['agirlik_g']}g)   \u2192 {s['daq']}")

    # Show run shorthand hints
    all_runs = sorted(set(s['run'] for s in ARA_DEGER_TESTLER))
    run_hint = "  |  ".join(f"'run{r}' = all RUN{r} files" for r in all_runs)
    print(f"\n  Shorthand : {run_hint}")
    print("  Enter the indices you want as TEST  (everything else \u2192 TRAIN)")
    print("  Examples  :  7            \u2192 index 7 is TEST,  rest TRAIN")
    print("               8,21,35      \u2192 those 3 are TEST, rest TRAIN")
    print("               run2         \u2192 all RUN2 files are TEST, rest TRAIN")
    print("               ENTER        \u2192 ALL files are TRAIN  (no test set)")
    raw = input("  TEST selection: ").strip().lower()

    all_idx = set(range(len(ARA_DEGER_TESTLER)))

    if not raw:
        return all_idx  # everything is training

    # 'runN' shorthand
    run_tokens = re.findall(r'run(\d+)', raw)
    if run_tokens:
        test_idx = set()
        for rn in run_tokens:
            wanted = int(rn)
            test_idx.update(i for i, s in enumerate(ARA_DEGER_TESTLER) if s['run'] == wanted)
        if test_idx:
            return all_idx - test_idx  # training = all minus test

    # Plain index list: "7"  or  "8,21,35"  or  "8 21 35"
    try:
        test_idx = set(
            int(x) for x in re.split(r'[,\s]+', raw) if x.strip()
        )
        return all_idx - test_idx  # training = all minus test
    except:
        return all_idx

def veri_kalitesi_raporu(tests, features, weights):
    """Data quality summary"""
    total = len(tests)
    valid = len([f for f in features if f is not None])
    failures = total - valid
    
    print("\n" + "="*120)
    print("📊 DATA QUALITY REPORT")
    print("="*120)
    print(f"  Total Tests                : {total}")
    print(f"  Successful                 : {valid} ({100*valid/total:.0f}%)")
    print(f"  Failed / Filtered          : {failures}")
    
    if failures > 0:
        print(f"\n  ⚠️  Issues detected in {failures} file(s) — check the error log!")
    
    if valid > 0:
        pass
    
    print("="*120 + "\n")

# Add to report (after data loading)
secilen_islem_modu, test_tipi = mod_secim_al()

if secilen_islem_modu == 2:
    hiperparametre_test_yap(test_tipi)
    sys.exit(0)

# If normal mode selected, only get index selection
egitim_indeksleri = secim_al()

# Read once for normal operation
features, weights = [], []
bar1 = ProgressBar(len(ARA_DEGER_TESTLER), desc=">> Data Loading")
for i, s in enumerate(ARA_DEGER_TESTLER):
    f, w = veri_ayikla_2647(s)
    features.append(f)
    weights.append(w)
    bar1.update()
bar1.close()

# ──── DATA QUALITY REPORT ────
veri_kalitesi_raporu(ARA_DEGER_TESTLER, features, weights)

# Filtering (Only KULLANILACAK_OZELLIKLER applied for model)
secilen_idx = [OZELLIK_ISIMLERI.index(col) for col in KULLANILACAK_OZELLIKLER]
filtered_features = [f[secilen_idx] if f is not None else None for f in features]

# Training
X_eg, y_eg = [], []
for idx in egitim_indeksleri:
    if filtered_features[idx] is not None:
        X_eg.append(filtered_features[idx])
        y_eg.append(weights[idx])

if len(X_eg) < 1:
    print("ERROR: Insufficient training data!")
    sys.exit()

# ── Synthetic Data Generation (less, cleaner noise) ───────────────────
X_orig = np.array(X_eg)
y_orig = np.array(y_eg)
n_feat = X_orig.shape[1]

rng = np.random.default_rng(42)

idx_syn    = rng.choice(len(X_orig), SENTETIK_ORNEK_SAYISI, replace=True)
noise_mask = 1 + rng.normal(0, SENTETIK_NOISE_X, (SENTETIK_ORNEK_SAYISI, n_feat))
X_syn      = X_orig[idx_syn] * noise_mask
y_syn      = y_orig[idx_syn] * (1 + rng.normal(0, SENTETIK_NOISE_Y, SENTETIK_ORNEK_SAYISI))

# Real data + synthetic data must be combined
X_train = np.vstack([X_orig, X_syn])
y_train = np.concatenate([y_orig, y_syn])

# ── 2-Model Ensemble ─────────────────────────────────────────────────────
print("\n  >> Training ensemble models...")
mod_gb  = GradientBoostingRegressor(
    n_estimators      = GB_N_ESTIMATORS,
    max_depth         = GB_MAX_DEPTH,
    learning_rate     = GB_LEARNING_RATE,
    subsample         = GB_SUBSAMPLE,
    min_samples_split = GB_MIN_SAMPLES_SPLIT,
    min_samples_leaf  = GB_MIN_SAMPLES_LEAF,
    max_features      = GB_MAX_FEATURES,
    random_state      = 42
).fit(X_train, y_train)

mod_rf  = RandomForestRegressor(
    n_estimators      = RF_N_ESTIMATORS,
    max_depth         = RF_MAX_DEPTH,
    min_samples_split = RF_MIN_SAMPLES_SPLIT,
    min_samples_leaf  = RF_MIN_SAMPLES_LEAF,
    max_features      = RF_MAX_FEATURES,
    bootstrap         = RF_BOOTSTRAP,
    random_state      = 42,
    n_jobs            = -1
).fit(X_train, y_train)

def ensemble_tahmin(X):
    p_gb    = mod_gb.predict(X)
    p_rf    = mod_rf.predict(X)
    return ENS_W_GB * p_gb + ENS_W_RF * p_rf

def ensemble_tahmin_with_ci(X, ci_method='tree_variance', n_samples=30):
    """
    Prediction + 95% Confidence Interval
    ci_method:
      - 'tree_variance': CI from individual tree predictions
      - 'bootstrap': CI via bootstrap resampling
    """
    if ci_method == 'tree_variance':
        # GB predictions
        gb_preds = []
        for tree_idx in range(min(n_samples, len(mod_gb.estimators_))):
            tree_pred = mod_gb.estimators_[tree_idx, 0].predict(X)
            gb_preds.append(tree_pred)
        gb_preds = np.array(gb_preds)  # (n_trees, n_samples)
        
        # RF predictions
        rf_preds = []
        for tree_idx in range(min(n_samples, len(mod_rf.estimators_))):
            tree_pred = mod_rf.estimators_[tree_idx].predict(X)
            rf_preds.append(tree_pred)
        rf_preds = np.array(rf_preds)  # (n_trees, n_samples)
        
        # Ensemble at each tree level
        ens_preds = ENS_W_GB * gb_preds + ENS_W_RF * rf_preds
        
        # Prediction = mean, CI = ±1.96×std
        pred_mean = ens_preds.mean(axis=0)
        pred_std  = ens_preds.std(axis=0)
        ci_lower  = pred_mean - 1.96 * pred_std
        ci_upper  = pred_mean + 1.96 * pred_std
        
        return pred_mean, ci_lower, ci_upper
    
    else:  # bootstrap
        pred_mean = ensemble_tahmin(X)
        rng = np.random.default_rng(42)
        n_bootstrap = n_samples
        bootstrap_preds = []
        
        for _ in range(n_bootstrap):
            indices = rng.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train mini model
            gb_boot = GradientBoostingRegressor(**{
                'n_estimators': min(50, GB_N_ESTIMATORS),
                'max_depth': GB_MAX_DEPTH,
                'learning_rate': GB_LEARNING_RATE,
                'random_state': 42
            }).fit(X_boot, y_boot)
            
            rf_boot = RandomForestRegressor(**{
                'n_estimators': min(50, RF_N_ESTIMATORS),
                'max_depth': RF_MAX_DEPTH,
                'random_state': 42,
                'n_jobs': -1
            }).fit(X_boot, y_boot)
            
            pred_boot = ENS_W_GB * gb_boot.predict(X) + ENS_W_RF * rf_boot.predict(X)
            bootstrap_preds.append(pred_boot)
        
        bootstrap_preds = np.array(bootstrap_preds)
        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)
        
        return pred_mean, ci_lower, ci_upper

joblib.dump({"gb": mod_gb, "rf": mod_rf},
            os.path.join(OUTPUT_DIR, "GB_2647_Zeka.pkl"))
print(f"  OK: Ensemble model saved (GB x{ENS_W_GB:.2f} + RF x{ENS_W_RF:.2f}).")

# ── Report ─────────────────────────────────────────────────────────────────
G, Y, RD, OR, B, R = "\033[92m", "\033[93m", "\033[91m", "\033[38;5;214m", "\033[1m", "\033[0m"
BLUE, PURPLE, BROWN, CYAN = "\033[94m", "\033[95m", "\033[38;5;130m", "\033[96m"

print("\n" + "="*140)
print(f"{B}  MODEL: Ensemble (GB x{ENS_W_GB:.2f} + RF x{ENS_W_RF:.2f})  |  STATUS: Completed{R}")
print(f"{B}{'TEST NAME':<30} | {'STATUS':<10} | {BROWN}{'ACTUAL(g)':<10}{R}{B} | {BLUE}{'ENS.PRED(g)':<10}{R}{B} | {CYAN}{'GB PRED(g) %60':<13}{R}{B} | {PURPLE}{'RF PRED(g) %40':<13}{R}{B} | {'ERR(g)':<8} | {'% ERR'}{R}")
print("-" * 140)

error_pcts = []
for i, s in enumerate(ARA_DEGER_TESTLER):
    if filtered_features[i] is None: continue
    X_row   = filtered_features[i].reshape(1, -1)
    pred_ens = ensemble_tahmin(X_row)[0]
    pred_gb  = mod_gb.predict(X_row)[0]
    pred_rf  = mod_rf.predict(X_row)[0]
    w_true   = weights[i]
    err  = pred_ens - w_true
    pct  = (err / w_true) * 100
    error_pcts.append(abs(pct))
    is_train = i in egitim_indeksleri

    durum_s = f"{G}TRAIN{R}" if is_train else f"{Y}TEST{R}"
    h_renk  = RD if abs(pct) > 2.0 else (OR if abs(pct) > 1.0 else G)

    test_adi = f"{s['tip']} Run{s['run']}"
    print(f"  {test_adi:<28} | {durum_s:<18} | {w_true:<10.0f} | {pred_ens:<10.0f} | {pred_gb:<13.0f} | {pred_rf:<13.0f} | {h_renk}{err:>+7.0f}g{R} | {h_renk}{pct:>+6.2f}%{R}")

print("=" * 140)
if error_pcts:
    print(f"\n  STATS >> Mean |% Error|: {np.mean(error_pcts):.2f}%   |   Max |% Error|: {np.max(error_pcts):.2f}%   |   Median: {np.median(error_pcts):.2f}%")


# ── LOO Cross-Validation (True Generalization Error) ──────────────────────
print("\n" + "="*120)
print(f"{B}📊 LEAVE-ONE-OUT CROSS-VALIDATION (True Generalization Error){R}")
print("="*120)

valid_all = [i for i, f in enumerate(filtered_features) if f is not None]
X_all = np.array([filtered_features[i] for i in valid_all])
y_all = np.array([weights[i] for i in valid_all])
loo_preds = np.zeros(len(X_all))

bar_loo = ProgressBar(len(X_all), desc=">> LOO Calculation")
for li in range(len(X_all)):
    X_tr_loo = np.delete(X_all, li, axis=0)
    y_tr_loo = np.delete(y_all, li, axis=0)
    
    # Add synthetic data
    rng_loo = np.random.default_rng(42)
    idx_s = rng_loo.choice(len(X_tr_loo), SENTETIK_ORNEK_SAYISI, replace=True)
    noise_m = 1 + rng_loo.normal(0, SENTETIK_NOISE_X, (SENTETIK_ORNEK_SAYISI, X_tr_loo.shape[1]))
    X_syn_l = X_tr_loo[idx_s] * noise_m
    y_syn_l = y_tr_loo[idx_s] * (1 + rng_loo.normal(0, SENTETIK_NOISE_Y, SENTETIK_ORNEK_SAYISI))
    X_trF = np.vstack([X_tr_loo, X_syn_l])
    y_trF = np.concatenate([y_tr_loo, y_syn_l])
    
    gb_l = GradientBoostingRegressor(
        n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE, subsample=GB_SUBSAMPLE,
        min_samples_split=GB_MIN_SAMPLES_SPLIT, min_samples_leaf=GB_MIN_SAMPLES_LEAF,
        max_features=GB_MAX_FEATURES, random_state=42
    ).fit(X_trF, y_trF)
    rf_l = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES, bootstrap=RF_BOOTSTRAP,
        random_state=42, n_jobs=-1
    ).fit(X_trF, y_trF)
    
    p_gb_l = gb_l.predict(X_all[li:li+1])[0]
    p_rf_l = rf_l.predict(X_all[li:li+1])[0]
    loo_preds[li] = ENS_W_GB * p_gb_l + ENS_W_RF * p_rf_l
    bar_loo.update()
bar_loo.close()

print(f"\n{B}{'TEST NAME':<30} | {'ACTUAL(g)':<10} | {'LOO PRED(g)':<11} | {'ERR(g)':<8} | {'% ERR'}{R}")
print("-" * 90)
loo_error_pcts = []
table2_export = []
for li, vi in enumerate(valid_all):
    s = ARA_DEGER_TESTLER[vi]
    w_true_l = y_all[li]
    pred_l   = loo_preds[li]
    err_l    = pred_l - w_true_l
    pct_l    = (err_l / w_true_l) * 100
    loo_error_pcts.append(abs(pct_l))
    h_renk = RD if abs(pct_l) > 2.0 else (OR if abs(pct_l) > 1.0 else G)
    test_adi = f"{s['tip']} Run{s['run']}"
    print(f"  {test_adi:<28} | {w_true_l:<10.0f} | {pred_l:<11.0f} | {h_renk}{err_l:>+7.0f}g{R} | {h_renk}{pct_l:>+6.2f}%{R}")
    
    table2_export.append({
        "Test Name": test_adi,
        "Actual (g)": round(w_true_l, 0),
        "LOO Pred (g)": round(pred_l, 0),
        "Error (g)": round(err_l, 0),
        "Error (%)": round(pct_l, 2)
    })

print("=" * 90)
if loo_error_pcts:
    print(f"\n  LOO STATS >> Mean |%Error|: {np.mean(loo_error_pcts):.2f}%   |   Max: {np.max(loo_error_pcts):.2f}%   |   Median: {np.median(loo_error_pcts):.2f}%")
    print(f"  (These values show the model's true performance against UNSEEN data)\n")

# ── Performance Bar Chart ──────────────────────────────────────────────────
idx_gecerli = [i for i in range(len(filtered_features)) if filtered_features[i] is not None]
tips       = [f"{ARA_DEGER_TESTLER[i]['tip']} R{ARA_DEGER_TESTLER[i]['run']}" for i in idx_gecerli]
y_true     = [weights[i] for i in idx_gecerli]
y_pred     = [ensemble_tahmin(filtered_features[i].reshape(1, -1))[0] for i in idx_gecerli]
is_train_l = [i in egitim_indeksleri for i in idx_gecerli]

fig, ax = plt.subplots(figsize=(max(12, len(tips) * 0.8), 7))
x = np.arange(len(tips))
bar_w = 0.4

# Color scheme
TRAIN_ACTUAL_COLOR = 'navy'
TRAIN_PRED_COLOR   = 'orange'
TEST_ACTUAL_COLOR  = '#2196F3'   # bright steel blue — clearly different from navy
TEST_PRED_COLOR    = '#C62828'   # deep crimson — clearly different from orange

for idx_b in range(len(tips)):
    if is_train_l[idx_b]:
        # TRAIN: actual (navy) left, prediction (orange) right
        ax.bar(x[idx_b] - bar_w/2, y_true[idx_b], bar_w, color=TRAIN_ACTUAL_COLOR)
        ax.bar(x[idx_b] + bar_w/2, y_pred[idx_b], bar_w, color=TRAIN_PRED_COLOR)
        pred_x = x[idx_b] + bar_w/2
    else:
        # TEST: actual (bright blue) left, prediction (crimson) right
        ax.bar(x[idx_b] - bar_w/2, y_true[idx_b], bar_w, color=TEST_ACTUAL_COLOR)
        ax.bar(x[idx_b] + bar_w/2, y_pred[idx_b], bar_w, color=TEST_PRED_COLOR)
        pred_x = x[idx_b] + bar_w/2

    # Deviation % label on top of prediction bar
    pct_dev   = (y_pred[idx_b] - y_true[idx_b]) / y_true[idx_b] * 100
    lbl_color = 'green' if abs(pct_dev) <= 2.0 else ('red' if pct_dev < 0 else '#8B008B')
    lbl_text  = f"{pct_dev:+.1f}%"
    ax.text(pred_x, y_pred[idx_b] + max(y_true) * 0.008, lbl_text,
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=lbl_color)


# X-axis: test name + TRAIN/TEST label below
x_labels = []
for idx_b in range(len(tips)):
    status = "TRAIN" if is_train_l[idx_b] else "TEST"
    x_labels.append(f"{tips[idx_b]}\n({status})")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
for i, tick_label in enumerate(ax.get_xticklabels()):
    tick_label.set_color('#008080' if is_train_l[i] else '#d35400')
    tick_label.set_fontweight('bold')
ax.set_ylabel("Weight (g)")

# Custom dot legend — colored circles matching bar colors
legend_entries = [
    (TRAIN_ACTUAL_COLOR, "Actual load (used in training)"),
    (TRAIN_PRED_COLOR,   "AI prediction (trained sample)"),
    (TEST_ACTUAL_COLOR,  "Actual load — model never saw this"),
    (TEST_PRED_COLOR,    "AI prediction on unseen test data"),
]
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor=c, markersize=9, label=lbl)
    for c, lbl in legend_entries
]
ax.legend(handles=legend_handles, loc='upper left', fontsize=8,
          framealpha=0.85, edgecolor='#cccccc')


plt.tight_layout()
fig.subplots_adjust(top=0.88)  # make room for the manual title

# ── Dynamic mixed-weight title ──────────────────────────────────────────────
train_runs_used = sorted(set(
    ARA_DEGER_TESTLER[i]['run'] for i in egitim_indeksleri
    if filtered_features[i] is not None
))
run_label = "Run " + ",".join(str(r) for r in train_runs_used)
prefix = "Load Estimation Performance Trained with "
suffix = " for 2647"
_fs = 12
_ty = 1.04  # y position in axes-transAxes coords

# t1 (normal) right-aligned at 0.5 → t2 (bold) left-aligned at 0.5 → suffix right after t2
t1 = ax.text(0.5, _ty, prefix, transform=ax.transAxes,
             ha='right', va='bottom', fontsize=_fs, fontweight='normal', color='black')
t2 = ax.text(0.5, _ty, run_label, transform=ax.transAxes,
             ha='left', va='bottom', fontsize=_fs, fontweight='bold', color='black')

# measure t2 width to position suffix precisely
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_ax_win  = ax.get_window_extent(_renderer)
_t2_win  = t2.get_window_extent(_renderer)
_suf_x   = (_t2_win.x1 - _ax_win.x0) / _ax_win.width

ax.text(_suf_x, _ty, suffix, transform=ax.transAxes,
        ha='left', va='bottom', fontsize=_fs, fontweight='normal', color='black')

plt.savefig(os.path.join(OUTPUT_DIR, "2647_Performans.png"), dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✅ Model and chart saved to: {OUTPUT_DIR}")