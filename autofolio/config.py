from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
DATA = ROOT / 'research_data' if (ROOT / 'research_data').exists() else ROOT / 'data'
RUNS = ROOT / 'runs'
DB = DATA / 'autofolio.sqlite3'
ARC = Path(os.environ.get('AUTOFOLIO_ARCTRADE', ROOT / 'integrations/timefolio' if (ROOT / 'integrations/timefolio').exists() else HOME / 'projects/ArcTrade'))
STUDY = ARC / '_workspace/timefolio_noncnn_4y_20261006'
REPORT_ROOTS = [RUNS / 'experiments', RUNS / 'market_experiments']
PORT = int(os.environ.get('AUTOFOLIO_PORT', '8997'))
