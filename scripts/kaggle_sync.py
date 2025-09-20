import os, yaml, subprocess, shutil, zipfile
from pathlib import Path

CAT = Path('conf/kaggle_catalog.yaml')
ROOT = Path('.')

def run(*args, **kw):
print('>>>', ' '.join(map(str,args)))
return subprocess.check_call(list(map(str,args)), **kw)

cfg = yaml.safe_load(open(CAT))
out_root = ROOT / cfg.get('root','data/raw/kaggle')
out_root.mkdir(parents=True, exist_ok=True)

for slug in cfg['slugs']:
slug_sane = slug.replace('/', '__')
dst = out_root / slug_sane
dst.mkdir(parents=True, exist_ok=True)
zip_path = dst / 'dump.zip'
if not zip_path.exists():
run('kaggle','datasets','download','-d', slug, '-p', str(dst), '--force')
# extract
for f in dst.glob('*.zip'):
with zipfile.ZipFile(f) as z:
z.extractall(dst)
print('ok:', slug)
