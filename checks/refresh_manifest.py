#!/usr/bin/env python3
"""Explicitly refresh the NEW submission manifest after reviewed changes only."""
from pathlib import Path
import argparse,hashlib
EXCLUDE={'reproducibility/CHECKSUMS.sha256','reproducibility/local_check.json','reproducibility/unit_tests.log','reproducibility/cloud_verification.json'}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--write',action='store_true');a=p.parse_args();r=a.root.resolve()
if not (r/'codex_handoff/HANDOFF.md').is_file():raise SystemExit('Not the new submission root; refused')
lines=[]
for f in sorted(r.rglob('*')):
 if not f.is_file():continue
 rel=f.relative_to(r).as_posix()
 if f.is_symlink():raise SystemExit('Symlink refused: '+rel)
 if rel in EXCLUDE or any(x in f.relative_to(r).parts for x in ['.git','.venv','__pycache__']):continue
 if f.suffix=='.pyc' or f.name=='.DS_Store':continue
 lines.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+rel)
text='\n'.join(lines)+'\n'
if a.write:(r/'reproducibility/CHECKSUMS.sha256').write_text(text,encoding='utf-8');print(f'wrote {len(lines)} entries')
else:print(text,end='')
