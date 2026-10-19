#!/usr/bin/env python3
"""Read-only, standard-library submission audit. Never loads models or training code.

This checks file identity and deterministic arithmetic on already-saved confusion
matrices. It is NOT the original model test suite, an AUC recalculation, a new
model evaluation, a clinical validation, or proof of GitHub deployment.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, statistics, sys, zipfile
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET

SCIENCE_SHA = 'd5e4a71ff6a9e485fe857792461387f27dc84d1d'
FREEZE_ID = 'c4f809a629d97e68936d190a6c824b18f6713fa206b3a658d21f52bd64de6f55'
MODEL_SHA = '8592824c3b716d672faa1089235d0a85dca5fc5057a7b5bcaf5571cd5fcfcf33'
DYNAMIC = {'reproducibility/CHECKSUMS.sha256','reproducibility/local_check.json',
           'reproducibility/unit_tests.log','reproducibility/cloud_verification.json'}

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def safe_path(root: Path, rel: str) -> Path:
    if not isinstance(rel,str) or not rel or '\\' in rel:
        raise ValueError('empty, non-string or backslash path')
    p=Path(rel)
    if p.is_absolute() or '..' in p.parts or re.match(r'^[A-Za-z]:',rel):
        raise ValueError('absolute/parent path refused')
    target=(root/p).resolve()
    if not target.is_relative_to(root.resolve()): raise ValueError('path escapes root')
    return target

def load_json(p: Path):
    def bad(v): raise ValueError('non-finite JSON token '+v)
    return json.loads(p.read_text(encoding='utf-8'), parse_constant=bad)

def cm_metrics(cm):
    if not isinstance(cm,list) or len(cm)!=9 or any(not isinstance(row,list) or len(row)!=9 for row in cm):
        raise ValueError('confusion matrix must be 9 x 9')
    if any(type(v) is not int or v<0 for row in cm for v in row):
        raise ValueError('confusion counts must be nonnegative integers, not bool')
    n=sum(map(sum,cm))
    if n<=0: raise ValueError('empty confusion matrix')
    correct=sum(cm[i][i] for i in range(9));f=[]
    for i in range(9):
        denom=sum(cm[i])+sum(row[i] for row in cm)
        f.append(2*cm[i][i]/denom if denom else 0.0)
    return {'n':n,'correct':correct,'accuracy':correct/n,'macro_f1':sum(f)/9,'per_class_f1':f}

def same(a,b,tol=1e-9):
    if isinstance(a,bool) or isinstance(b,bool): raise ValueError('boolean is not a metric')
    a=float(a);b=float(b)
    if not (math.isfinite(a) and math.isfinite(b)): raise ValueError('nonfinite metric')
    if abs(a-b)>tol: raise ValueError(f'metric mismatch {a} != {b} (tol={tol})')

def selection(rows):
    groups={g:[r for r in rows if r['group']==g] for g in 'ABCD'}
    if any(len(v)!=3 for v in groups.values()): raise ValueError('requires 3 runs per A-D')
    scores={g:statistics.mean(float(r['val_macro_f1']) for r in rr) for g,rr in groups.items()}
    top=max(scores.values());admissible=[g for g in 'ABCD' if top-scores[g]<=.005+1e-12]
    # Selection uses median of run medians; plot uses mean. Do not substitute.
    lat={g:statistics.median(float(r['cpu_median_ms']) for r in groups[g]) for g in admissible}
    g=min(lat,key=lat.get);run=max(groups[g],key=lambda r:float(r['val_macro_f1']))['id']
    return {'admissible':admissible,'group':g,'run_id':run,'cpu_statistic':'median of three run medians'}

def check_manifest(root):
    p=root/'reproducibility/CHECKSUMS.sha256';count=0;seen=set()
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        sha,rel=line.split('  ',1)
        if not re.fullmatch('[0-9a-f]{64}',sha): raise ValueError('invalid SHA field')
        if rel in seen: raise ValueError('duplicate manifest entry '+rel)
        seen.add(rel);target=safe_path(root,rel)
        if not target.is_file(): raise ValueError('missing manifest file '+rel)
        if digest(target)!=sha: raise ValueError('SHA mismatch '+rel)
        count+=1
    if count<20: raise ValueError('manifest incomplete')
    return {'files_verified':count,'coverage':'enumerated immutable files; dynamic logs excluded'}

def audit(root:Path,require_demo=False):
    checks=[];errors=[];warnings=[]
    def run(name,fn):
        try: detail=fn();checks.append({'name':name,'status':'passed','detail':detail})
        except Exception as exc:errors.append({'name':name,'error':str(exc)});checks.append({'name':name,'status':'failed'})
    run('file_manifest',lambda:check_manifest(root))
    def source():
        xs=load_json(root/'evidence/source_provenance.json')
        for x in xs:
            p=safe_path(root,x['destination'])
            if digest(p)!=x['sha256']:raise ValueError('source copy changed '+x['destination'])
        return {'source_copies':len(xs),'remote_redownload':False}
    run('historical_source_identity',source)
    rows=list(csv.DictReader((root/'evidence/all_runs.csv').open(encoding='utf-8')))
    def formal():
        expected={f'{g}_seed{s}' for g in 'ABCDE' for s in [42,2026,3407]}
        if len(rows)!=15 or {r['id'] for r in rows}!=expected:raise ValueError('15-run identities mismatch')
        results=[]
        for r in rows:
            j=load_json(root/'evidence/formal'/r['id']/'test_metrics.json');m=cm_metrics(j['confusion_matrix'])
            if m['n']!=7180 or j.get('n')!=7180:raise ValueError('test sample count mismatch '+r['id'])
            for k in ['accuracy','macro_f1']:same(m[k],j[k]);same(m[k],r[k])
            results.append({'run':r['id'],'n':m['n'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
        return {'count':15,'saved_confusion_matrices':results,'model_inference':False,'auc_recomputed':False}
    run('formal_saved_confusion_arithmetic',formal)
    def select_check():
        x=selection(rows);f=load_json(root/'evidence/frozen.json')
        if x['admissible']!=['A','B','D'] or x['run_id']!='D_seed2026':raise ValueError('selection replay mismatch')
        if f['selection']['run_id']!=x['run_id'] or f['freeze_id']!=FREEZE_ID:raise ValueError('freeze metadata changed')
        return x
    run('pretest_selection_replay',select_check)
    def quant():
        q=load_json(root/'evidence/quantization_result.json');m=cm_metrics(q['metrics']['confusion_matrix'])
        if m['n']!=10004:raise ValueError('quant val count mismatch')
        for k in ['accuracy','macro_f1']:same(m[k],q['metrics'][k])
        same(m['macro_f1'],.9956677882029596)
        return {'val_samples':m['n'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'new_test_inference':False}
    run('quant_saved_confusion_arithmetic',quant)
    def metadata():
        m=load_json(root/'demo/portable_demo/model/metadata.json')
        if m['model']['sha256']!=MODEL_SHA:raise ValueError('model identity mismatch')
        sp=load_json(root/'demo/portable_demo/samples/provenance.json')
        if len(sp['samples'])!=10:raise ValueError('fixed samples mismatch')
        return {'model_sha':MODEL_SHA,'samples_in_provenance':10,'weights_loaded':False}
    run('demo_metadata_only',metadata)
    required=load_json(root/'demo/required_runtime_assets.json')['files'];missing=[]
    def assets():
        n=0
        for x in required:
            p=safe_path(root,x['path'])
            if not p.is_file():missing.append(x['path']);continue
            if digest(p)!=x['sha256']:raise ValueError('runtime asset SHA mismatch '+x['path'])
            if 'bytes' in x and p.stat().st_size!=x['bytes']:raise ValueError('runtime bytes mismatch')
            n+=1
        if missing and require_demo:raise ValueError('missing demo assets: '+', '.join(missing))
        return {'verified':n,'missing':missing,'inference_executed':False}
    run('demo_asset_presence',assets)
    if missing:warnings.append({'name':'demo_not_self_contained','missing_count':len(missing),'message':'copy archived FP32 and fixed PNGs; basic check does not establish runnable demo'})
    def docs():
        fs=list((root/'reports').glob('*.docx'))+list((root/'presentation').glob('*.pptx'));counts={}
        if len(list((root/'reports').glob('*.docx')))!=4:raise ValueError('expected 4 reports')
        for p in fs:
            with zipfile.ZipFile(p) as z:
                if z.testzip():raise ValueError('bad zip member '+p.name)
                if p.suffix=='.pptx':
                    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
                    ids=ET.fromstring(z.read('ppt/presentation.xml')).find('p:sldIdLst',ns)
                    n=len(ids)
                    names=[x for x in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',x)]
                    hidden=sum(ET.fromstring(z.read(x)).get('show')=='0' for x in names)
                    notes=[x for x in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',x)]
                    if (n,hidden,len(notes))!=(30,4,30):raise ValueError('PPT page/hidden/notes mismatch')
                    counts[p.name]={'slides':n,'hidden':hidden,'notes':len(notes)}
                else:
                    if 'word/document.xml' not in z.namelist():raise ValueError('missing Word XML')
                    ET.fromstring(z.read('word/document.xml'));counts[p.name]={'zip_and_xml':'passed'}
        return counts
    run('docx_pptx_package_structure',docs)
    def links():
        # New authored Markdown only; historical immutable paths are intentionally relative to original repository.
        candidates=[root/'README.md',root/'00_教师阅读入口.md',root/'SOURCE_PROVENANCE.md',root/'SUBMISSION_MANIFEST.md',root/'THIRD_PARTY_NOTICES.md']
        for sub in ['ai_workflow','technical','presentation','demo','reproducibility','codex_handoff']:
            candidates += [p for p in (root/sub).glob('*.md')]
        n=0
        for p in candidates:
            if not p.is_file():raise ValueError('missing authored entry '+str(p))
            for raw in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                target=raw.split('#',1)[0]
                if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
                q=(p.parent/unquote(target)).resolve()
                if not q.is_relative_to(root.resolve()) or not q.exists():raise ValueError(f'broken relative link {p.relative_to(root)}: {raw}')
                n+=1
        return {'links_checked':n,'files':len(candidates),'historical_embedded_source_links_excluded':True,'online_access_tested':False}
    run('authored_relative_links',links)
    return {'status':'passed' if not errors else 'failed','scope':'local submission file/arithmetic checks only',
            'checks':checks,'errors':errors,'warnings':warnings,'new_training_runs':0,'new_model_inference':0,
            'official_test_predictions_generated':0,'cloud_ci_executed':False,'source_repository_written':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--require-demo-assets',action='store_true');args=parser.parse_args()
    try:result=audit(args.root.resolve(),args.require_demo_assets)
    except Exception as exc:result={'status':'failed','fatal_error':str(exc),'model_inference':False}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
