# coding: utf-8
"""Independent frozen FP32 CPU predictor. No training or test-set interfaces."""
from pathlib import Path
import argparse,hashlib,io,json,time,warnings
import numpy as np
from PIL import Image,UnidentifiedImageError
import onnxruntime as ort
ROOT=Path(__file__).resolve().parent
LABELS=['adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium']
LABELS_ZH=['脂肪','背景','碎屑','淋巴细胞','黏液','平滑肌','正常结肠黏膜','癌相关间质','结直肠腺癌上皮']
FREEZE='c4f809a629d97e68936d190a6c824b18f6713fa206b3a658d21f52bd64de6f55'
CHECKPOINT='a6bce45c31d039f1fd7375b9ee82f9b91f02ca962544fd8bfebdd208c781b0cc'
MODEL='8592824c3b716d672faa1089235d0a85dca5fc5057a7b5bcaf5571cd5fcfcf33'
MEAN=np.array([.485,.456,.406],np.float32).reshape(3,1,1)
STD=np.array([.229,.224,.225],np.float32).reshape(3,1,1)
MAX_PIXELS=16_000_000;MAX_BYTES=20_000_000
class DemoError(ValueError):pass
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def safe_path(base,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in name:raise DemoError('拒绝模型路径越界')
 result=(base/p).resolve()
 if not result.is_relative_to(base.resolve()):raise DemoError('拒绝软链接路径越界')
 return result

def validate_metadata(base):
 base=Path(base);m=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
 required={'schema_version':2,'variant':'fp32','selected_run':'D_seed2026','freeze_id':FREEZE,'checkpoint_sha256':CHECKPOINT,'labels':LABELS,'labels_zh':LABELS_ZH,'size':128,'dtype':'float32','layout':'NCHW','channels':'RGB','resize':'PIL.BILINEAR','scale_divisor':255,'exif_transpose':False}
 for k,v in required.items():
  if m.get(k)!=v:raise DemoError('模型元数据不匹配: '+k)
 for k,v in [('mean',MEAN.reshape(-1)),('std',STD.reshape(-1))]:
  a=np.asarray(m.get(k),dtype=np.float64)
  if a.shape!=(3,) or not np.isfinite(a).all() or not np.allclose(a,v,atol=1e-7,rtol=0):raise DemoError('预处理参数不匹配: '+k)
 model=safe_path(base,m['model']['path'])
 if m['model'].get('sha256')!=MODEL or sha(model)!=MODEL:raise DemoError('模型SHA256校验失败')
 if model.stat().st_size!=m['model'].get('bytes'):raise DemoError('模型大小校验失败')
 return m,model

def preprocess(data):
 if len(data)>MAX_BYTES:raise DemoError('图片文件超过20MB上限')
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('error',Image.DecompressionBombWarning)
   with Image.open(io.BytesIO(data)) as image:
    if image.width*image.height>MAX_PIXELS:raise DemoError('图片超过1600万像素上限')
    mode=image.mode
    if image.format not in ('PNG','JPEG'):raise DemoError('仅支持PNG或JPEG')
    image.load();image=image.convert('RGB').resize((128,128),Image.Resampling.BILINEAR)
    x=np.asarray(image,np.float32).transpose(2,0,1)/np.float32(255)
 except DemoError:raise
 except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning) as e:raise DemoError('图片损坏或无法解码') from e
 x=np.ascontiguousarray(((x-MEAN)/STD)[None],dtype=np.float32)
 if x.shape!=(1,3,128,128) or not np.isfinite(x).all():raise DemoError('预处理输出无效')
 return x,mode

def probabilities(logits):
 z=np.asarray(logits)
 if z.shape!=(1,9) or not np.isfinite(z).all():raise DemoError('模型输出必须是有限的1×9 logits')
 z=z.astype(np.float64);e=np.exp(z-z.max(axis=1,keepdims=True));p=e/e.sum(axis=1,keepdims=True)
 if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-9):raise DemoError('模型概率无效')
 return p[0]
class Predictor:
 def __init__(self,model_dir=None):
  self.metadata,path=validate_metadata(Path(model_dir) if model_dir else ROOT/'model')
  t=time.perf_counter();o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1
  self.session=ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
  self.session_ms=(time.perf_counter()-t)*1000
  ins=self.session.get_inputs();outs=self.session.get_outputs()
  if len(ins)!=1 or ins[0].type!='tensor(float)' or ins[0].shape[1:]!=[3,128,128] or len(outs)!=1 or outs[0].shape[-1]!=9:raise DemoError('模型I/O形状不匹配')
  self.name=ins[0].name;self.calls=0
 def predict(self,data):
  t=time.perf_counter();x,mode=preprocess(data);p0=time.perf_counter();logits=self.session.run(None,{self.name:x})[0];forward=(time.perf_counter()-p0)*1000;p=probabilities(logits);order=np.argsort(-p,kind='stable')
  self.calls+=1
  return {'status':'passed','variant':'fp32','selected_run':'D_seed2026','model_sha256':MODEL,'input_mode':mode,'predicted_label':int(order[0]),'predicted_label_zh':LABELS_ZH[order[0]],'probabilities':p.tolist(),'top3':[{'label':int(i),'label_zh':LABELS_ZH[i],'probability':float(p[i])} for i in order[:3]],'timing_ms':{'session_creation':self.session_ms,'forward_cpu':forward,'decode_preprocess_forward_probability':(time.perf_counter()-t)*1000},'scope':'病理图像块组织分类，不是患者诊断；没有自动域外检测'}

def load_samples():
 p=ROOT/'samples/provenance.json';m=json.loads(p.read_text(encoding='utf-8'))
 for s in m['samples']:
  if s['model_sha256']!=MODEL or sha(safe_path(ROOT/'samples',s['file']))!=s['sha256']:raise DemoError('固定样例哈希校验失败')
 return m

def write_new(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('image',nargs='?',type=Path,default=ROOT/'samples/val_0.png');parser.add_argument('--output',type=Path);a=parser.parse_args()
 try:
  if a.output and a.output.exists():raise DemoError('输出文件已存在，拒绝覆盖')
  r=Predictor().predict(a.image.read_bytes())
  if a.output:write_new(a.output,r)
  print(json.dumps(r,ensure_ascii=False,indent=2));return 0
 except (DemoError,OSError,KeyError,json.JSONDecodeError) as e:print(json.dumps({'status':'failed','error':str(e)},ensure_ascii=False));return 2
if __name__=='__main__':raise SystemExit(main())
