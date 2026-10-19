# coding: utf-8
"""Local-only FP32 demonstration using standard-library HTTP UI."""
import argparse,base64,json
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from predict import Predictor,load_samples,DemoError,ROOT,LABELS_ZH
HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PathMNIST FP32 离线演示</title><style>body{font:17px system-ui;max-width:850px;margin:40px auto;padding:20px;color:#183347;background:#f6f8fa}button,select,input{font:inherit;padding:9px;margin:8px 4px}main{background:white;padding:28px;border-radius:12px}img{width:200px;image-rendering:auto}li{margin:12px 0}#error{color:#b00020}small{color:#576574}</style><main><h1>病理组织图像块分类</h1><p>冻结模型 D_seed2026 · FP32 · CPU · 128 × 128</p><p>九类组织分类，结果不用于患者诊断。未提供自动域外图像检测。</p><select id="samples"></select><button id="sampleButton" onclick="sample()">预测内置样例</button><hr><input id="upload" type="file" accept="image/png,image/jpeg"><button id="uploadButton" onclick="upload()">预测上传图片</button><p><small>PNG/JPEG，最大20MB、1600万像素；灰度与RGBA转换为RGB。固定不一致样例未经病理专家复核。</small></p><img id="image" hidden><div id="given"></div><p id="error" role="alert"></p><h2>现场推理结果</h2><div id="result">选择一张图片开始</div></main><script>
let items=[];async function init(){items=await(await fetch('/samples')).json();let s=document.getElementById('samples');for(let i of items){let o=document.createElement('option');o.value=i.file;o.textContent=i.sample_id+' · 给定标签 '+i.label_zh+(i.reference_prediction!==i.given_label?' · 固定不一致样例':'');s.appendChild(o)}}
async function run(payload){document.getElementById('error').textContent='';document.getElementById('result').textContent='正在执行FP32模型…';try{let r=await(await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();if(r.status!=='passed')throw Error(r.error);let box=document.getElementById('result');box.textContent='预测：'+r.predicted_label_zh;let list=document.createElement('ol');for(let x of r.top3){let li=document.createElement('li');li.textContent=x.label_zh+' '+(x.probability*100).toFixed(3)+'%';list.appendChild(li)}box.appendChild(list);let t=document.createElement('p');t.textContent='CPU前向 '+r.timing_ms.forward_cpu.toFixed(2)+'ms；含解码处理 '+r.timing_ms.decode_preprocess_forward_probability.toFixed(2)+'ms';box.appendChild(t)}catch(e){document.getElementById('error').textContent=e.message;document.getElementById('result').textContent='本次未输出预测'}}
function sample(){let f=document.getElementById('samples').value;let i=items.find(x=>x.file===f);document.getElementById('given').textContent='数据集给定标签：'+i.label_zh+(i.reference_prediction!==i.given_label?'。此样例与给定标签不一致，未做病理专家复核。':'');let img=document.getElementById('image');img.src='/image/'+f;img.hidden=false;run({sample:f})}
async function upload(){let f=document.getElementById('upload').files[0];if(!f)return;document.getElementById('given').textContent='上传图片，没有真实标签';let d=await new Promise((resolve,reject)=>{let r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(f)});let img=document.getElementById('image');img.src=d;img.hidden=false;run({data:d.split(',')[1]})}init();</script></html>'''
def serve(port):
 predictor=Predictor();manifest=load_samples();samples={s['file']:s for s in manifest['samples']}
 class Handler(BaseHTTPRequestHandler):
  def send(self,data,kind='application/json',status=200):
   if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
   self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   if self.path=='/':return self.send(HTML.encode(),'text/html; charset=utf-8')
   if self.path=='/samples':return self.send([dict(s,label_zh=LABELS_ZH[s['given_label']]) for s in samples.values()])
   if self.path.startswith('/image/') and self.path[7:] in samples:return self.send((ROOT/'samples'/self.path[7:]).read_bytes(),'image/png')
   self.send({'status':'failed','error':'路径不存在'},status=404)
  def do_POST(self):
   if self.path!='/predict':return self.send({'status':'failed','error':'路径不存在'},status=404)
   try:
    n=int(self.headers.get('Content-Length','0'))
    if n<=0 or n>28_000_000:raise DemoError('上传大小不合法')
    v=json.loads(self.rfile.read(n))
    if 'sample' in v:
     if v['sample'] not in samples:raise DemoError('固定样例不存在')
     load_samples();data=(ROOT/'samples'/v['sample']).read_bytes()
    else:data=base64.b64decode(v['data'],validate=True)
    self.send(predictor.predict(data))
   except Exception as e:self.send({'status':'failed','error':str(e)},status=400)
 print(f'Local FP32 UI: http://127.0.0.1:{port}',flush=True);ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8767);serve(p.parse_args().port)
