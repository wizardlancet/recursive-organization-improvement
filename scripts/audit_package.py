from pathlib import Path
import sys,json,re,hashlib,csv
ROOT=Path(__file__).resolve().parents[1]
if (ROOT.parent/'.paper_runtime').exists():sys.path.insert(0,str(ROOT.parent/'.paper_runtime'))
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
from source_graph import sources
alltex='\n'.join(p.read_text(encoding='utf-8') for p in sources())
bib=(ROOT/'bibliography.bib').read_text(encoding='utf-8')
keys=re.findall(r'@\w+\{([^,]+),',bib)
cites=set(k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',alltex) for k in block.split(','))
log=(ROOT/'main.log').read_text(encoding='utf-8',errors='replace')
pdf=PdfReader(ROOT/'main.pdf');texts=[p.extract_text() or '' for p in pdf.pages]
report={'pages':len(pdf.pages),'bib_entries':len(keys),'unique_cited_entries':len(cites),'missing_citations':sorted(cites-set(keys)),'uncited_entries':sorted(set(keys)-cites),'duplicate_bib_keys':len(keys)-len(set(keys)),'figures':len(re.findall(r'\\begin\{figure\}',alltex)),'tables':len(re.findall(r'\\begin\{table\}',alltex)),'appendix_sections':sum(len(re.findall(r'^\\section\{',p.read_text(encoding='utf-8'),re.M)) for p in (ROOT/'sections').glob('appendices*.tex')),'overfull_boxes':re.findall(r'Overfull[^\n]*',log),'missing_glyphs':re.findall(r'Missing character:[^\n]*',log),'unresolved_references':re.findall(r'[^\n]*(?:undefined|multiply defined)[^\n]*',log),'extracted_word_count_including_refs':sum(len(t.split()) for t in texts),'pdf_sha256':hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest()}
report['appendix_sections']=len(re.findall(r'^\\section\{','\n'.join((ROOT/'sections'/name).read_text(encoding='utf-8') for name in ['appendices.tex','learning_protocol.tex']),re.M))
(ROOT/'qa/text_extraction.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts)),encoding='utf-8')
(ROOT/'qa/audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
doc=pdfium.PdfDocument(str(ROOT/'main.pdf'))
thumbs=[]
for i in range(len(doc)):
    im=doc[i].render(scale=1.5).to_pil().convert('RGB');im.save(ROOT/'qa'/f'page-{i+1:02}.png')
    t=ImageOps.contain(im,(500,650));out=Image.new('RGB',(520,690),'#e7ebef');out.paste(t,((520-t.width)//2,25));ImageDraw.Draw(out).text((12,665),f'Page {i+1}',fill='black');thumbs.append(out)
for start in range(0,len(thumbs),6):
    subset=thumbs[start:start+6];contact=Image.new('RGB',(1560,1380),'#dce2e8')
    for j,t in enumerate(subset):contact.paste(t,((j%3)*520,(j//3)*690))
    contact.save(ROOT/'qa'/f'contact-{start//6+1:02}.png')
print(json.dumps(report,indent=2))
