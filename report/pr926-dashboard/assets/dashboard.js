'use strict';

const $ = id => document.getElementById(id);
const DATA = window.PR926;
window.PR926_TOKENS = {};
const BLUE = '#235d99', ORANGE = '#b0521e', GRAY = '#87959b';
const languages = {ja:'Japanese', ko:'Korean', zh_cn:'Chinese'};
const representations = {original:'Original text', ipa:'IPA'};
const tokenizers = {char:'Char', char_bpe:'CharBPE'};
const colors = [BLUE, ORANGE];
const lightColors = ['#a1bbd5','#ddb4a1'];
const metrics = [
  ['Training occurrence','Occurrence'], ['Validation loss','Val loss'], ['Sampled training loss','Train loss'],
  ['Token-vector L2 norm','L2 norm'], ['Minimum pairwise angle (°)','Min angle'],
  ['Target probability','Target prob.'], ['Target rank','Target rank'], ['Left probability','Left prob.'],
];
const budgetLabels = {'500':'500 updates','1000':'1,000 updates',final:'About 3 corpus epochs'};
const order = ['ja_original','ja_ipa','ko_original','ko_ipa','zh_cn_original','zh_cn_ipa'];
const runs = new Map(DATA.runs.map(r => [r.id,r]));
const pairRuns = key => [runs.get(key+'_char'),runs.get(key+'_char_bpe')];
const pairName = key => {
  const run = pairRuns(key)[0];
  return languages[run.language]+' · '+(run.representation==='ipa'?'IPA':'original');
};
const runName = run => languages[run.language]+' · '+representations[run.representation]+' · '+tokenizers[run.tokenizer];
const num = (n,d=0) => Number.isFinite(n) ? n.toLocaleString('en-US',{maximumFractionDigits:d}) : '—';
const fixed = (n,d=4) => Number.isFinite(n) ? n.toFixed(d) : '—';
const escapeHtml = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const valueAt = (run, budget) => budget==='final' ? run.finalBpb : run.history.find(p => p.iteration===Number(budget)).bpb;
const isLog = () => $('log-occurrence').checked;
const displayValue = (value,metric) => value!==null && metric===0 && isLog() ? Math.log1p(value) : value;
const metricName = metric => metric===0 && isLog() ? metrics[0][0]+' · ln(1 + count)' : metrics[metric][0];

const plotConfig = {responsive:true,displayModeBar:false,scrollZoom:false};
const axis = {gridcolor:'#e7ece9',linecolor:'#b7c6c1',zerolinecolor:'#a2b4ac',automargin:true,tickfont:{size:11},fixedrange:false};
const base = {
  autosize:true, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  font:{family:'Arial, sans-serif',size:12,color:'#263b37'},
  margin:{l:64,r:22,t:20,b:60}, colorway:colors,
  legend:{orientation:'h',x:0,y:-.22}, hoverlabel:{namelength:-1},
};
const exportInfo = new Map();
const exportButton = id => document.querySelector('[data-export="'+id+'"]');
let statusTimer;
function status(message) {
  $('status').textContent = message;
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => { $('status').textContent = ''; }, 5200);
}
function select(id, items, preferred) {
  const target = $(id);
  target.replaceChildren(...items.map(([value,label]) => new Option(label,value)));
  if (items.some(([value])=>String(value)===String(preferred))) target.value = preferred;
}
function saveBlob(blob,name) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
function pngBlob(canvas) {
  return new Promise((resolve,reject) => canvas.toBlob(
    blob => blob ? resolve(blob) : reject(new Error('Unable to encode the chart.')),
    'image/png'
  ));
}
async function plot(id,traces,layout,info) {
  const button = exportButton(id);
  button.disabled = true;
  await Plotly.react($(id),traces,{...base,...layout},plotConfig);
  exportInfo.set(id,info);
  button.disabled = false;
}
async function savePlot(id) {
  const button = exportButton(id);
  const info = exportInfo.get(id);
  if (!info) return;
  button.disabled = true;
  button.textContent = 'Saving…';
  try {
    const chart = $(id);
    const width = Math.round(chart.getBoundingClientRect().width);
    const height = chart._fullLayout.height;
    const scale = 2;
    const uri = await Plotly.toImage(chart,{format:'png',width,height,scale});
    const img = new Image();
    img.src = uri;
    await img.decode();
    const canvas = document.createElement('canvas');
    const header = 78;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight + header * scale;
    const ctx = canvas.getContext('2d');
    ctx.scale(scale,scale);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,width,height+header);
    ctx.fillStyle = '#172a28';
    ctx.font = 'bold 19px Arial';
    ctx.fillText(info.title,18,27,width-36);
    ctx.fillStyle = '#526460';
    ctx.font = '12px Arial';
    ctx.fillText(info.subtitle,18,49,width-36);
    ctx.strokeStyle = '#d8e1da';
    ctx.beginPath(); ctx.moveTo(18,65); ctx.lineTo(width-18,65); ctx.stroke();
    ctx.drawImage(img,0,header,width,height);
    saveBlob(await pngBlob(canvas),'pr926-'+info.filename+'.png');
  } catch (error) {
    console.error(error);
    status('The chart could not be saved. Please try again.');
  } finally {
    button.disabled = false;
    button.textContent = 'Save PNG';
  }
}

function renderAdvantage() {
  const budgets = ['500','1000','final'];
  const labels = order.map(pairName);
  const cells = order.map(key => budgets.map(budget => {
    const [char,bpe] = pairRuns(key).map(run => valueAt(run,budget));
    const winner = char < bpe ? 'Char' : 'CharBPE';
    const improvement = 100*(1-Math.min(char,bpe)/Math.max(char,bpe));
    return {char,bpe,winner,improvement};
  }));
  return plot('advantage',[{
    type:'heatmap', x:['500 updates','1,000 updates','~3 epochs'], y:labels, xgap:5, ygap:5,
    z:cells.map(row=>row.map(c=>c.winner==='Char'?c.improvement:-c.improvement)),
    text:cells.map(row=>row.map(c=>fixed(c.improvement,1)+'%')),
    texttemplate:'%{text}', textfont:{size:13,color:'#172a28'},
    zmin:-45,zmax:45,colorscale:[[0,'#f6b98c'],[.495,'#fff0e3'],[.505,'#eaf2fa'],[1,'#9ac1e4']],showscale:false,
    customdata:cells.map(row=>row.map(c=>[c.winner,c.improvement,c.char,c.bpe])),
    hovertemplate:'%{y}<br>%{x}<br>%{customdata[0]} is %{customdata[1]:.2f}% lower<br>Char %{customdata[2]:.4f} · CharBPE %{customdata[3]:.4f}<extra></extra>',
  }],{
    height:378,margin:{l:119,r:12,t:36,b:48},
    xaxis:{side:'top',fixedrange:true,tickfont:{size:11}},
    yaxis:{autorange:'reversed',fixedrange:true,tickfont:{size:11}},
    annotations:[{xref:'paper',yref:'paper',x:0,y:-.125,xanchor:'left',showarrow:false,
      text:'<span style="color:'+BLUE+'">Blue · Char</span> &nbsp;&nbsp; <span style="color:'+ORANGE+'">Orange · CharBPE</span>'}],
  },{title:'BPB advantage by budget',subtitle:'Percent lower than the other tokenizer. Blue favors Char. Orange favors CharBPE.',filename:'bpb-advantage'});
}

function renderAbsolute() {
  const budget = $('budget').value;
  const traces = ['char','char_bpe'].map((tokenizer,index)=>({
    type:'bar',orientation:'h',name:tokenizers[tokenizer],y:order.map(pairName),
    x:order.map(key=>valueAt(pairRuns(key)[index],budget)),
    marker:{color:colors[index]},
    hovertemplate:'%{y}<br>Validation BPB %{x:.5f}<extra>%{fullData.name}</extra>',
  }));
  return plot('absolute',traces,{
    height:327,barmode:'group',margin:{l:118,r:12,t:8,b:60},
    xaxis:{...axis,title:{text:'Validation BPB'},rangemode:'tozero'},
    yaxis:{autorange:'reversed',tickfont:{size:11}},
    legend:{...base.legend,y:-.23},bargap:.30,
  },{title:'Validation bits per UTF-8 byte',subtitle:budgetLabels[budget]+' · lower is better',filename:'bpb-'+budget});
}

function renderHistory() {
  const key = $('curve-language').value+'_'+$('curve-rep').value;
  const xKey = $('curve-axis').value;
  const xTitle = xKey==='iteration'?'Optimizer updates':'Corpus epochs';
  const selected = pairRuns(key);
  const ratio = selected[0].trainTokens / selected[1].trainTokens;
  const title = pairName(key);
  $('curve-caption').textContent = title+' · Char / CharBPE corpus token ratio '+fixed(ratio,2)+'×';
  const traces = selected.flatMap((run,i)=>{
    const points = run.history.filter(point=>point.iteration>0);
    const last = points.at(-1);
    const endX = xKey==='iteration'?run.endpoint:run.finalEpoch;
    const name = tokenizers[run.tokenizer];
    const symbol = i?'diamond':'circle';
    const group = {legendgroup:name,line:{color:colors[i],width:2.6},name};
    return [{
      ...group,type:'scatter',mode:'lines+markers',x:points.map(p=>p[xKey]),y:points.map(p=>p.bpb),
      customdata:points.map(p=>[p.iteration,p.epochs]),
      marker:{color:colors[i],size:7,symbol},
      hovertemplate:'Step %{customdata[0]:,d}<br>%{customdata[1]:.3f} epochs<br>BPB %{y:.5f}<extra>'+name+'</extra>',
    },{
      ...group,type:'scatter',mode:'lines+markers',x:[last[xKey],endX],y:[last.bpb,run.finalBpb],
      line:{...group.line,dash:'dot'},marker:{color:colors[i],size:[0,14],symbol:['circle','star']},
      showlegend:false,hoverinfo:'skip',
    },{
      ...group,type:'scatter',mode:'markers',x:[endX],y:[run.finalBpb],
      marker:{color:colors[i],size:14,symbol:'star'},showlegend:false,
      hovertemplate:'Resumed endpoint · '+num(run.endpoint)+' updates<br>'+fixed(run.finalEpoch,4)+' epochs<br>BPB %{y:.5f}<extra>'+name+'</extra>',
    }];
  });
  const guide = xKey==='iteration'?1000:3;
  return plot('history',traces,{
    height:365,margin:{l:63,r:22,t:14,b:72},
    xaxis:{...axis,title:{text:xTitle}},yaxis:{...axis,title:{text:'Validation BPB'}},
    shapes:[{type:'line',xref:'x',yref:'paper',x0:guide,x1:guide,y0:0,y1:1,layer:'below',line:{color:'#91a6a0',dash:'dash',width:1}}],
  },{title:'Validation BPB history · '+title,subtitle:xTitle+' · stars mark resumed endpoint evaluations',filename:'learning-'+key+'-'+xKey});
}

function showResults() {
  const tbody = $('results').querySelector('tbody');
  for (const run of order.flatMap(pairRuns)) {
    const tr = tbody.insertRow();
    for (const value of [languages[run.language],representations[run.representation],tokenizers[run.tokenizer],
        num(run.vocab),num(run.endpoint),fixed(run.finalEpoch,4),fixed(run.finalBpb)]) tr.insertCell().textContent=value;
  }
}

const loading = new Map();
function loadRun(run) {
  if (loading.has(run.id)) return loading.get(run.id);
  const promise = new Promise((resolve,reject)=>{
    const script = document.createElement('script');
    script.src = 'data/'+run.id+'.js';
    script.onload = () => {
      const data = window.PR926_TOKENS[run.id];
      script.remove();
      if (data) resolve(data);
      else reject(new Error('The run data is incomplete.'));
    };
    script.onerror = () => {
      script.remove();
      reject(new Error('Keep the dashboard and its data folder together.'));
    };
    document.head.append(script);
  });
  loading.set(run.id,promise);
  promise.catch(()=>loading.delete(run.id));
  return promise;
}

function rankTokens(snapshot, metric, includeMissing) {
  const present = [], absent = [];
  snapshot[metric].forEach((value,id)=>(value===null?absent:present).push(id));
  const direction = metric===0?1:-1;
  present.sort((a,b)=>direction*(snapshot[metric][a]-snapshot[metric][b])||a-b);
  const ids = includeMissing?[...absent,...present]:present;
  return ids.map((id,i)=>({id,x:ids.length>1?100*i/(ids.length-1):50,unranked:snapshot[metric][id]===null}));
}
function pointsFor(data,snapshot,order,metric,index,useOccurrence=false) {
  const x=[],y=[],pointColors=[],customdata=[];
  for (const point of order) {
    const {id} = point;
    const value = snapshot[metric][id];
    if (value===null) continue;
    x.push(useOccurrence?displayValue(snapshot[0][id],0):point.x);
    y.push(displayValue(value,metric));
    pointColors.push(point.unranked?GRAY:snapshot[0][id]===0?lightColors[index]:colors[index]);
    customdata.push([id,escapeHtml(data.tokens[id]),value,snapshot[0][id],snapshot[8][id]]);
  }
  return {x,y,customdata,marker:{color:pointColors,opacity:.78,size:3.5,symbol:index?'diamond':'circle'}};
}

function comparisonRuns() {
  const language = $('compare-language').value;
  const fixedValue = $('compare-fixed').value;
  if ($('compare-mode').value==='tokenizer') return pairRuns(language+'_'+fixedValue);
  return ['original','ipa'].map(rep=>runs.get(language+'_'+rep+'_'+fixedValue));
}
function configureComparison(changeDimension=false) {
  const byTokenizer = $('compare-mode').value==='tokenizer';
  if (changeDimension) {
    $('compare-fixed-label').textContent=byTokenizer?'Text':'Tokenizer';
    select('compare-fixed',Object.entries(byTokenizer?representations:tokenizers));
  }
  const [a,b] = comparisonRuns();
  const shared = a.steps.filter(s=>s!==a.endpoint&&s!==b.endpoint&&b.steps.includes(s));
  const previous = $('compare-step').value || 'final';
  select('compare-step',[['final','Resumed endpoints (~3 epochs)'],...shared.map(s=>[s,'Step '+num(s)])],previous);
}
const checkpoint = (run,value) => value==='final'?run.endpoint:Number(value);
const checkpointName = (run,step) => step===run.endpoint?'Endpoint · step '+num(step):'Step '+num(step);
let comparisonVersion = 0;
let activeComparison;
async function renderComparison() {
  const revision = ++comparisonVersion;
  activeComparison = undefined;
  $('make-matrix').disabled = true;
  $('save-matrix').hidden = true;
  $('matrix-figure').hidden = true;
  exportButton('compare-chart').disabled = true;
  const selected = comparisonRuns();
  const steps = selected.map(run=>checkpoint(run,$('compare-step').value));
  const sort = Number($('compare-sort').value);
  const range = {all:[0,1,2,3,4,5,6,7],core:[0,1,2,3,4],probability:[5,6,7]}[$('compare-metrics').value];
  const includeMissing = $('include-missing').checked;
  const seriesName = selected.map(run=>$('compare-mode').value==='tokenizer'?tokenizers[run.tokenizer]:representations[run.representation]);
  const title = languages[selected[0].language]+' · '+($('compare-mode').value==='tokenizer'?representations[selected[0].representation]:tokenizers[selected[0].tokenizer]);
  const stepText = seriesName.map((name,i)=>name+' '+checkpointName(selected[i],steps[i]).toLowerCase()).join(' · ');
  $('compare-caption').textContent = 'Loading token measurements…';
  try {
    const data = await Promise.all(selected.map(loadRun));
    if (revision!==comparisonVersion) return;
    const snapshots = data.map((d,i)=>d.snapshots[steps[i]]);
    const ranks = snapshots.map(snapshot=>rankTokens(snapshot,sort,includeMissing));
    const layout = {height:range.length*141+75,margin:{l:56,r:16,t:23,b:66},annotations:[],legend:{...base.legend,y:-.042}};
    const traces = [];
    range.forEach((metric,row)=>{
      const suffix = row===0?'':String(row+1);
      const top = 1-row/range.length;
      layout['xaxis'+suffix] = {...axis,anchor:'y'+suffix,domain:[0,1],range:[-1,101],tickvals:[0,25,50,75,100],
        showticklabels:row===range.length-1,
        title:row===range.length-1?{text:'Rank percentile'}:undefined,
        matches:row?'x':undefined};
      layout['yaxis'+suffix] = {...axis,anchor:'x'+suffix,domain:[top-0.79/range.length,top-0.16/range.length],
        tickformat:metric===5||metric===7?'.0%':undefined,nticks:4};
      layout.annotations.push({xref:'paper',yref:'paper',x:0,y:top,showarrow:false,xanchor:'left',yanchor:'top',
        text:'<b>'+escapeHtml(metricName(metric))+'</b>',font:{size:12}});
      snapshots.forEach((snapshot,i)=>{
        traces.push({type:'scattergl',mode:'markers',xaxis:'x'+suffix,yaxis:'y'+suffix,
          ...pointsFor(data[i],snapshot,ranks[i],metric,i),
          name:seriesName[i],legendgroup:seriesName[i],showlegend:row===0,
          hovertemplate:'ID %{customdata[0]} · %{customdata[1]}<br>Value %{customdata[2]:.6g}<br>Training occurrence %{customdata[3]:,d}<br>Validation observations %{customdata[4]:,d}<extra>'+seriesName[i]+'</extra>'});
      });
    });
    await plot('compare-chart',traces,layout,{
      title:'Ranked token metrics · '+title,subtitle:stepText+' · sorted by '+metrics[sort][0],
      filename:'ranked-'+selected.map(run=>run.id).join('-vs-')+'-'+steps.join('-')+'-'+DATA.metrics[sort],
    });
    if (revision!==comparisonVersion) return;
    const direction = sort===0?'low to high':'high to low';
    $('compare-caption').textContent = title+' · '+stepText+' · '+metrics[sort][0]+' '+direction;
    activeComparison = {selected,steps,snapshots,title,stepText,seriesName,includeMissing};
    $('make-matrix').disabled = false;
  } catch (error) {
    if (revision===comparisonVersion) $('compare-caption').textContent=error.message;
    console.error(error);
  }
}

function metricBounds(snapshots,metric) {
  let low=Infinity, high=-Infinity;
  for (const snapshot of snapshots) for (const v of snapshot[metric]) {
    if (v===null) continue;
    const value = displayValue(v,metric);
    low = Math.min(low,value);
    high = Math.max(high,value);
  }
  if (!Number.isFinite(low)) return [0,1];
  if (metric!==3&&metric!==4) low=0;
  const pad = Math.max(.00001,(high-low)*.045);
  return [Math.max(0,low-pad),high+pad];
}
function drawMatrix() {
  if (!activeComparison) return;
  const {selected,steps,snapshots,title,stepText,seriesName,includeMissing}=activeComparison;
  const canvas = $('matrix');
  const width=1820,height=1320,cellWidth=208,cellHeight=139;
  const left=128,top=145;
  canvas.width=width*2;
  canvas.height=height*2;
  const ctx=canvas.getContext('2d');
  ctx.scale(2,2);
  ctx.fillStyle='#fff';
  ctx.fillRect(0,0,width,height);
  function text(x,y,value,style={}) {
    ctx.fillStyle=style.color||'#263b37';
    ctx.font=(style.bold?'bold ':'')+(style.size||12)+'px Arial';
    ctx.textAlign=style.align||'left';
    ctx.fillText(value,x,y,style.maxWidth||width-x);
  }
  function line(x1,y1,x2,y2,stroke) {
    ctx.strokeStyle=stroke;
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }
  text(24,34,'Token sorting matrix · '+title,{bold:true,size:22});
  text(24,61,stepText+' · independent rank percentiles',{size:15});
  seriesName.forEach((name,i)=>text(24+260*i,88,name+' · vocabulary '+num(selected[i].vocab),{bold:true,color:colors[i],size:14}));
  text(24,112,'Occurrence '+(isLog()?'uses ln(1 + count)':'uses raw counts')+' and sorts low to high. Other columns sort high to low. Missing measurements are blank.');
  const rankings = metrics.map((_,m)=>snapshots.map(s=>rankTokens(s,m,includeMissing)));
  metrics.forEach((m,c)=>text(left+c*cellWidth+95,135,'Sort by '+m[1],{bold:true,align:'center',maxWidth:195}));
  metrics.forEach((m,r)=>{
    const [low,high]=metricBounds(snapshots,r);
    const y0=top+r*cellHeight;
    text(13,y0+18,m[1],{bold:true,maxWidth:77});
    for (let j=0;j<3;j++) {
      const v=low+(high-low)*(2-j)/2;
      text(left-7,y0+9+j*51,num(v,3),{align:'right',color:'#526460',maxWidth:43});
    }
    metrics.forEach((_,c)=>{
      const x0=left+c*cellWidth,w=191,h=106;
      if (c===r) {
        ctx.fillStyle='#f2f6f3'; ctx.fillRect(x0,y0,w,h);
      }
      for (let tick=0;tick<3;tick++) {
        line(x0+tick*w/2,y0,x0+tick*w/2,y0+h,'#e1e8e3');
        line(x0,y0+tick*h/2,x0+w,y0+tick*h/2,'#e1e8e3');
      }
      for (const [i,snapshot] of snapshots.entries()) {
        const draw=rankings[c][i];
        for (const point of draw) {
          const v=snapshot[r][point.id];
          if (v===null) continue;
          const x=x0+point.x*w/100;
          const y=y0+(high-displayValue(v,r))*h/(high-low);
          ctx.fillStyle=point.unranked?GRAY:snapshot[0][point.id]===0?lightColors[i]:colors[i];
          ctx.globalAlpha=point.unranked?.26:.58;
          ctx.fillRect(x-.7,y-.7,1.4,1.4);
        }
        ctx.globalAlpha=1;
      }
      if (r===metrics.length-1) for (const tick of [0,50,100]) text(x0+tick*w/100,y0+h+16,String(tick),{align:'center',color:'#526460'});
    });
  });
  text(width/2,height-25,'Rank percentile · each series sorted independently',{align:'center',color:'#526460',size:14});
  $('matrix-figure').hidden=false;
  $('save-matrix').hidden=false;
  $('save-matrix').onclick=async ()=>{
    try { saveBlob(await pngBlob(canvas),'pr926-matrix-'+selected.map(r=>r.id).join('-vs-')+'-'+steps.join('-')+'.png'); }
    catch (error) { status(error.message); }
  };
}

let explorerVersion=0;
let activeExplorer;
let tokenSelection;
function configureExplorer() {
  const run=runs.get($('explore-run').value);
  const old=$('explore-step').value || 'final';
  select('explore-step',[['final',checkpointName(run,run.endpoint)],
    ...run.steps.filter(step=>step!==run.endpoint).map(step=>[step,'Step '+num(step)])],old);
  $('token-id').max=String(run.vocab-1);
  $('token-id').value='';
  tokenSelection=undefined;
}
async function renderExplorer() {
  const revision=++explorerVersion;
  activeExplorer=undefined;
  $('find-token').disabled=true;
  $('save-token-csv').disabled=true;
  $('token-detail').hidden=true;
  exportButton('token-chart').disabled=true;
  exportButton('token-history').disabled=true;
  const run=runs.get($('explore-run').value);
  const step=checkpoint(run,$('explore-step').value);
  const metric=Number($('explore-metric').value);
  const occurrence=$('explore-x').value==='occurrence';
  $('explore-sort').disabled=occurrence;
  const sort=occurrence?0:Number($('explore-sort').value);
  const i=run.tokenizer==='char'?0:1;
  const caption=runName(run)+' · '+checkpointName(run,step);
  $('explore-caption').textContent='Loading token measurements…';
  try {
    const data=await loadRun(run);
    if (revision!==explorerVersion) return;
    const snapshot=data.snapshots[step];
    const values=pointsFor(data,snapshot,rankTokens(snapshot,sort,false),metric,i,occurrence);
    await plot('token-chart',[{...values,type:'scattergl',mode:'markers',name:tokenizers[run.tokenizer],
      hovertemplate:'ID %{customdata[0]} · %{customdata[1]}<br>Value %{customdata[2]:.6g}<br>Training occurrence %{customdata[3]:,d}<br>Validation observations %{customdata[4]:,d}<extra></extra>'}],{
      height:337,showlegend:false,
      xaxis:{...axis,title:{text:occurrence?metricName(0):'Rank percentile · '+metrics[sort][0]}},
      yaxis:{...axis,title:{text:metricName(metric)},tickformat:metric===5||metric===7?'.0%':undefined},
    },{title:'Token distribution · '+metricName(metric),subtitle:caption,filename:'token-distribution-'+run.id+'-'+step+'-'+DATA.metrics[metric]});
    if (revision!==explorerVersion) return;
    const chart=$('token-chart');
    chart.removeAllListeners('plotly_click');
    chart.on('plotly_click',event=>{
      const id=event.points[0]?.customdata?.[0];
      if (Number.isInteger(id)) report(showToken(id));
    });
    $('explore-caption').textContent=caption+' · '+num(values.x.length)+' plotted tokens';
    activeExplorer={run,step,metric,i,data,snapshot,caption};
    $('find-token').disabled=false;
    $('save-token-csv').disabled=false;
    if (Number.isInteger(tokenSelection)) await showToken(tokenSelection);
  } catch (error) {
    if (revision===explorerVersion) $('explore-caption').textContent=error.message;
    console.error(error);
  }
}
async function showToken(id) {
  if (!activeExplorer) return;
  const {run,step,metric,i,data,snapshot}=activeExplorer;
  if (!Number.isInteger(id)||id<0||id>=run.vocab) return;
  tokenSelection=id;
  $('token-id').value=id;
  const token=data.tokens[id].trim()===''?JSON.stringify(data.tokens[id]):data.tokens[id];
  const title='Token '+id+' · '+token;
  $('token-title').textContent=title;
  const details=metricName(metric)+': '+fixed(displayValue(snapshot[metric][id],metric),5)
    +' · '+num(snapshot[0][id])+' training occurrences · '+num(snapshot[8][id])+' validation observations';
  $('token-values').textContent=checkpointName(run,step)+' · '+details;
  $('token-detail').hidden=false;
  const series=run.steps.map(s=>data.snapshots[s][metric][id]);
  await plot('token-history',[{type:'scatter',mode:'lines+markers',x:run.steps,
    y:series.map(v=>displayValue(v,metric)),customdata:series,
    line:{color:colors[i],width:2.5},
    marker:{color:colors[i],symbol:run.steps.map(s=>s===run.endpoint?'star':i?'diamond':'circle'),size:run.steps.map(s=>s===run.endpoint?13:6)},
    connectgaps:false,
    hovertemplate:'Step %{x:,d}<br>Raw value %{customdata:.6g}<extra></extra>'}],{
      height:270,showlegend:false,margin:{l:62,r:18,t:12,b:54},
      xaxis:{...axis,title:{text:'Optimizer updates'}},
      yaxis:{...axis,title:{text:metricName(metric)},tickformat:metric===5||metric===7?'.0%':undefined},
      shapes:[{type:'line',xref:'x',yref:'paper',x0:step,x1:step,y0:0,y1:1,line:{color:'#91a6a0',width:1,dash:'dash'}}],
    },{title:title+' · '+metricName(metric),subtitle:runName(run)+' · star marks the resumed endpoint',filename:'token-'+run.id+'-'+id+'-'+DATA.metrics[metric]});
}
function exportCheckpoint() {
  if (!activeExplorer) return;
  const {run,step,data,snapshot}=activeExplorer;
  const csvField = value => value===null?'':'"'+String(value).replace(/"/g,'""')+'"';
  const rows=[['iteration','token_id','token_text_escaped',...DATA.metrics].join(',')];
  data.tokens.forEach((token,id)=>rows.push([step,id,token,...snapshot.map(metric=>metric[id])].map(csvField).join(',')));
  saveBlob(new Blob([rows.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'}),'pr926-'+run.id+'-'+step+'.csv');
}
function report(promise) {
  Promise.resolve(promise).catch(error=>{console.error(error);status('A chart could not be rendered. '+error.message);});
}
function setup() {
  const langOptions=Object.entries(languages);
  const metricOptions=metrics.map((metric,i)=>[i,metric[0]]);
  ['curve-language','compare-language'].forEach(id=>select(id,langOptions,'ja'));
  select('curve-rep',Object.entries(representations),'original');
  select('compare-sort',metricOptions,1);
  configureComparison(true);
  select('explore-run',order.flatMap(pairRuns).map(run=>[run.id,runName(run)]),'ja_original_char');
  select('explore-metric',metricOptions,1);
  select('explore-sort',metricOptions,0);
  configureExplorer();
  for (const button of document.querySelectorAll('[data-export]')) button.onclick=()=>report(savePlot(button.dataset.export));
  $('budget').onchange=()=>report(renderAbsolute());
  for (const id of ['curve-language','curve-rep','curve-axis']) $(id).onchange=()=>report(renderHistory());
  for (const id of ['compare-mode','compare-language','compare-fixed']) $(id).onchange=()=>{
    configureComparison(id==='compare-mode');
    report(renderComparison());
  };
  for (const id of ['compare-step','compare-sort','compare-metrics','include-missing']) $(id).onchange=()=>report(renderComparison());
  $('log-occurrence').onchange=()=>{report(renderComparison());report(renderExplorer());};
  $('make-matrix').onclick=drawMatrix;
  $('explore-run').onchange=()=>{configureExplorer();report(renderExplorer());};
  for (const id of ['explore-step','explore-metric','explore-x','explore-sort']) $(id).onchange=()=>report(renderExplorer());
  $('token-form').onsubmit=event=>{event.preventDefault();report(showToken($('token-id').valueAsNumber));};
  $('save-token-csv').onclick=exportCheckpoint;
  showResults();
  [renderAdvantage(),renderAbsolute(),renderHistory()].forEach(report);
  const lazy = new IntersectionObserver(entries=>{
    for (const entry of entries) if (entry.isIntersecting) {
      lazy.unobserve(entry.target);
      report(entry.target.id==='comparison'?renderComparison():renderExplorer());
    }
  },{rootMargin:'300px'});
  ['comparison','explorer'].forEach(id=>lazy.observe($(id)));
}
setup();
