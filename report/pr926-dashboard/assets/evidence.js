'use strict';

const TOKEN_EVIDENCE = window.PR926_TOKEN_EVIDENCE;
const EVIDENCE_EXAMPLE = 'ko_original_char';
const evidenceStages = [
  ['paper','Paper · 3,500','#687987','circle-open',-.21],
  ['early','PR · 1,000',BLUE,'diamond',0],
  ['endpoint','PR endpoint',ORANGE,'square',.21],
];
const coefficientText = value => (value>0?'+':'')+value.toFixed(3);
const evidenceRunLabel = run => languages[run.language]+' · '+(run.representation==='ipa'?'IPA':'orig.')+' · '+tokenizers[run.tokenizer];
const evidenceCheckpoint = (run,stage) => checkpointName(run,TOKEN_EVIDENCE.summary[run.id][stage].step);

function renderCoefficientChart() {
  const method = $('evidence-method').value;
  const adjusted = method==='adjusted';
  const selected = order.flatMap(pairRuns);
  const example = TOKEN_EVIDENCE.summary[EVIDENCE_EXAMPLE];
  const row = selected.findIndex(run=>run.id===EVIDENCE_EXAMPLE);
  const bounds = adjusted?[-.70,.135]:[-.57,.57];
  const point = example.endpoint[method];
  const traces = evidenceStages.map(([key,name,color,symbol,offset])=>({
    type:'scatter',mode:'markers',name,
    x:selected.map(run=>TOKEN_EVIDENCE.summary[run.id][key][method]),
    y:selected.map((_,i)=>i+offset),
    marker:{symbol,color,size:8,line:{color,width:1.3}},
    customdata:selected.map(run=>{
      const record=TOKEN_EVIDENCE.summary[run.id][key];
      return [run.id,key,runName(run),num(record.step),num(record.n),coefficientText(record[method])];
    }),
    hovertemplate:'%{customdata[2]}<br>Step %{customdata[3]}<br>ρ %{customdata[5]}<br>%{customdata[4]} included token IDs<extra>'+name+'</extra>',
  }));
  $('coefficient-note').textContent=adjusted
    ? 'In the shaded Korean original Char row, the orange square is just to the right of zero (+0.014). The paper value (−0.067) and PR step 1,000 (−0.291) are both left of zero.'
    : 'In the shaded row, the orange square is +0.306 before adjustment. Switch to the adjusted view to see it move close to zero (+0.014).';
  const tbody=$('evidence-table').querySelector('tbody');
  tbody.replaceChildren();
  selected.forEach(run=>{
    const tr=tbody.insertRow();
    tr.classList.toggle('is-example',run.id===EVIDENCE_EXAMPLE);
    const rec=TOKEN_EVIDENCE.summary[run.id];
    const values=[runName(run),...evidenceStages.flatMap(([key])=>[coefficientText(rec[key][method]),num(rec[key].n)])];
    for (const value of values) tr.insertCell().textContent=value;
  });
  return plot('evidence-coefficients',traces,{
    height:502,margin:{l:146,r:18,t:13,b:70},
    xaxis:{...axis,zeroline:false,range:bounds,title:{text:adjusted?'Loss–norm partial rank correlation (ρ)':'Loss–norm Spearman correlation (ρ)'}},
    yaxis:{...axis,showgrid:false,zeroline:false,range:[selected.length-.5,-.5],tickvals:selected.map((_,i)=>i),
      ticktext:selected.map(run=>evidenceRunLabel(run)),tickfont:{size:11},fixedrange:true},
    legend:{orientation:'h',x:0,y:-.14,font:{size:11}},
    shapes:[
      {type:'rect',xref:'paper',yref:'y',x0:0,x1:1,y0:row-.46,y1:row+.46,line:{width:0},fillcolor:'#e8f1eb',layer:'below'},
      {type:'line',xref:'x',yref:'paper',x0:0,x1:0,y0:0,y1:1,line:{color:'#718a7e',width:1.3,dash:'dash'},layer:'below'},
    ],
    annotations:[{xref:'x',yref:'y',x:point,y:row+.21,text:'<b>'+coefficientText(point)+'</b>',xshift:9,
      showarrow:false,xanchor:'left',font:{size:11,color:ORANGE}}],
  },{
    title:'Loss–norm correlation across the 12 runs',
    subtitle:adjusted?'After adjusting for training occurrence · paper Table 4 and PR #926':'Unadjusted Spearman correlation · paper Table 4 and PR #926',
    filename:'evidence-loss-norm-'+method,
  }).then(()=>{
    const chart=$('evidence-coefficients');
    chart.removeAllListeners('plotly_click');
    chart.on('plotly_click',event=>{
      const [id,stage]=event.points[0]?.customdata??[];
      if (!id) return;
      $('evidence-run').value=id;
      if (stage!=='paper') $('evidence-step').value=stage;
      report(renderEvidenceDetail());
    });
  });
}

function tiedPercentileRanks(values) {
  const order = values.map((_,i)=>i).sort((i,j)=>values[i]-values[j]||i-j);
  const result = new Array(order.length);
  for (let start=0;start<order.length;) {
    let end=start+1;
    while (end<order.length&&values[order[start]]===values[order[end]]) end++;
    const rank=100*(start+end-1)/(2*(order.length-1));
    for (let i=start;i<end;i++) result[order[i]]=rank;
    start=end;
  }
  return result;
}
const mean = values=>values.reduce((sum,value)=>sum+value,0)/values.length;
function linearFit(x,y) {
  const xMean=mean(x),yMean=mean(y);
  let xy=0,xx=0,yy=0;
  for (let i=0;i<x.length;i++) {
    const a=x[i]-xMean,b=y[i]-yMean;
    xy+=a*b;
    xx+=a*a;
    yy+=b*b;
  }
  const slope=xy/xx;
  return {slope,intercept:yMean-slope*xMean,rho:xy/Math.sqrt(xx*yy)};
}
function residualOf(y,x) {
  const {slope,intercept}=linearFit(x,y);
  return y.map((value,i)=>value-(slope*x[i]+intercept));
}
function makeRankPairs(snapshot) {
  const ids = snapshot[0].map((_,id)=>id).filter(id=>snapshot[0][id]>0&&snapshot[8][id]>=20
    && Number.isFinite(snapshot[1][id])&&Number.isFinite(snapshot[3][id]));
  const ranks=metric=>tiedPercentileRanks(ids.map(id=>snapshot[metric][id]));
  const count=ranks(0),norm=ranks(3),loss=ranks(1);
  return {ids, raw:[norm,loss], adjusted:[residualOf(norm,count),residualOf(loss,count)]};
}
let evidenceRevision=0;
async function renderEvidenceDetail() {
  const revision=++evidenceRevision;
  const run=runs.get($('evidence-run').value);
  const stage=$('evidence-step').value;
  const expected=TOKEN_EVIDENCE.summary[run.id][stage];
  const context=runName(run)+' · '+evidenceCheckpoint(run,stage);
  $('evidence-detail-caption').textContent='Loading '+context+'…';
  exportButton('evidence-raw').disabled=true;
  exportButton('evidence-adjusted').disabled=true;
  try {
    const data=await loadRun(run);
    if (revision!==evidenceRevision) return;
    const snapshot=data.snapshots[expected.step];
    const pairs=makeRankPairs(snapshot);
    if (pairs.ids.length!==expected.n) throw new Error('Token counts differ from the recorded comparison.');
    const customdata=pairs.ids.map(id=>[id,escapeHtml(data.tokens[id]),snapshot[1][id],snapshot[3][id],num(snapshot[0][id]),num(snapshot[8][id])]);
    const plots=['raw','adjusted'].map(kind=>{
      const isAdjusted=kind==='adjusted';
      const [x,y]=pairs[kind];
      const fit=linearFit(x,y);
      if (Math.abs(fit.rho-expected[kind])>0.0005) throw new Error('The plotted correlation differs from the recorded comparison.');
      const limit=Math.ceil(Math.max(...x.map(Math.abs),...y.map(Math.abs))/10)*10;
      const bounds=isAdjusted?[-limit,limit]:[-2,102];
      const lineX=[Math.min(...x),Math.max(...x)];
      const suffix=isAdjusted?' after adjustment':' before adjustment';
      $('evidence-'+kind+'-rho').textContent='ρ = '+coefficientText(fit.rho)+' · '+num(pairs.ids.length)+' token IDs';
      return plot('evidence-'+kind,[
        {type:'scattergl',mode:'markers',x,y,customdata,name:'Token ID',showlegend:false,
          marker:{color:isAdjusted?'#358276':BLUE,opacity:.48,size:4.5},
          hovertemplate:'Token %{customdata[0]} · %{customdata[1]}<br>Plotted x %{x:.2f} · y %{y:.2f}<br>Val loss %{customdata[2]:.4f} · L2 norm %{customdata[3]:.4f}<br>Training occurrences %{customdata[4]}<br>Validation observations %{customdata[5]}<extra></extra>'},
        {type:'scatter',mode:'lines',x:lineX,y:lineX.map(value=>fit.slope*value+fit.intercept),showlegend:false,
          hoverinfo:'skip',line:{color:ORANGE,width:2.8}},
      ],{
        height:310,margin:{l:62,r:16,t:12,b:64},
        xaxis:{...axis,range:bounds,title:{text:isAdjusted?'L2 rank residual (pp)':'L2 norm percentile rank'}},
        yaxis:{...axis,range:bounds,title:{text:isAdjusted?'Loss rank residual (pp)':'Validation loss percentile rank'}},
      },{
        title:'Token loss and L2 norm'+suffix,
        subtitle:context+' · ρ '+coefficientText(fit.rho)+' · '+num(pairs.ids.length)+' token IDs',
        filename:'evidence-scatter-'+run.id+'-'+expected.step+'-'+kind,
      });
    });
    await Promise.all(plots);
    if (revision===evidenceRevision) {
      const hint=run.id===EVIDENCE_EXAMPLE
        ? (stage==='endpoint'?'The right-hand fitted line is almost flat.':'The right-hand fitted line slopes down at this checkpoint.')
        : 'The orange line summarizes the direction of the points.';
      $('evidence-detail-caption').textContent=context+' · '+hint;
    }
  } catch(error) {
    if (revision===evidenceRevision) $('evidence-detail-caption').textContent=error.message;
    console.error(error);
  }
}

function renderProbabilityEvidence() {
  const examples=TOKEN_EVIDENCE.examples;
  const step=examples[0].step;
  const rankText=examples.map(e=>e.token+' '+e.rank.toFixed(3)).join(' · ');
  $('evidence-prob-caption').textContent='Korean IPA Char · step '+num(step)+' · Mean target rank '+rankText;
  const traces=examples.map((e,i)=>({
    type:'bar',name:e.token,x:['Target probability','Left probability'],y:[e.target,e.left],
    text:[e.target,e.left].map(value=>(100*value).toFixed(1)+'%'),textposition:'outside',cliponaxis:false,
    marker:{color:colors[i]},
    hovertemplate:'Token '+e.token+' (ID '+e.id+')<br>%{x}: %{y:.2%}<br>Mean target rank '+e.rank.toFixed(3)+'<br>'+num(e.n)+' validation observations<extra></extra>',
  }));
  return plot('evidence-prob',traces,{
    height:300,barmode:'group',margin:{l:58,r:16,t:18,b:67},
    xaxis:{...axis,showgrid:false},
    yaxis:{...axis,range:[0,.8],tickformat:'.0%',title:{text:'Mean probability'}},
    legend:{...base.legend,y:-.19},
  },{title:'Similar target rank, different probabilities',subtitle:'Korean IPA Char · step '+num(step)+' · Mean ranks '+rankText,filename:'evidence-target-and-left-probability'});
}

function renderByteEvidence() {
  const series=TOKEN_EVIDENCE.byte;
  const trace = (key,name,color,symbol) => ({
    type:'scatter',mode:'lines+markers+text',name,x:series.map(s=>s.step),y:series.map(s=>s[key]),
    text:key==='reserved'?series.map((s,i)=>[0,2,series.length-1].includes(i)?s[key].toFixed(1)+'°':''):[],
    textposition:'bottom center',textfont:{color,size:11},cliponaxis:false,
    line:{color,width:2.5},marker:{color,symbol,size:7},
    hovertemplate:'Step %{x:,d}<br>Median angle %{y:.2f}°<extra>'+name+'</extra>',
  });
  return plot('evidence-bytes',[
    trace('reserved','256 reserved bytes',ORANGE,'diamond'),trace('other','Other vocabulary IDs',BLUE,'circle'),
  ],{
    height:300,margin:{l:63,r:29,t:16,b:68},
    xaxis:{...axis,title:{text:'Optimizer updates'}},
    yaxis:{...axis,range:[19,88],title:{text:'Median minimum angle (°)'}},
    legend:{...base.legend,y:-.25,font:{size:11}},
  },{title:'Nearest-vector angle for reserved byte IDs',subtitle:'Korean IPA CharBPE · 256 reserved byte IDs and 7,936 other IDs',filename:'evidence-reserved-byte-angle'});
}

function setupPaperEvidence() {
  select('evidence-run',order.flatMap(pairRuns).map(run=>[run.id,runName(run)]),EVIDENCE_EXAMPLE);
  for (const id of ['evidence-coefficients','evidence-raw','evidence-adjusted','evidence-prob','evidence-bytes']) {
    exportButton(id).onclick=()=>report(savePlot(id));
  }
  $('evidence-method').onchange=()=>report(renderCoefficientChart());
  for (const id of ['evidence-run','evidence-step']) $(id).onchange=()=>report(renderEvidenceDetail());
  $('evidence-open-matrix').onclick=()=>{
    const run=runs.get($('evidence-run').value);
    $('compare-mode').value='tokenizer';
    $('compare-language').value=run.language;
    configureComparison(true);
    $('compare-fixed').value=run.representation;
    configureComparison();
    $('compare-step').value=$('evidence-step').value==='endpoint'?'final':'1000';
    $('compare-sort').value='3';
    $('compare-metrics').value='core';
    location.hash='comparison';
    report(renderComparison());
  };
  const lazy = new IntersectionObserver(entries=>{
    if (!entries.some(entry=>entry.isIntersecting)) return;
    lazy.disconnect();
    [renderCoefficientChart(),renderEvidenceDetail(),renderProbabilityEvidence(),renderByteEvidence()].forEach(report);
  },{rootMargin:'260px'});
  lazy.observe($('paper-evidence'));
}
setupPaperEvidence();
