import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as math from "mathjs";
import _ from "lodash";

// ============================================================
// CONFIG
// ============================================================
const CLAUDE_API_KEY = process.env.NEXT_PUBLIC_CLAUDE_API_KEY || "";
const CLAUDE_MODEL = "claude-sonnet-4-20250514";
const MC_PATHS = 50000;
const MC_STEPS = 252;

// ============================================================
// MATH ENGINE
// ============================================================
const normCDF = (x) => {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign=x<0?-1:1;
  const t=1/(1+p*Math.abs(x)/Math.sqrt(2));
  const erf=1-((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*Math.exp(-x*x/2);
  return 0.5*(1+sign*erf);
};

const normalRandom = () => {
  let u1; do { u1=Math.random(); } while(u1===0);
  return Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*Math.random());
};

const generateGBMPath = (S0,mu,sigma,T,steps) => {
  const dt=T/steps, path=[S0];
  for(let i=1;i<=steps;i++){
    const z=normalRandom();
    path.push(path[i-1]*Math.exp((mu-0.5*sigma*sigma)*dt+sigma*Math.sqrt(dt)*z));
  }
  return path;
};

// ============================================================
// PAYOFF BUILDING BLOCKS
// ============================================================
const PayoffBlocks = {
  linear:(v,threshold,rate) => Math.max(0,(v-threshold)*rate),
  binary:(v,threshold,payoff) => v>=threshold?payoff:0,
  tiered:(v,tiers) => {
    let total=0;
    for(const t of tiers){
      if(v>t.lower){
        const applicable=Math.min(v,t.upper||Infinity)-t.lower;
        total+=Math.max(0,applicable)*t.rate;
      }
    }
    return total;
  },
  capped:(v,cap) => Math.min(v,cap),
  floored:(v,floor) => Math.max(v,floor),
  multiMetric:(values,conditions,payoff) => {
    const allMet=conditions.every((c,i) => values[i]>=c.threshold);
    return allMet?payoff:0;
  },
  catchUp:(currentVal,priorShortfall,threshold,rate,cap) => {
    const effectiveThreshold=Math.max(0,threshold-priorShortfall);
    let payout=Math.max(0,(currentVal-effectiveThreshold)*rate);
    if(cap) payout=Math.min(payout,cap);
    return payout;
  },
  acceleration:(metricVal,trigger,percentile,projections) => {
    if(trigger==="change_of_control"){
      const targetVal=projections*(percentile/100);
      return targetVal;
    }
    return metricVal;
  }
};

// ============================================================
// MONTE CARLO ENGINE
// ============================================================
const runMonteCarlo = (config) => {
  const {S0,mu,sigma,T,riskFreeRate,numPaths,numSteps,payoffFn,discountRate,creditAdj=0} = config;
  const results=[];
  const effectiveDiscount=discountRate+creditAdj;
  
  for(let i=0;i<numPaths;i++){
    const path=generateGBMPath(S0,mu,sigma,T,numSteps);
    const terminalValue=path[path.length-1];
    const payoff=payoffFn(terminalValue,path);
    results.push(payoff*Math.exp(-effectiveDiscount*T));
  }
  
  results.sort((a,b)=>a-b);
  const mean=results.reduce((a,b)=>a+b,0)/results.length;
  const variance=results.reduce((a,b)=>a+(b-mean)**2,0)/(results.length-1);
  const stdErr=Math.sqrt(variance/results.length);
  
  const numBins=50;
  const minVal=results[0],maxVal=results[results.length-1];
  const binWidth=(maxVal-minVal)/numBins||1;
  const histogram=Array.from({length:numBins},(_,i)=>({
    x:minVal+i*binWidth+binWidth/2,count:0,
    lower:minVal+i*binWidth,upper:minVal+(i+1)*binWidth
  }));
  results.forEach(v=>{
    const idx=Math.min(Math.floor((v-minVal)/binWidth),numBins-1);
    if(idx>=0) histogram[idx].count++;
  });
  
  const convergence=[];
  let sum=0;
  const step=Math.max(1,Math.floor(results.length/100));
  for(let i=0;i<results.length;i++){
    sum+=results[i];
    if(i%step===0&&i>0) convergence.push({n:i,mean:sum/(i+1)});
  }
  
  return {
    fairValue:mean,stdError:stdErr,
    ci95:[mean-1.96*stdErr,mean+1.96*stdErr],
    percentiles:{
      p5:results[Math.floor(0.05*results.length)],
      p10:results[Math.floor(0.10*results.length)],
      p25:results[Math.floor(0.25*results.length)],
      p50:results[Math.floor(0.50*results.length)],
      p75:results[Math.floor(0.75*results.length)],
      p90:results[Math.floor(0.90*results.length)],
      p95:results[Math.floor(0.95*results.length)],
    },
    histogram,convergence,rawResults:results,
    probPayoff:(results.filter(r=>r>0).length/results.length*100).toFixed(1),
  };
};

// ============================================================
// SENSITIVITY ENGINE
// ============================================================
const runSensitivityGrid = (baseConfig,payoffFn,paramKey,paramRange,steps=15) => {
  const data=[];
  for(let i=0;i<=steps;i++){
    const val=paramRange[0]+(paramRange[1]-paramRange[0])*i/steps;
    const cfg={...baseConfig,[paramKey]:val};
    let S0=cfg.projectedMetric||cfg.S0;
    const r=runMonteCarlo({S0,mu:0.05,sigma:cfg.volatility,T:cfg.timeToMaturity,riskFreeRate:cfg.riskFreeRate,numPaths:5000,numSteps:126,payoffFn,discountRate:cfg.discountRate,creditAdj:cfg.creditAdj||0});
    data.push({value:val,fairValue:r.fairValue});
  }
  return data;
};

// ============================================================
// DOCUMENT EXTRACTION (Claude API)
// ============================================================
const extractFromDocument = async (text, mode) => {
  const systemPrompts = {
    backtest: `You are a financial data extraction specialist. Extract earnout/contingent consideration information from SEC filings (10-K, 10-Q).

Return ONLY valid JSON with no preamble, no markdown backticks:
{
  "earnouts": [{
    "name": "string (acquisition name or target company)",
    "acquisitionDate": "string or null",
    "maxPayout": number or null,
    "initialFairValue": number or null,
    "currentFairValue": number or null,
    "priorFairValue": number or null,
    "fairValueChange": number or null,
    "metric": "string (Revenue, EBITDA, etc.) or null",
    "metricDefinition": "string or null",
    "structure": "linear|tiered|binary|milestone|unknown",
    "threshold": number or null,
    "participationRate": number or null,
    "cap": number or null,
    "floor": number or null,
    "measurementPeriodEnd": "string or null",
    "timeRemaining": number or null (years),
    "methodology": "Monte Carlo|probability-weighted|DCF|unknown",
    "discountRate": number or null (as decimal),
    "volatility": number or null (as decimal),
    "riskFreeRate": number or null (as decimal),
    "projectedMetric": number or null,
    "level3Rollforward": {
      "openingBalance": number or null,
      "additions": number or null,
      "fairValueChanges": number or null,
      "payments": number or null,
      "closingBalance": number or null
    },
    "confidenceScore": number (0-1)
  }],
  "reportingPeriod": "string",
  "companyName": "string",
  "filingType": "10-K|10-Q"
}

Extract EVERY earnout mentioned. If a value is not disclosed, use null. Never guess.`,

    live_ppa: `You are a valuation report extraction specialist. Extract earnout terms and methodology from a PPA valuation report.

Return ONLY valid JSON with no preamble, no markdown backticks:
{
  "earnout": {
    "name": "string (deal/target name)",
    "metric": "string (Revenue, EBITDA, Adjusted EBITDA, etc.)",
    "metricDefinition": "string (how the metric is calculated/adjusted)",
    "structure": "linear|tiered|binary|milestone|multi-metric",
    "thresholds": [{"level": number, "label": "string"}],
    "participationRate": number (as decimal),
    "tiers": [{"lower": number, "upper": number, "rate": number}] or null,
    "cap": number or null,
    "floor": number or null,
    "measurementPeriods": [{"start": "string", "end": "string", "label": "string"}],
    "accelerationClauses": [{"trigger": "string", "treatment": "string"}] or null,
    "clawbackProvisions": "string or null",
    "catchUpProvisions": "string or null",
    "paymentTiming": "string",
    "methodology": "Monte Carlo|probability-weighted|DCF",
    "assumptions": {
      "projectedMetric": number,
      "volatility": number (as decimal),
      "discountRate": number (as decimal),
      "riskFreeRate": number (as decimal),
      "creditAdjustment": number or null (as decimal),
      "comparableCompanies": ["string"] or null,
      "volatilityMethod": "string or null",
      "volatilityLookback": "string or null"
    },
    "initialFairValue": number or null,
    "currency": "string",
    "confidenceScore": number (0-1),
    "ambiguities": ["string"],
    "alternativeInterpretations": [{"clause":"string","interpretation1":"string","interpretation2":"string"}] or null
  }
}`,

    live_update: `You are a financial data extraction specialist. Extract updated forecast and performance data from management reports, board presentations, or financial updates.

Return ONLY valid JSON:
{
  "updates": {
    "currentPeriodActuals": number or null,
    "projectedMetric": number or null,
    "projectionBasis": "string (e.g., management forecast, board-approved budget)",
    "periodCovered": "string",
    "keyAssumptions": ["string"],
    "materialEvents": ["string"] or null,
    "confidenceScore": number (0-1)
  }
}`
  };

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method:"POST",
      headers:{"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
      body:JSON.stringify({model:CLAUDE_MODEL,max_tokens:4096,system:systemPrompts[mode],messages:[{role:"user",content:`Extract all earnout/contingent consideration information from this document:\n\n${text.substring(0,80000)}`}]})
    });
    const data=await response.json();
    const txt=data.content?.map(c=>c.text||"").join("")||"";
    return JSON.parse(txt.replace(/```json|```/g,"").trim());
  } catch(err){
    console.error("Extraction error:",err);
    return null;
  }
};

const verifyExtraction = async (text,extracted) => {
  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method:"POST",
      headers:{"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
      body:JSON.stringify({model:CLAUDE_MODEL,max_tokens:2048,
        system:`You are a senior valuation QA reviewer. Verify extracted terms against the source document. Return ONLY valid JSON:
{"verified":boolean,"errors":[{"field":"string","issue":"string","correction":"string or null"}],"missingTerms":[{"term":"string","location":"string"}],"overallConfidence":number,"recommendation":"proceed|review_needed|high_risk"}`,
        messages:[{role:"user",content:`Document:\n${text.substring(0,40000)}\n\nExtracted:\n${JSON.stringify(extracted,null,2)}\n\nVerify.`}]})
    });
    const data=await response.json();
    const txt=data.content?.map(c=>c.text||"").join("")||"";
    return JSON.parse(txt.replace(/```json|```/g,"").trim());
  } catch(err){
    return {verified:false,overallConfidence:0,recommendation:"review_needed",errors:[],missingTerms:[]};
  }
};

// ============================================================
// FORMATTING
// ============================================================
const fmt=(n,d=0)=>{
  if(n==null||isNaN(n))return"—";
  if(Math.abs(n)>=1e9)return`$${(n/1e9).toFixed(2)}B`;
  if(Math.abs(n)>=1e6)return`$${(n/1e6).toFixed(2)}M`;
  if(Math.abs(n)>=1e3)return`$${(n/1e3).toFixed(1)}K`;
  return`$${n.toFixed(d)}`;
};
const fmtPct=(n)=>n==null?"—":`${(n*100).toFixed(1)}%`;

// ============================================================
// EXCEL GENERATION
// ============================================================
const generateExcel = async (params,results,sensitivities) => {
  // Dynamic import of SheetJS
  const XLSX = await import("https://cdn.sheetjs.com/xlsx-0.20.3/package/xlsx.mjs");
  const wb = XLSX.utils.book_new();
  
  // Sheet 1: Summary
  const summaryData = [
    ["ValuProEarnout — Valuation Summary"],
    ["Generated",new Date().toISOString().split("T")[0]],
    [],
    ["EARNOUT TERMS"],
    ["Metric",params.metric||"EBITDA"],
    ["Structure",params.structure],
    ["Threshold",params.threshold],
    ["Participation Rate",params.participationRate],
    ["Cap",params.cap||"None"],
    ["Floor",params.floor||"None"],
    [],
    ["ASSUMPTIONS"],
    ["Projected Metric",params.projectedMetric],
    ["Volatility",params.volatility],
    ["Discount Rate",params.discountRate],
    ["Risk-Free Rate",params.riskFreeRate],
    ["Credit Risk Adjustment",params.creditAdj||0],
    ["Time to Maturity (years)",params.timeToMaturity],
    [],
    ["RESULTS"],
    ["Fair Value",results.fairValue],
    ["95% CI Low",results.ci95[0]],
    ["95% CI High",results.ci95[1]],
    ["Standard Error",results.stdError],
    ["Probability of Payoff",`${results.probPayoff}%`],
    [],
    ["PERCENTILES"],
    ["5th",results.percentiles.p5],
    ["10th",results.percentiles.p10],
    ["25th",results.percentiles.p25],
    ["50th (Median)",results.percentiles.p50],
    ["75th",results.percentiles.p75],
    ["90th",results.percentiles.p90],
    ["95th",results.percentiles.p95],
  ];
  const ws1=XLSX.utils.aoa_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(wb,ws1,"Summary");
  
  // Sheet 2: Sensitivity Tables
  if(sensitivities){
    const sensData=[["Sensitivity Analysis"],[]];
    for(const [label,data] of Object.entries(sensitivities)){
      sensData.push([label]);
      sensData.push(["Input Value","Fair Value"]);
      data.forEach(d=>sensData.push([d.value,d.fairValue]));
      sensData.push([]);
    }
    const ws2=XLSX.utils.aoa_to_sheet(sensData);
    XLSX.utils.book_append_sheet(wb,ws2,"Sensitivity");
  }
  
  // Sheet 3: Distribution
  const distData=[["Distribution Analysis"],["Bin Midpoint","Frequency"]];
  results.histogram.forEach(h=>distData.push([h.x,h.count]));
  const ws3=XLSX.utils.aoa_to_sheet(distData);
  XLSX.utils.book_append_sheet(wb,ws3,"Distribution");
  
  // Sheet 4: Monte Carlo Parameters
  const mcData=[
    ["Monte Carlo Simulation Parameters"],
    [],
    ["Number of Paths",MC_PATHS],
    ["Time Steps",MC_STEPS],
    ["Process","Geometric Brownian Motion"],
    ["Framework","Risk-Neutral"],
    ["Convergence Std Error",results.stdError],
    ["Convergence % of FV",results.fairValue>0?`${(results.stdError/results.fairValue*100).toFixed(2)}%`:"N/A"],
  ];
  const ws4=XLSX.utils.aoa_to_sheet(mcData);
  XLSX.utils.book_append_sheet(wb,ws4,"MC Parameters");
  
  const wbout=XLSX.write(wb,{bookType:"xlsx",type:"array"});
  const blob=new Blob([wbout],{type:"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;a.download=`ValuProEarnout_${new Date().toISOString().split("T")[0]}.xlsx`;
  a.click();URL.revokeObjectURL(url);
};

// ============================================================
// MEMO GENERATION (HTML → Print to PDF / Copy as DOCX)
// ============================================================
const generateMemo = (params,results,sensitivities,format="pdf") => {
  const date=new Date().toLocaleDateString("en-US",{year:"numeric",month:"long",day:"numeric"});
  const html=`<!DOCTYPE html><html><head><meta charset="utf-8"><title>ValuProEarnout - Methodology Memo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Georgia,'Times New Roman',serif;font-size:11pt;line-height:1.6;color:#1a1a1a;max-width:8.5in;margin:0 auto;padding:1in}
h1{font-size:16pt;font-weight:700;margin-bottom:4pt;color:#1a1a1a}
h2{font-size:13pt;font-weight:700;margin-top:18pt;margin-bottom:6pt;color:#1a1a1a;border-bottom:1px solid #ccc;padding-bottom:4pt}
h3{font-size:11pt;font-weight:700;margin-top:12pt;margin-bottom:4pt}
p{margin-bottom:8pt;text-align:justify}
table{width:100%;border-collapse:collapse;margin:10pt 0;font-size:10pt}
th,td{border:1px solid #ccc;padding:4pt 8pt;text-align:left}
th{background:#f5f5f5;font-weight:700}
td.num{text-align:right;font-family:'Courier New',monospace}
.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20pt;padding-bottom:10pt;border-bottom:2px solid #1a1a1a}
.footer{margin-top:30pt;padding-top:10pt;border-top:1px solid #ccc;font-size:9pt;color:#666}
.conf{display:inline-block;padding:2pt 8pt;border-radius:3pt;font-size:9pt;font-weight:700}
@media print{body{padding:0.5in}}
</style></head><body>
<div class="header"><div><h1>Earnout Fair Value Measurement</h1><p style="font-size:10pt;color:#666">Methodology Memorandum — ${date}</p></div><div style="text-align:right"><strong>ValuProEarnout</strong><br><span style="font-size:9pt;color:#666">Automated Valuation Platform</span></div></div>

<h2>1. Overview</h2>
<p>This memorandum documents the fair value measurement of contingent consideration (earnout) in accordance with ASC 820, <em>Fair Value Measurement</em>, and ASC 805, <em>Business Combinations</em>. The earnout liability is classified as Level 3 within the fair value hierarchy as the valuation relies on significant unobservable inputs.</p>

<h2>2. Earnout Terms</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Performance Metric</td><td>${params.metric||"Adjusted EBITDA"}</td></tr>
<tr><td>Payoff Structure</td><td style="text-transform:capitalize">${params.structure}</td></tr>
<tr><td>Threshold / Target</td><td class="num">${fmt(params.threshold)}</td></tr>
<tr><td>Participation Rate</td><td class="num">${fmtPct(params.participationRate)}</td></tr>
<tr><td>Maximum Payout (Cap)</td><td class="num">${params.cap?fmt(params.cap):"Uncapped"}</td></tr>
<tr><td>Minimum Payout (Floor)</td><td class="num">${params.floor?fmt(params.floor):"None"}</td></tr>
<tr><td>Remaining Measurement Period</td><td class="num">${params.timeToMaturity?.toFixed(1)} years</td></tr>
</table>

<h2>3. Valuation Methodology</h2>
<p>Fair value was estimated using a Monte Carlo simulation under the risk-neutral framework. The underlying performance metric was modeled as a Geometric Brownian Motion (GBM) process with ${MC_PATHS.toLocaleString()} simulation paths and ${MC_STEPS} time steps (daily frequency). The simulated terminal metric values were passed through the earnout payoff function and discounted to present value using a risk-adjusted discount rate.</p>

<h2>4. Key Assumptions</h2>
<table>
<tr><th>Assumption</th><th>Value</th><th>Source / Basis</th></tr>
<tr><td>Projected Performance Metric</td><td class="num">${fmt(params.projectedMetric)}</td><td>Management forecast / board-approved budget</td></tr>
<tr><td>Volatility</td><td class="num">${fmtPct(params.volatility)}</td><td>Comparable public company equity volatility</td></tr>
<tr><td>Discount Rate</td><td class="num">${fmtPct(params.discountRate)}</td><td>Risk-adjusted rate reflecting earnout-specific risk</td></tr>
<tr><td>Risk-Free Rate</td><td class="num">${fmtPct(params.riskFreeRate)}</td><td>U.S. Treasury yield matching measurement period</td></tr>
<tr><td>Counterparty Credit Adjustment</td><td class="num">${fmtPct(params.creditAdj||0)}</td><td>Issuer credit risk assessment</td></tr>
</table>

<h2>5. Fair Value Conclusion</h2>
<table>
<tr><th>Measure</th><th>Value</th></tr>
<tr><td><strong>Fair Value (Mean)</strong></td><td class="num"><strong>${fmt(results.fairValue)}</strong></td></tr>
<tr><td>95% Confidence Interval</td><td class="num">${fmt(results.ci95[0])} — ${fmt(results.ci95[1])}</td></tr>
<tr><td>Standard Error</td><td class="num">${fmt(results.stdError)}</td></tr>
<tr><td>Monte Carlo Convergence</td><td class="num">${results.fairValue>0?(results.stdError/results.fairValue*100).toFixed(2):0}% of fair value</td></tr>
<tr><td>Probability of Any Payoff</td><td class="num">${results.probPayoff}%</td></tr>
</table>

<h2>6. Percentile Analysis</h2>
<table>
<tr><th>Percentile</th><th>Fair Value</th></tr>
${Object.entries(results.percentiles).map(([k,v])=>`<tr><td>${k.replace("p","")}th</td><td class="num">${fmt(v)}</td></tr>`).join("")}
</table>

<h2>7. Sensitivity Analysis</h2>
<p>The following table presents the sensitivity of the fair value estimate to changes in key assumptions. Each parameter was varied independently while holding all other assumptions constant.</p>
${sensitivities?Object.entries(sensitivities).map(([label,data])=>{
  const mid=Math.floor(data.length/2);
  const low=data[0],high=data[data.length-1],base=data[mid];
  return`<table><tr><th colspan="3">${label}</th></tr><tr><th>Input</th><th>Fair Value</th><th>Change from Base</th></tr>
  <tr><td class="num">${typeof low.value==="number"&&low.value<1?fmtPct(low.value):fmt(low.value)}</td><td class="num">${fmt(low.fairValue)}</td><td class="num">${fmt(low.fairValue-base.fairValue)}</td></tr>
  <tr><td class="num"><strong>${typeof base.value==="number"&&base.value<1?fmtPct(base.value):fmt(base.value)} (Base)</strong></td><td class="num"><strong>${fmt(base.fairValue)}</strong></td><td class="num">—</td></tr>
  <tr><td class="num">${typeof high.value==="number"&&high.value<1?fmtPct(high.value):fmt(high.value)}</td><td class="num">${fmt(high.fairValue)}</td><td class="num">${fmt(high.fairValue-base.fairValue)}</td></tr></table>`;
}).join(""):"<p>Not available.</p>"}

<h2>8. ASC 820 Fair Value Hierarchy</h2>
<p>The earnout liability is classified as <strong>Level 3</strong> within the fair value hierarchy. Level 3 fair value measurements are based on significant unobservable inputs including management's projected performance metrics and the estimated volatility of the underlying metric. The valuation technique (Monte Carlo simulation) and significant unobservable inputs are consistent with those used in the initial purchase price allocation and prior reporting periods.</p>

<div class="footer">
<p>This memorandum was generated by ValuProEarnout. The valuation methodology, assumptions, and conclusions documented herein are intended for use in financial reporting under ASC 805 and ASC 820. This document does not constitute investment advice.</p>
<p>ValuProEarnout — Automated Earnout Remeasurement Platform — ${date}</p>
</div>
</body></html>`;

  if(format==="pdf"||format==="both"){
    const win=window.open("","_blank");
    win.document.write(html);
    win.document.close();
    setTimeout(()=>win.print(),500);
  }
  if(format==="docx"||format==="both"){
    // Generate .doc (HTML-based, opens in Word)
    const blob=new Blob([`<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'><head><meta charset="utf-8"><!--[if gte mso 9]><xml><w:WordDocument><w:View>Print</w:View></w:WordDocument></xml><![endif]--></head><body>${html}</body></html>`],{type:"application/msword"});
    const url=URL.createObjectURL(blob);
    const a=document.createElement("a");
    a.href=url;a.download=`ValuProEarnout_Memo_${new Date().toISOString().split("T")[0]}.doc`;
    a.click();URL.revokeObjectURL(url);
  }
};

// ============================================================
// ICON COMPONENTS (inline SVG to avoid dependency issues)
// ============================================================
const Icon = ({name,size=16,color="currentColor"}) => {
  const icons={
    target:`<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>`,
    upload:`<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>`,
    file:`<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>`,
    check:`<polyline points="20 6 9 17 4 12"/>`,
    alert:`<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>`,
    refresh:`<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>`,
    download:`<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>`,
    bar:`<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>`,
    sliders:`<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>`,
    dollar:`<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>`,
    shield:`<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>`,
    sun:`<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`,
    moon:`<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>`,
    arrowRight:`<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>`,
    info:`<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>`,
    activity:`<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>`,
    layers:`<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>`,
    scale:`<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22V8"/><path d="M20 8L12 16 4 8"/>`,
    book:`<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>`,
    plus:`<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>`,
    chevronRight:`<polyline points="9 18 15 12 9 6"/>`,
    brain:`<path d="M12 2a7 7 0 0 1 7 7c0 2.4-1.2 4.5-3 5.7V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.3C6.2 13.5 5 11.4 5 9a7 7 0 0 1 7-7z"/>`,
    sparkles:`<path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z"/>`,
    fileSearch:`<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><circle cx="11.5" cy="14.5" r="2.5"/><line x1="13.25" y1="16.25" x2="15" y2="18"/>`,
    trendUp:`<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>`,
    percent:`<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>`,
    hash:`<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>`,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" dangerouslySetInnerHTML={{__html:icons[name]||""}} />;
};

// ============================================================
// COMPONENTS
// ============================================================
const AnimatedValue = ({value,prefix="$"}) => {
  const [display,setDisplay]=useState(0);
  const ref=useRef(null);
  useEffect(()=>{
    const start=display,end=value,duration=800,startTime=Date.now();
    const animate=()=>{
      const elapsed=Date.now()-startTime;
      const progress=Math.min(elapsed/duration,1);
      setDisplay(start+(end-start)*(1-Math.pow(1-progress,3)));
      if(progress<1) ref.current=requestAnimationFrame(animate);
    };
    ref.current=requestAnimationFrame(animate);
    return()=>cancelAnimationFrame(ref.current);
  },[value]);
  const formatted=Math.abs(display)>=1e6?`${(display/1e6).toFixed(2)}M`:Math.abs(display)>=1e3?`${(display/1e3).toFixed(1)}K`:display.toFixed(0);
  return <span>{prefix}{formatted}</span>;
};

const Histogram = ({data,fairValue,theme}) => {
  const maxCount=Math.max(...data.map(d=>d.count));
  const barColor=theme==="dark"?"#6366f1":"#4f46e5";
  return (
    <div style={{display:"flex",alignItems:"flex-end",gap:1,height:140,padding:"0 4px"}}>
      {data.map((bin,i)=>{
        const height=maxCount>0?(bin.count/maxCount)*100:0;
        const isFV=fairValue>=bin.x-(data[1]?.x-data[0]?.x)/2&&fairValue<bin.x+(data[1]?.x-data[0]?.x)/2;
        return <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"flex-end",height:"100%"}}>
          <div style={{width:"100%",height:`${height}%`,backgroundColor:isFV?"#f59e0b":barColor,borderRadius:"2px 2px 0 0",minHeight:bin.count>0?2:0,opacity:isFV?1:0.7,transition:"height 0.3s ease"}} />
        </div>;
      })}
    </div>
  );
};

const ParamSlider = ({label,value,onChange,min,max,step,format="number",suffix="",tooltip,theme}) => {
  const tc=theme==="dark";
  const displayVal=format==="percent"?`${(value*100).toFixed(1)}%`:format==="currency"?fmt(value):format==="years"?`${value.toFixed(1)} yrs`:`${value.toFixed(step<1?1:0)}${suffix}`;
  return (
    <div style={{marginBottom:14}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
        <span style={{fontSize:12,color:tc?"#8896b0":"#6b7280",display:"flex",alignItems:"center",gap:4}}>
          {label}
          {tooltip&&<span title={tooltip} style={{cursor:"help",opacity:0.5}}><Icon name="info" size={11}/></span>}
        </span>
        <span style={{fontSize:13,fontFamily:"'IBM Plex Mono',monospace",color:tc?"#e8edf5":"#111827",fontWeight:600}}>{displayVal}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(parseFloat(e.target.value))}
        style={{width:"100%",accentColor:"#6366f1",height:4,cursor:"pointer"}} />
      <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:tc?"#556480":"#9ca3af",fontFamily:"'IBM Plex Mono',monospace",marginTop:2}}>
        <span>{format==="percent"?fmtPct(min):format==="currency"?fmt(min):min}</span>
        <span>{format==="percent"?fmtPct(max):format==="currency"?fmt(max):max}</span>
      </div>
    </div>
  );
};

const TornadoChart = ({sensitivities,baseValue,theme}) => {
  if(!sensitivities) return null;
  const tc=theme==="dark";
  const entries=Object.entries(sensitivities).map(([label,data])=>{
    const values=data.map(d=>d.fairValue);
    return {label,min:Math.min(...values),max:Math.max(...values)};
  }).sort((a,b)=>(b.max-b.min)-(a.max-a.min));
  const globalMin=Math.min(...entries.map(e=>e.min));
  const globalMax=Math.max(...entries.map(e=>e.max));
  const range=globalMax-globalMin||1;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:8}}>
      {entries.map((entry,i)=>{
        const leftPct=((entry.min-globalMin)/range)*100;
        const widthPct=((entry.max-entry.min)/range)*100;
        const basePct=((baseValue-globalMin)/range)*100;
        return <div key={i}>
          <div style={{fontSize:11,color:tc?"#8896b0":"#6b7280",marginBottom:2}}>{entry.label}</div>
          <div style={{position:"relative",height:24,background:tc?"#1e293b":"#e2e8f0",borderRadius:4}}>
            <div style={{position:"absolute",left:`${leftPct}%`,width:`${widthPct}%`,height:"100%",background:"linear-gradient(90deg,#ef4444,#6366f1,#22c55e)",borderRadius:4,opacity:0.8}} />
            <div style={{position:"absolute",left:`${basePct}%`,top:0,bottom:0,width:2,background:"#f59e0b"}} />
            <div style={{position:"absolute",left:4,top:4,fontSize:10,color:tc?"#e8edf5":"#111827",fontFamily:"'IBM Plex Mono',monospace"}}>{fmt(entry.min)}</div>
            <div style={{position:"absolute",right:4,top:4,fontSize:10,color:tc?"#e8edf5":"#111827",fontFamily:"'IBM Plex Mono',monospace"}}>{fmt(entry.max)}</div>
          </div>
        </div>;
      })}
    </div>
  );
};

// ============================================================
// MAIN APPLICATION
// ============================================================
export default function ValuProEarnout(){
  const [theme,setTheme]=useState("light");
  const [view,setView]=useState("landing"); // landing, mode_select, backtest_upload, live_upload, processing, results
  const [mode,setMode]=useState(null); // "backtest" or "live"
  
  // Document state
  const [files,setFiles]=useState([]);
  const [docText,setDocText]=useState("");
  const [extractedData,setExtractedData]=useState(null);
  const [verification,setVerification]=useState(null);
  const [progress,setProgress]=useState(0);
  const [stage,setStage]=useState("");
  
  // Earnout params
  const [params,setParams]=useState({
    metric:"Adjusted EBITDA",structure:"linear",
    threshold:18e6,participationRate:0.30,cap:5e6,floor:0,
    projectedMetric:20e6,volatility:0.40,discountRate:0.12,
    riskFreeRate:0.043,creditAdj:0.01,timeToMaturity:2.0,
    tiers:[{lower:18e6,upper:22e6,rate:0.20},{lower:22e6,upper:28e6,rate:0.30},{lower:28e6,upper:Infinity,rate:0.40}],
  });
  
  // Results
  const [results,setResults]=useState(null);
  const [sensitivities,setSensitivities]=useState(null);
  const [backtestComparison,setBacktestComparison]=useState(null);
  const [isComputing,setIsComputing]=useState(false);
  
  // Theme colors — refined palette
  const tc=theme==="dark";
  const c={
    bg:tc?"#0c1222":"#f8f9fb",
    card:tc?"#151e30":"#ffffff",
    cardBorder:tc?"#1e2d4a":"#e5e7eb",
    cardShadow:tc?"none":"0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)",
    accent:"#2563eb",
    accentLight:tc?"rgba(37,99,235,0.12)":"rgba(37,99,235,0.06)",
    accentHover:tc?"rgba(37,99,235,0.2)":"rgba(37,99,235,0.1)",
    success:"#059669",warning:"#d97706",danger:"#dc2626",
    text:tc?"#e8edf5":"#111827",
    textSecondary:tc?"#a3b1cc":"#374151",
    textMuted:tc?"#8896b0":"#6b7280",
    textDim:tc?"#556480":"#9ca3af",
    headerBg:tc?"rgba(12,18,34,0.85)":"rgba(255,255,255,0.92)",
    inputBg:tc?"#1a2540":"#f3f4f6",
  };
  const cardStyle={background:c.card,border:`1px solid ${c.cardBorder}`,borderRadius:12,padding:22,position:"relative",boxShadow:c.cardShadow};

  // Build payoff function from params
  const buildPayoffFn = useCallback((p=params)=>{
    return (terminalValue)=>{
      let payoff=0;
      if(p.structure==="linear") payoff=PayoffBlocks.linear(terminalValue,p.threshold,p.participationRate);
      else if(p.structure==="binary") payoff=PayoffBlocks.binary(terminalValue,p.threshold,p.cap||p.threshold*p.participationRate);
      else if(p.structure==="tiered"&&p.tiers) payoff=PayoffBlocks.tiered(terminalValue,p.tiers);
      else if(p.structure==="milestone") payoff=PayoffBlocks.binary(terminalValue,p.threshold,p.cap||1e6);
      if(p.cap) payoff=PayoffBlocks.capped(payoff,p.cap);
      if(p.floor) payoff=PayoffBlocks.floored(payoff,p.floor);
      return payoff;
    };
  },[params]);

  // Run valuation
  const runValuation = useCallback((p=params)=>{
    setIsComputing(true);
    setTimeout(()=>{
      const payoffFn=buildPayoffFn(p);
      const res=runMonteCarlo({S0:p.projectedMetric,mu:0.05,sigma:p.volatility,T:p.timeToMaturity,riskFreeRate:p.riskFreeRate,numPaths:MC_PATHS,numSteps:MC_STEPS,payoffFn,discountRate:p.discountRate,creditAdj:p.creditAdj||0});
      setResults(res);
      
      // Sensitivity
      const sens={};
      sens["Volatility"]=runSensitivityGrid({...p},(tv)=>{let po=0;if(p.structure==="linear")po=PayoffBlocks.linear(tv,p.threshold,p.participationRate);else if(p.structure==="tiered"&&p.tiers)po=PayoffBlocks.tiered(tv,p.tiers);else if(p.structure==="binary")po=PayoffBlocks.binary(tv,p.threshold,p.cap||p.threshold*p.participationRate);if(p.cap)po=Math.min(po,p.cap);return po;},"volatility",[0.15,0.75]);
      sens["Discount Rate"]=runSensitivityGrid({...p},(tv)=>{let po=0;if(p.structure==="linear")po=PayoffBlocks.linear(tv,p.threshold,p.participationRate);else if(p.structure==="tiered"&&p.tiers)po=PayoffBlocks.tiered(tv,p.tiers);else if(p.structure==="binary")po=PayoffBlocks.binary(tv,p.threshold,p.cap||p.threshold*p.participationRate);if(p.cap)po=Math.min(po,p.cap);return po;},"discountRate",[0.06,0.20]);
      sens["Projected Metric"]=runSensitivityGrid({...p},(tv)=>{let po=0;if(p.structure==="linear")po=PayoffBlocks.linear(tv,p.threshold,p.participationRate);else if(p.structure==="tiered"&&p.tiers)po=PayoffBlocks.tiered(tv,p.tiers);else if(p.structure==="binary")po=PayoffBlocks.binary(tv,p.threshold,p.cap||p.threshold*p.participationRate);if(p.cap)po=Math.min(po,p.cap);return po;},"projectedMetric",[p.projectedMetric*0.5,p.projectedMetric*1.5]);
      setSensitivities(sens);
      setIsComputing(false);
    },100);
  },[params,buildPayoffFn]);

  // File handling
  const handleFileUpload = (e) => {
    const newFiles=Array.from(e.target.files);
    setFiles(prev=>[...prev,...newFiles]);
    newFiles.forEach(file=>{
      const reader=new FileReader();
      reader.onload=(ev)=>setDocText(prev=>prev+"\n"+ev.target.result);
      reader.readAsText(file);
    });
  };

  // Full pipeline
  const runPipeline = async () => {
    setView("processing");setProgress(0);
    
    setStage("Analyzing document structure...");setProgress(10);
    await new Promise(r=>setTimeout(r,400));
    
    setStage(mode==="backtest"?"Extracting earnout disclosures from SEC filing...":"Extracting earnout terms from valuation report...");
    setProgress(25);
    
    const extractMode=mode==="backtest"?"backtest":"live_ppa";
    const extracted=await extractFromDocument(docText,extractMode);
    setExtractedData(extracted);
    setProgress(50);
    
    setStage("Running adversarial verification...");
    const verif=await verifyExtraction(docText,extracted);
    setVerification(verif);
    setProgress(65);
    
    // Map extracted data to params
    setStage("Configuring valuation model...");
    setProgress(75);
    
    if(mode==="backtest"&&extracted?.earnouts?.length>0){
      const e=extracted.earnouts[0];
      const newParams={...params,
        metric:e.metric||params.metric,
        structure:e.structure||params.structure,
        threshold:e.threshold||params.threshold,
        participationRate:e.participationRate||params.participationRate,
        cap:e.maxPayout||e.cap||params.cap,
        projectedMetric:e.projectedMetric||params.projectedMetric,
        volatility:e.volatility||params.volatility,
        discountRate:e.discountRate||params.discountRate,
        riskFreeRate:e.riskFreeRate||params.riskFreeRate,
        timeToMaturity:e.timeRemaining||params.timeToMaturity,
      };
      setParams(newParams);
      
      setStage("Running Monte Carlo simulation...");setProgress(85);
      await new Promise(r=>setTimeout(r,200));
      
      // Run and compare
      const payoffFn=buildPayoffFn(newParams);
      const res=runMonteCarlo({S0:newParams.projectedMetric,mu:0.05,sigma:newParams.volatility,T:newParams.timeToMaturity,riskFreeRate:newParams.riskFreeRate,numPaths:MC_PATHS,numSteps:MC_STEPS,payoffFn,discountRate:newParams.discountRate,creditAdj:newParams.creditAdj||0});
      setResults(res);
      
      if(e.currentFairValue||e.initialFairValue){
        const reported=e.currentFairValue||e.initialFairValue;
        setBacktestComparison({reported,computed:res.fairValue,gap:Math.abs(res.fairValue-reported)/reported*100,direction:res.fairValue>reported?"above":"below"});
      }
    } else if(mode==="live"&&extracted?.earnout){
      const e=extracted.earnout;
      const a=e.assumptions||{};
      const newParams={...params,
        metric:e.metric||params.metric,
        structure:e.structure||params.structure,
        threshold:e.thresholds?.[0]?.level||params.threshold,
        participationRate:e.participationRate||params.participationRate,
        cap:e.cap||params.cap,
        projectedMetric:a.projectedMetric||params.projectedMetric,
        volatility:a.volatility||params.volatility,
        discountRate:a.discountRate||params.discountRate,
        riskFreeRate:a.riskFreeRate||params.riskFreeRate,
        creditAdj:a.creditAdjustment||params.creditAdj,
        timeToMaturity:params.timeToMaturity,
      };
      setParams(newParams);
      setStage("Running Monte Carlo simulation...");setProgress(85);
      await new Promise(r=>setTimeout(r,200));
      runValuation(newParams);
    }
    
    setStage("Generating sensitivity analysis...");setProgress(95);
    await new Promise(r=>setTimeout(r,300));
    
    // Run sensitivity for backtest too
    if(mode==="backtest"&&results){
      const p=params;
      const makePf=(tv)=>{let po=0;if(p.structure==="linear")po=PayoffBlocks.linear(tv,p.threshold,p.participationRate);else if(p.structure==="tiered"&&p.tiers)po=PayoffBlocks.tiered(tv,p.tiers);if(p.cap)po=Math.min(po,p.cap);return po;};
      const sens={};
      sens["Volatility"]=runSensitivityGrid({...p},makePf,"volatility",[0.15,0.75]);
      sens["Discount Rate"]=runSensitivityGrid({...p},makePf,"discountRate",[0.06,0.20]);
      sens["Projected Metric"]=runSensitivityGrid({...p},makePf,"projectedMetric",[p.projectedMetric*0.5,p.projectedMetric*1.5]);
      setSensitivities(sens);
    }
    
    setProgress(100);setStage("Complete");
    setTimeout(()=>setView("results"),400);
  };

  // ============================================================
  // RENDER
  // ============================================================
  const fontLink=<><link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet"/><style>{`*{box-sizing:border-box}body{font-family:'Inter',system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}::selection{background:${c.accent};color:white}`}</style></>;
  
  const themeToggle=(
    <button onClick={()=>setTheme(t=>t==="dark"?"light":"dark")} style={{padding:"6px 12px",background:c.inputBg,border:`1px solid ${c.cardBorder}`,borderRadius:8,color:c.textMuted,cursor:"pointer",display:"flex",alignItems:"center",gap:5,fontSize:12,fontWeight:500,fontFamily:"'Inter',sans-serif",transition:"all 0.15s"}}>
      <Icon name={tc?"sun":"moon"} size={13}/>{tc?"Light":"Dark"}
    </button>
  );

  const header=(showBack=false)=>(
    <header style={{padding:"12px 36px",display:"flex",justifyContent:"space-between",alignItems:"center",borderBottom:`1px solid ${c.cardBorder}`,background:c.headerBg,backdropFilter:"blur(16px)",position:"sticky",top:0,zIndex:100}}>
      <div style={{display:"flex",alignItems:"center",gap:14}}>
        {showBack&&<button onClick={()=>{setView("landing");setResults(null);setSensitivities(null);setExtractedData(null);setDocText("");setFiles([]);setBacktestComparison(null);}} style={{background:"none",border:"none",color:c.textMuted,cursor:"pointer",padding:4,transform:"rotate(180deg)"}}><Icon name="chevronRight" size={16}/></button>}
        <div style={{display:"flex",alignItems:"center",gap:10,cursor:"pointer"}} onClick={()=>setView("landing")}>
          <div style={{width:30,height:30,borderRadius:8,background:"linear-gradient(135deg,#2563eb,#1d4ed8)",display:"flex",alignItems:"center",justifyContent:"center"}}><Icon name="scale" size={15} color="white"/></div>
          <span style={{fontSize:16,fontWeight:600,color:c.text,letterSpacing:"-0.3px"}}>ValuPro<span style={{color:c.accent}}>Earnout</span></span>
        </div>
      </div>
      <div style={{display:"flex",gap:8,alignItems:"center"}}>
        {mode&&<span style={{padding:"4px 12px",borderRadius:6,background:mode==="backtest"?tc?"rgba(5,150,105,0.12)":"rgba(5,150,105,0.06)":c.accentLight,fontSize:11,color:mode==="backtest"?c.success:c.accent,fontWeight:500,letterSpacing:"0.01em"}}>{mode==="backtest"?"Backtest":"Live Valuation"}</span>}
        {themeToggle}
      </div>
    </header>
  );

  // ---- LANDING ----
  if(view==="landing"){
    return (
      <div style={{minHeight:"100vh",background:c.bg,fontFamily:"'Inter',system-ui,sans-serif",color:c.text}}>
        {fontLink}{header()}
        <main style={{maxWidth:760,margin:"0 auto",padding:"100px 36px 60px",textAlign:"center"}}>
          <div style={{display:"inline-flex",padding:"5px 14px",borderRadius:6,background:c.accentLight,border:`1px solid ${tc?"rgba(37,99,235,0.2)":"rgba(37,99,235,0.12)"}`,fontSize:11,fontWeight:500,color:c.accent,marginBottom:28,gap:5,alignItems:"center",letterSpacing:"0.02em"}}>
            <Icon name="sparkles" size={12} color={c.accent}/> Quarterly Earnout Remeasurement — Automated
          </div>
          <h1 style={{fontSize:44,fontWeight:700,lineHeight:1.12,marginBottom:18,letterSpacing:"-1.2px",fontFamily:"'Source Serif 4','Georgia',serif",color:c.text}}>
            Stop overpaying for<br/>
            <span style={{color:c.accent}}>
              a quarterly number refresh
            </span>
          </h1>
          <p style={{fontSize:16,color:c.textMuted,maxWidth:520,margin:"0 auto",lineHeight:1.75,marginBottom:56,fontWeight:400}}>
            Upload your PPA valuation report once. Update your forecast each quarter. 
            Get an audit-ready fair value with methodology memo in minutes.
          </p>
          
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,maxWidth:620,margin:"0 auto 56px"}}>
            <div onClick={()=>{setMode("backtest");setView("backtest_upload");}} style={{...cardStyle,cursor:"pointer",padding:30,textAlign:"left",transition:"all 0.2s",borderColor:c.cardBorder}} onMouseEnter={e=>{e.currentTarget.style.borderColor=c.success;e.currentTarget.style.boxShadow=`0 4px 12px rgba(5,150,105,0.08)`;}} onMouseLeave={e=>{e.currentTarget.style.borderColor=c.cardBorder;e.currentTarget.style.boxShadow=c.cardShadow;}}>
              <div style={{width:36,height:36,borderRadius:8,background:tc?"rgba(5,150,105,0.12)":"rgba(5,150,105,0.06)",display:"flex",alignItems:"center",justifyContent:"center",color:c.success,marginBottom:16}}>
                <Icon name="target" size={18} color={c.success}/>
              </div>
              <h3 style={{fontSize:15,fontWeight:600,marginBottom:6,color:c.text,letterSpacing:"-0.2px"}}>Backtest Engine</h3>
              <p style={{fontSize:13,color:c.textMuted,lineHeight:1.65,margin:0}}>
                Upload SEC 10-K/10-Q filings. Extract earnout disclosures and validate against reported fair values.
              </p>
              <div style={{display:"flex",gap:6,marginTop:14}}>
                <span style={{padding:"3px 8px",borderRadius:4,background:tc?"rgba(5,150,105,0.08)":"rgba(5,150,105,0.04)",fontSize:10,color:c.success,fontWeight:500}}>10-K / 10-Q</span>
                <span style={{padding:"3px 8px",borderRadius:4,background:tc?"rgba(5,150,105,0.08)":"rgba(5,150,105,0.04)",fontSize:10,color:c.success,fontWeight:500}}>Validation</span>
              </div>
            </div>
            
            <div onClick={()=>{setMode("live");setView("live_upload");}} style={{...cardStyle,cursor:"pointer",padding:30,textAlign:"left",transition:"all 0.2s",borderColor:c.cardBorder}} onMouseEnter={e=>{e.currentTarget.style.borderColor=c.accent;e.currentTarget.style.boxShadow=`0 4px 12px rgba(37,99,235,0.08)`;}} onMouseLeave={e=>{e.currentTarget.style.borderColor=c.cardBorder;e.currentTarget.style.boxShadow=c.cardShadow;}}>
              <div style={{width:36,height:36,borderRadius:8,background:c.accentLight,display:"flex",alignItems:"center",justifyContent:"center",color:c.accent,marginBottom:16}}>
                <Icon name="activity" size={18} color={c.accent}/>
              </div>
              <h3 style={{fontSize:15,fontWeight:600,marginBottom:6,color:c.text,letterSpacing:"-0.2px"}}>Live Valuation</h3>
              <p style={{fontSize:13,color:c.textMuted,lineHeight:1.65,margin:0}}>
                Upload your PPA report and latest forecast. Get the updated fair value with audit-ready deliverables.
              </p>
              <div style={{display:"flex",gap:6,marginTop:14}}>
                <span style={{padding:"3px 8px",borderRadius:4,background:c.accentLight,fontSize:10,color:c.accent,fontWeight:500}}>PPA Report</span>
                <span style={{padding:"3px 8px",borderRadius:4,background:c.accentLight,fontSize:10,color:c.accent,fontWeight:500}}>Quarterly Update</span>
              </div>
            </div>
          </div>
          
          <div style={{display:"flex",gap:40,justifyContent:"center",marginBottom:24}}>
            {[{n:"50,000",l:"MC Paths"},{n:"< 30 min",l:"Turnaround"},{n:"ASC 820",l:"Compliant"},{n:"Level 3",l:"Audit-Ready"}].map((s,i)=>(
              <div key={i} style={{textAlign:"center"}}>
                <div style={{fontSize:16,fontWeight:600,color:c.accent,fontFamily:"'IBM Plex Mono',monospace",letterSpacing:"-0.3px"}}>{s.n}</div>
                <div style={{fontSize:11,color:c.textDim,marginTop:2,fontWeight:400}}>{s.l}</div>
              </div>
            ))}
          </div>
        </main>
        <footer style={{padding:"18px 36px",borderTop:`1px solid ${c.cardBorder}`,display:"flex",justifyContent:"space-between",fontSize:11,color:c.textDim}}>
          <span>ValuProEarnout</span>
          <div style={{display:"flex",gap:6,alignItems:"center"}}><Icon name="shield" size={11} color={c.textDim}/> End-to-end encrypted</div>
        </footer>
      </div>
    );
  }

  // ---- UPLOAD SCREENS ----
  if(view==="backtest_upload"||view==="live_upload"){
    const isBacktest=view==="backtest_upload";
    const requiredDocs=isBacktest?[
      {label:"SEC 10-K Filing",desc:"Annual report with earnout fair value disclosures, Level 3 rollforward, and methodology notes",accept:".txt,.htm,.html"},
      {label:"SEC 10-Q Filing (optional)",desc:"Quarterly report for additional remeasurement data points",accept:".txt,.htm,.html"},
    ]:[
      {label:"PPA Valuation Report",desc:"Initial earnout valuation from your PPA firm (Kroll, Duff & Phelps, etc.) containing methodology, assumptions, and fair value",accept:".txt,.pdf,.doc,.docx"},
      {label:"Management Forecast Update",desc:"Latest management projections for the earnout metric (board-approved budget, financial model, or forecast memo)",accept:".txt,.pdf,.xlsx,.csv"},
    ];
    
    return (
      <div style={{minHeight:"100vh",background:c.bg,fontFamily:"'Inter',system-ui,sans-serif",color:c.text}}>
        {fontLink}{header(true)}
        <div style={{maxWidth:640,margin:"48px auto",padding:"0 32px"}}>
          <h2 style={{fontSize:28,fontWeight:700,marginBottom:6,fontFamily:"'Source Serif 4','Georgia',serif"}}>{isBacktest?"Upload SEC Filing":"Upload Valuation Documents"}</h2>
          <p style={{color:c.textMuted,marginBottom:32,fontSize:14}}>{isBacktest?"Upload 10-K or 10-Q filings to extract earnout disclosures and validate ValuPro's engine against reported fair values.":"Upload the initial PPA valuation report and latest forecast to run your quarterly remeasurement."}</p>
          
          {requiredDocs.map((doc,i)=>(
            <div key={i} style={{marginBottom:16}}>
              <div style={{fontSize:13,fontWeight:600,color:c.text,marginBottom:6}}>{doc.label}</div>
              <label style={{...cardStyle,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",minHeight:100,cursor:"pointer",borderStyle:"dashed",borderWidth:2,transition:"border-color 0.2s"}}>
                <input type="file" accept={doc.accept} onChange={handleFileUpload} style={{display:"none"}} />
                <Icon name="upload" size={24} color={c.accent}/>
                <span style={{fontSize:12,color:c.textMuted,marginTop:8}}>{doc.desc}</span>
              </label>
            </div>
          ))}
          
          {files.length>0&&(
            <div style={{marginTop:12}}>
              {files.map((f,i)=>(
                <div key={i} style={{display:"flex",alignItems:"center",gap:10,padding:"8px 14px",background:c.accentLight,borderRadius:8,marginBottom:6}}>
                  <Icon name="file" size={14} color={c.accent}/>
                  <span style={{fontSize:12,flex:1,color:c.text}}>{f.name}</span>
                  <span style={{fontSize:11,color:c.textDim}}>{(f.size/1024).toFixed(1)} KB</span>
                  <Icon name="check" size={14} color={c.success}/>
                </div>
              ))}
            </div>
          )}
          
          <div style={{display:"flex",gap:12,marginTop:28}}>
            <button onClick={()=>{setView("landing");setMode(null);setFiles([]);setDocText("");}} style={{padding:"11px 22px",background:"transparent",border:`1px solid ${c.cardBorder}`,borderRadius:10,color:c.text,cursor:"pointer",fontSize:13,fontFamily:"'Inter',system-ui,sans-serif"}}>Back</button>
            <button onClick={runPipeline} disabled={!docText} style={{flex:1,padding:"11px 22px",background:docText?"linear-gradient(135deg,#2563eb,#1d4ed8)":c.cardBorder,border:"none",borderRadius:10,color:"white",cursor:docText?"pointer":"not-allowed",fontSize:14,fontWeight:600,fontFamily:"'Inter',system-ui,sans-serif",display:"flex",alignItems:"center",justifyContent:"center",gap:8}}>
              <Icon name="brain" size={16} color="white"/>{isBacktest?"Run Backtest":"Run Valuation"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---- PROCESSING ----
  if(view==="processing"){
    return (
      <div style={{minHeight:"100vh",background:c.bg,fontFamily:"'Inter',system-ui,sans-serif",color:c.text}}>
        {fontLink}{header(false)}
        <div style={{maxWidth:560,margin:"80px auto",padding:"0 32px",textAlign:"center"}}>
          <div style={{width:72,height:72,borderRadius:"50%",background:c.accentLight,display:"flex",alignItems:"center",justifyContent:"center",margin:"0 auto 28px"}}>
            <div style={{animation:"spin 2s linear infinite"}}><Icon name="refresh" size={28} color={c.accent}/></div>
          </div>
          <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
          <h2 style={{fontSize:22,fontWeight:700,marginBottom:6,fontFamily:"'Source Serif 4','Georgia',serif"}}>{mode==="backtest"?"Running Backtest":"Processing Valuation"}</h2>
          <p style={{fontSize:14,color:c.accent,marginBottom:28,fontWeight:500}}>{stage}</p>
          <div style={{width:"100%",height:5,background:c.cardBorder,borderRadius:100,overflow:"hidden",marginBottom:36}}>
            <div style={{width:`${progress}%`,height:"100%",background:"linear-gradient(90deg,#6366f1,#a78bfa)",borderRadius:100,transition:"width 0.5s ease"}} />
          </div>
          <div style={{textAlign:"left"}}>
            {[
              {label:"Document ingestion",threshold:10},
              {label:mode==="backtest"?"Extract earnout disclosures from filing":"Extract terms from PPA report",threshold:25},
              {label:"Adversarial verification",threshold:50},
              {label:"Configure valuation model",threshold:65},
              {label:`Monte Carlo simulation (${MC_PATHS.toLocaleString()} paths)`,threshold:85},
              {label:"Sensitivity analysis",threshold:95},
            ].map((s,i)=>(
              <div key={i} style={{display:"flex",alignItems:"center",gap:10,padding:"6px 0",opacity:progress>=s.threshold?1:0.3}}>
                {progress>s.threshold+5?<Icon name="check" size={15} color={c.success}/>:progress>=s.threshold?<div style={{width:15,height:15,borderRadius:"50%",border:`2px solid ${c.accent}`,borderTopColor:"transparent",animation:"spin 1s linear infinite"}}/>:<div style={{width:15,height:15,borderRadius:"50%",border:`1px solid ${c.textDim}`}}/>}
                <span style={{fontSize:13,color:progress>=s.threshold?c.text:c.textDim}}>{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ---- RESULTS ----
  if(view==="results"&&results){
    return (
      <div style={{minHeight:"100vh",background:c.bg,fontFamily:"'Inter',system-ui,sans-serif",color:c.text}}>
        {fontLink}{header(true)}
        <div style={{padding:"28px 32px",maxWidth:1300,margin:"0 auto"}}>
          {/* Header */}
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:24}}>
            <div>
              <h2 style={{fontSize:24,fontWeight:700,marginBottom:4,fontFamily:"'Source Serif 4','Georgia',serif"}}>{mode==="backtest"?"Backtest Results":"Earnout Remeasurement"}</h2>
              <p style={{fontSize:13,color:c.textMuted}}>{params.metric} • {new Date().toLocaleDateString("en-US",{year:"numeric",month:"long",day:"numeric"})}</p>
            </div>
            <div style={{display:"flex",gap:8}}>
              <button onClick={()=>generateExcel(params,results,sensitivities)} style={{padding:"9px 16px",background:"transparent",border:`1px solid ${c.cardBorder}`,borderRadius:8,color:c.text,cursor:"pointer",fontSize:12,fontFamily:"'Inter',system-ui,sans-serif",display:"flex",alignItems:"center",gap:6}}><Icon name="download" size={14}/>Excel</button>
              <button onClick={()=>generateMemo(params,results,sensitivities,"pdf")} style={{padding:"9px 16px",background:"transparent",border:`1px solid ${c.cardBorder}`,borderRadius:8,color:c.text,cursor:"pointer",fontSize:12,fontFamily:"'Inter',system-ui,sans-serif",display:"flex",alignItems:"center",gap:6}}><Icon name="file" size={14}/>PDF Memo</button>
              <button onClick={()=>generateMemo(params,results,sensitivities,"docx")} style={{padding:"9px 16px",background:"transparent",border:`1px solid ${c.cardBorder}`,borderRadius:8,color:c.text,cursor:"pointer",fontSize:12,fontFamily:"'Inter',system-ui,sans-serif",display:"flex",alignItems:"center",gap:6}}><Icon name="file" size={14}/>Word Memo</button>
              <button onClick={()=>{setView("landing");setResults(null);setSensitivities(null);setExtractedData(null);setDocText("");setFiles([]);setBacktestComparison(null);setMode(null);}} style={{padding:"9px 16px",background:"transparent",border:`1px solid ${c.cardBorder}`,borderRadius:8,color:c.text,cursor:"pointer",fontSize:12,fontFamily:"'Inter',system-ui,sans-serif",display:"flex",alignItems:"center",gap:6}}><Icon name="plus" size={14}/>New</button>
            </div>
          </div>

          {/* Backtest comparison banner */}
          {backtestComparison&&(
            <div style={{...cardStyle,marginBottom:16,padding:18,background:backtestComparison.gap<5?"rgba(34,197,94,0.06)":backtestComparison.gap<10?"rgba(245,158,11,0.06)":"rgba(239,68,68,0.06)",borderColor:backtestComparison.gap<5?"rgba(34,197,94,0.2)":backtestComparison.gap<10?"rgba(245,158,11,0.2)":"rgba(239,68,68,0.2)"}}>
              <div style={{display:"flex",alignItems:"center",gap:16}}>
                <div style={{flex:1}}>
                  <div style={{fontSize:13,fontWeight:600,color:backtestComparison.gap<5?c.success:backtestComparison.gap<10?c.warning:c.danger,marginBottom:4}}>
                    <Icon name={backtestComparison.gap<5?"check":"alert"} size={14} color={backtestComparison.gap<5?c.success:c.warning}/>{" "}
                    Backtest Result: {backtestComparison.gap.toFixed(1)}% {backtestComparison.direction} reported value
                  </div>
                  <div style={{fontSize:12,color:c.textMuted}}>
                    Company reported {fmt(backtestComparison.reported)} — ValuPro computed {fmt(backtestComparison.computed)}
                  </div>
                </div>
                <div style={{display:"flex",gap:20,fontSize:12}}>
                  <div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:700,fontFamily:"'IBM Plex Mono',monospace",color:c.textMuted}}>{fmt(backtestComparison.reported)}</div><div style={{color:c.textDim}}>Reported</div></div>
                  <div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:700,fontFamily:"'IBM Plex Mono',monospace",color:c.accent}}>{fmt(backtestComparison.computed)}</div><div style={{color:c.textDim}}>ValuPro</div></div>
                  <div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:700,fontFamily:"'IBM Plex Mono',monospace",color:backtestComparison.gap<5?c.success:c.warning}}>{backtestComparison.gap.toFixed(1)}%</div><div style={{color:c.textDim}}>Gap</div></div>
                </div>
              </div>
            </div>
          )}

          <div style={{display:"grid",gridTemplateColumns:"300px 1fr",gap:20}}>
            {/* LEFT: Params */}
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {/* Extracted terms */}
              {extractedData&&(
                <div style={cardStyle}>
                  <h3 style={{fontSize:13,fontWeight:600,marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                    <Icon name="fileSearch" size={14} color={c.accent}/> Extracted Terms
                    {verification&&<span style={{marginLeft:"auto",padding:"2px 8px",borderRadius:100,fontSize:10,background:verification.overallConfidence>0.8?"rgba(34,197,94,0.1)":"rgba(245,158,11,0.1)",color:verification.overallConfidence>0.8?c.success:c.warning}}>{(verification.overallConfidence*100).toFixed(0)}% confidence</span>}
                  </h3>
                  <div style={{fontSize:12,color:c.textMuted,display:"flex",flexDirection:"column",gap:6}}>
                    <div><span style={{color:c.textDim}}>Metric:</span> <span style={{color:c.text}}>{params.metric}</span></div>
                    <div><span style={{color:c.textDim}}>Structure:</span> <span style={{color:c.text,textTransform:"capitalize"}}>{params.structure}</span></div>
                    <div><span style={{color:c.textDim}}>Threshold:</span> <span style={{color:c.text}}>{fmt(params.threshold)}</span></div>
                    <div><span style={{color:c.textDim}}>Participation:</span> <span style={{color:c.text}}>{fmtPct(params.participationRate)}</span></div>
                    <div><span style={{color:c.textDim}}>Cap:</span> <span style={{color:c.text}}>{params.cap?fmt(params.cap):"None"}</span></div>
                  </div>
                </div>
              )}
              
              {/* Sliders */}
              <div style={cardStyle}>
                <h3 style={{fontSize:13,fontWeight:600,marginBottom:16,display:"flex",alignItems:"center",gap:6}}>
                  <Icon name="sliders" size={14} color={c.accent}/> Assumptions
                </h3>
                <ParamSlider theme={theme} label="Projected Metric" value={params.projectedMetric} onChange={v=>setParams(p=>({...p,projectedMetric:v}))} min={5e6} max={50e6} step={500000} format="currency" tooltip="Management's current forecast"/>
                <ParamSlider theme={theme} label="Volatility" value={params.volatility} onChange={v=>setParams(p=>({...p,volatility:v}))} min={0.10} max={0.80} step={0.01} format="percent" tooltip="Comparable company equity volatility"/>
                <ParamSlider theme={theme} label="Discount Rate" value={params.discountRate} onChange={v=>setParams(p=>({...p,discountRate:v}))} min={0.05} max={0.25} step={0.005} format="percent" tooltip="Risk-adjusted discount rate"/>
                <ParamSlider theme={theme} label="Risk-Free Rate" value={params.riskFreeRate} onChange={v=>setParams(p=>({...p,riskFreeRate:v}))} min={0.01} max={0.08} step={0.001} format="percent"/>
                <ParamSlider theme={theme} label="Credit Risk Adj." value={params.creditAdj||0} onChange={v=>setParams(p=>({...p,creditAdj:v}))} min={0} max={0.05} step={0.005} format="percent" tooltip="Counterparty credit risk adjustment"/>
                <ParamSlider theme={theme} label="Time Remaining" value={params.timeToMaturity} onChange={v=>setParams(p=>({...p,timeToMaturity:v}))} min={0.25} max={5} step={0.25} format="years"/>
                <ParamSlider theme={theme} label="Threshold" value={params.threshold} onChange={v=>setParams(p=>({...p,threshold:v}))} min={5e6} max={50e6} step={500000} format="currency" tooltip="Earnout metric target"/>
                <ParamSlider theme={theme} label="Participation Rate" value={params.participationRate} onChange={v=>setParams(p=>({...p,participationRate:v}))} min={0.05} max={1.0} step={0.05} format="percent"/>
                <ParamSlider theme={theme} label="Cap" value={params.cap||0} onChange={v=>setParams(p=>({...p,cap:v||null}))} min={0} max={20e6} step={500000} format="currency" tooltip="Maximum earnout payout"/>
                
                <button onClick={()=>runValuation(params)} style={{width:"100%",padding:"10px",background:"linear-gradient(135deg,#2563eb,#1d4ed8)",border:"none",borderRadius:8,color:"white",cursor:"pointer",fontSize:13,fontWeight:600,fontFamily:"'Inter',system-ui,sans-serif",marginTop:6,display:"flex",alignItems:"center",justifyContent:"center",gap:6}}>
                  <Icon name="refresh" size={14} color="white"/> Recalculate
                </button>
              </div>
            </div>

            {/* RIGHT: Results */}
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {/* KPIs */}
              <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10}}>
                {[
                  {label:"Fair Value",value:results.fairValue,icon:"dollar",color:c.accent},
                  {label:"95% CI Low",value:results.ci95[0],icon:"trendUp",color:c.danger},
                  {label:"95% CI High",value:results.ci95[1],icon:"trendUp",color:c.success},
                  {label:"Std Error",value:results.stdError,icon:"target",color:c.warning},
                  {label:"Prob. Payoff",value:null,display:`${results.probPayoff}%`,icon:"percent",color:c.text},
                ].map((kpi,i)=>(
                  <div key={i} style={{...cardStyle,padding:16}}>
                    <div style={{display:"flex",alignItems:"center",gap:4,marginBottom:6}}>
                      <Icon name={kpi.icon} size={13} color={kpi.color}/>
                      <span style={{fontSize:10,color:c.textMuted}}>{kpi.label}</span>
                    </div>
                    <div style={{fontSize:20,fontWeight:700,fontFamily:"'IBM Plex Mono',monospace",color:kpi.color}}>
                      {kpi.display||<AnimatedValue value={kpi.value}/>}
                    </div>
                  </div>
                ))}
              </div>

              {/* Charts */}
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                <div style={cardStyle}>
                  <h3 style={{fontSize:13,fontWeight:600,marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                    <Icon name="bar" size={14} color={c.accent}/> Value Distribution
                    <span style={{fontSize:10,color:c.textDim,fontWeight:400,marginLeft:"auto"}}>{MC_PATHS.toLocaleString()} paths</span>
                  </h3>
                  <Histogram data={results.histogram} fairValue={results.fairValue} theme={theme}/>
                  <div style={{display:"flex",justifyContent:"space-between",marginTop:6,fontSize:10,fontFamily:"'IBM Plex Mono',monospace",color:c.textDim}}>
                    <span>{fmt(results.histogram[0]?.x)}</span>
                    <span style={{color:c.warning}}>FV: {fmt(results.fairValue)}</span>
                    <span>{fmt(results.histogram[results.histogram.length-1]?.x)}</span>
                  </div>
                </div>

                <div style={cardStyle}>
                  <h3 style={{fontSize:13,fontWeight:600,marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                    <Icon name="activity" size={14} color={c.accent}/> Sensitivity (Tornado)
                  </h3>
                  <TornadoChart sensitivities={sensitivities} baseValue={results.fairValue} theme={theme}/>
                </div>
              </div>

              {/* Percentiles */}
              <div style={cardStyle}>
                <h3 style={{fontSize:13,fontWeight:600,marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                  <Icon name="hash" size={14} color={c.accent}/> Percentile Analysis
                </h3>
                <div style={{display:"grid",gridTemplateColumns:"repeat(7,1fr)",gap:8}}>
                  {Object.entries(results.percentiles).map(([key,val])=>{
                    const pct=parseInt(key.replace("p",""));
                    return (
                      <div key={key} style={{textAlign:"center"}}>
                        <div style={{fontSize:10,color:c.textDim,marginBottom:4}}>{pct}th</div>
                        <div style={{height:60,display:"flex",alignItems:"flex-end",justifyContent:"center",marginBottom:4}}>
                          <div style={{width:"100%",maxWidth:36,height:`${Math.min(100,(val/(results.percentiles.p95||1))*100)}%`,background:`linear-gradient(180deg,${c.accent},rgba(99,102,241,0.3))`,borderRadius:"4px 4px 0 0",minHeight:4}}/>
                        </div>
                        <div style={{fontSize:11,fontFamily:"'IBM Plex Mono',monospace",color:c.text}}>{fmt(val)}</div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Methodology */}
              <div style={{...cardStyle,background:tc?"rgba(99,102,241,0.03)":"rgba(99,102,241,0.02)",borderColor:tc?"rgba(99,102,241,0.1)":"rgba(99,102,241,0.08)"}}>
                <h3 style={{fontSize:13,fontWeight:600,marginBottom:6,display:"flex",alignItems:"center",gap:6}}>
                  <Icon name="book" size={14} color={c.accent}/> Methodology & Compliance
                </h3>
                <p style={{fontSize:12,color:c.textMuted,lineHeight:1.7,margin:0}}>
                  Fair value estimated using Monte Carlo simulation ({MC_PATHS.toLocaleString()} paths, {MC_STEPS} daily steps) under the risk-neutral framework per ASC 820 / IFRS 13. 
                  Underlying metric modeled as Geometric Brownian Motion. Discount rate ({fmtPct(params.discountRate)}) reflects earnout-specific risk premium. 
                  Volatility ({fmtPct(params.volatility)}) estimated from comparable public company equity volatility. 
                  Counterparty credit adjustment of {fmtPct(params.creditAdj||0)} applied. 
                  Level 3 classification within the fair value hierarchy.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return <div style={{minHeight:"100vh",background:c.bg,fontFamily:"'Inter',system-ui,sans-serif",color:c.text}}>{fontLink}{header()}<div style={{padding:40,textAlign:"center",color:c.textMuted}}>Loading...</div></div>;
}
