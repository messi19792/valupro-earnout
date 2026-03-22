import { useState, useEffect, useRef, useCallback } from "react";
import _ from "lodash";

// ============================================================
// CONFIG
// ============================================================
const CLAUDE_API_KEY = process.env.NEXT_PUBLIC_CLAUDE_API_KEY || "";
const CLAUDE_MODEL = "claude-sonnet-4-20250514";
const MC_PATHS = 50000;

// ============================================================
// MATH PRIMITIVES
// ============================================================
const normalRandom = () => {
  let u1; do { u1 = Math.random(); } while (u1 === 0);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
};

// Generate annual metric values using GBM (one value per year)
const generateAnnualMetrics = (S0, growthRate, sigma, numYears) => {
  const values = [S0];
  for (let y = 1; y <= numYears; y++) {
    const z = normalRandom();
    const drift = growthRate - 0.5 * sigma * sigma;
    values.push(values[y - 1] * Math.exp(drift + sigma * z));
  }
  return values; // [Year0=current, Year1, Year2, ...]
};

// Generate correlated metric pair (for multi-metric earnouts)
const generateCorrelatedMetrics = (S0_a, S0_b, growth_a, growth_b, sigma_a, sigma_b, correlation, numYears) => {
  const valuesA = [S0_a], valuesB = [S0_b];
  for (let y = 1; y <= numYears; y++) {
    const z1 = normalRandom();
    const z2 = correlation * z1 + Math.sqrt(1 - correlation * correlation) * normalRandom();
    valuesA.push(valuesA[y - 1] * Math.exp((growth_a - 0.5 * sigma_a * sigma_a) + sigma_a * z1));
    valuesB.push(valuesB[y - 1] * Math.exp((growth_b - 0.5 * sigma_b * sigma_b) + sigma_b * z2));
  }
  return { a: valuesA, b: valuesB };
};

// ============================================================
// PAYOFF STRUCTURES (per-period evaluation)
// ============================================================
const evaluatePayoff = (metricValue, period, allPeriodValues, priorPayments, config) => {
  const p = period; // current period config
  let payoff = 0;

  switch (p.structure) {
    case "linear":
      payoff = Math.max(0, (metricValue - (p.threshold || 0)) * (p.participationRate || 1));
      break;

    case "binary":
      payoff = metricValue >= (p.threshold || 0) ? (p.fixedPayment || 0) : 0;
      break;

    case "tiered":
      if (p.tiers && p.tiers.length > 0) {
        for (const tier of p.tiers) {
          if (metricValue > tier.lower) {
            const applicable = Math.min(metricValue, tier.upper || Infinity) - tier.lower;
            payoff += Math.max(0, applicable) * tier.rate;
          }
        }
      }
      break;

    case "percentage":
      // Fixed percentage of metric value (no threshold)
      payoff = metricValue * (p.participationRate || 0);
      break;

    case "cagr":
      // Based on compound annual growth rate
      if (p.baseValue && p.cagrTarget && allPeriodValues.length > 1) {
        const actualCAGR = Math.pow(metricValue / p.baseValue, 1 / (allPeriodValues.length - 1)) - 1;
        if (actualCAGR >= p.cagrTarget) {
          payoff = p.fixedPayment || 0;
        }
      }
      break;

    case "milestone":
      // Binary milestone (non-financial: regulatory, product launch, etc.)
      // Modeled as probability-weighted binary
      payoff = (Math.random() < (p.milestoneProbability || 0.5)) ? (p.fixedPayment || 0) : 0;
      break;

    default:
      payoff = Math.max(0, (metricValue - (p.threshold || 0)) * (p.participationRate || 1));
  }

  // Apply per-period cap
  if (p.cap != null && p.cap > 0) payoff = Math.min(payoff, p.cap);
  // Apply per-period floor
  if (p.floor != null && p.floor > 0) payoff = Math.max(payoff, p.floor);

  return payoff;
};

// ============================================================
// MULTI-PERIOD MONTE CARLO ENGINE
// Handles: path dependency, catch-ups, clawbacks, cumulative targets,
// carry-forwards, multi-year caps, acceleration, multi-metric
// ============================================================
const runMultiPeriodMC = (config) => {
  const {
    periods, // Array of period configs [{year, threshold, structure, cap, floor, ...}]
    currentMetric, // Current metric value (starting point for simulation)
    metricGrowthRate, // Expected annual growth rate (real-world)
    volatility,
    discountRate, // Metric-specific discount rate (e.g., 10% for EBITDA)
    riskFreeRate,
    creditAdj = 0, // Credit risk adjustment for counterparty risk
    payoffDiscountRate = null, // Risk-adjusted rate for discounting payoffs (if null, auto-calculated)
    numPaths = MC_PATHS,
    // Path dependency features
    hasCatchUp = false,
    hasCumulativeTarget = false,
    cumulativeTarget = 0,
    hasMultiYearCap = false,
    multiYearCap = 0,
    hasCarryForward = false,
    hasClawback = false,
    clawbackThreshold = 0, // If cumulative metric falls below this, buyer recovers
    clawbackRate = 0, // Rate of clawback
    clawbackCap = 0,
    // Acceleration
    hasAcceleration = false,
    accelerationProb = 0, // Annual probability of acceleration trigger
    accelerationTreatment = "max", // "max" pays full remaining, "percentile" uses projection
    accelerationPercentile = 75,
    // Multi-metric
    isMultiMetric = false,
    secondMetric = null, // {currentValue, growthRate, volatility, threshold}
    metricCorrelation = 0.5,
    // Payment timing
    paymentDelay = 0, // Days after period end
    // Escrow
    isEscrowed = false,
  } = config;

  // ---- DISCOUNT RATE FRAMEWORK ----
  // Risk-neutral simulation: drift = metricGrowthRate - (discountRate - riskFreeRate)
  //   This converts real-world growth to risk-neutral by subtracting the risk premium
  // Payoff discounting: at riskFreeRate + creditAdj (unless escrowed)
  //   This is the correct rate for discounting contingent cash flows under risk-neutral
  // If payoffDiscountRate is explicitly provided, use that instead (for flexibility)
  const metricRiskPremium = discountRate - riskFreeRate; // e.g., 10% - 2.5% = 7.5%
  const riskNeutralDrift = metricGrowthRate - metricRiskPremium; // Real growth minus risk premium
  const effectivePayoffDiscount = payoffDiscountRate != null 
    ? payoffDiscountRate 
    : riskFreeRate + (isEscrowed ? 0 : creditAdj);
  const numPeriods = periods.length;
  const allResults = []; // Total discounted payoff per simulation
  const periodResults = Array.from({ length: numPeriods }, () => []); // Per-period payoffs
  const pathData = []; // Store subset for visualization

  for (let sim = 0; sim < numPaths; sim++) {
    // Generate metric path for all periods using RISK-NEUTRAL drift
    const metricPath = generateAnnualMetrics(currentMetric, riskNeutralDrift, volatility, numPeriods);
    
    // Generate second metric if multi-metric
    let secondMetricPath = null;
    if (isMultiMetric && secondMetric) {
      const correlated = generateCorrelatedMetrics(
        currentMetric, secondMetric.currentValue,
        metricGrowthRate, secondMetric.growthRate,
        volatility, secondMetric.volatility,
        metricCorrelation, numPeriods
      );
      secondMetricPath = correlated.b;
    }

    let totalPayoff = 0;
    let cumulativeMetric = 0;
    let cumulativePayments = 0;
    let priorShortfall = 0;
    let excessCarryForward = 0;
    let accelerated = false;
    let clawbackAmount = 0;
    const periodPayoffs = [];

    for (let pIdx = 0; pIdx < numPeriods; pIdx++) {
      const period = periods[pIdx];
      const yearFromNow = period.yearFromNow || (pIdx + 1);
      let metricValue = metricPath[pIdx + 1]; // Year pIdx+1 value
      const paymentDelayYears = paymentDelay / 365;
      const discountT = yearFromNow + paymentDelayYears;

      // Check acceleration trigger
      if (hasAcceleration && !accelerated && Math.random() < accelerationProb) {
        accelerated = true;
        // Pay out remaining periods
        let acceleratedPayoff = 0;
        if (accelerationTreatment === "max") {
          // Pay maximum remaining
          for (let rIdx = pIdx; rIdx < numPeriods; rIdx++) {
            acceleratedPayoff += periods[rIdx].cap || periods[rIdx].fixedPayment || 0;
          }
        } else {
          // Pay at percentile of projections
          const projectedMetric = currentMetric * Math.pow(1 + metricGrowthRate, yearFromNow);
          const percentileMetric = projectedMetric * (accelerationPercentile / 100);
          for (let rIdx = pIdx; rIdx < numPeriods; rIdx++) {
            acceleratedPayoff += evaluatePayoff(percentileMetric, periods[rIdx], metricPath, cumulativePayments, config);
          }
        }
        if (hasMultiYearCap) acceleratedPayoff = Math.min(acceleratedPayoff, multiYearCap - cumulativePayments);
        totalPayoff += acceleratedPayoff * Math.exp(-effectivePayoffDiscount * discountT);
        periodPayoffs.push(acceleratedPayoff);
        break;
      }

      // Apply catch-up: reduce threshold by prior shortfall
      if (hasCatchUp && priorShortfall > 0) {
        metricValue = metricValue + priorShortfall;
      }

      // Apply carry-forward: add excess from prior period
      if (hasCarryForward && excessCarryForward > 0) {
        metricValue = metricValue + excessCarryForward;
        excessCarryForward = 0;
      }

      // Multi-metric check
      if (isMultiMetric && secondMetricPath && secondMetric) {
        const secondValue = secondMetricPath[pIdx + 1];
        if (secondValue < (secondMetric.threshold || 0)) {
          // Second metric not met — zero payoff for this period
          periodPayoffs.push(0);
          periodResults[pIdx].push(0);
          // Track shortfall for catch-up
          if (hasCatchUp) priorShortfall += (period.threshold || 0) - metricValue;
          continue;
        }
      }

      // Calculate period payoff
      let periodPayoff = evaluatePayoff(metricValue, period, metricPath, cumulativePayments, config);

      // Track excess for carry-forward
      if (hasCarryForward && period.cap && periodPayoff >= period.cap) {
        const uncapped = evaluatePayoff(metricValue, { ...period, cap: null }, metricPath, cumulativePayments, config);
        excessCarryForward = uncapped - periodPayoff;
      }

      // Track shortfall for catch-up
      if (hasCatchUp) {
        const target = period.threshold || 0;
        if (metricValue < target) {
          priorShortfall = target - metricValue;
        } else {
          priorShortfall = 0;
        }
      }

      // Multi-year cap enforcement
      if (hasMultiYearCap && multiYearCap > 0) {
        const remainingCap = multiYearCap - cumulativePayments;
        periodPayoff = Math.min(periodPayoff, Math.max(0, remainingCap));
      }

      // Cumulative target check (only pay if cumulative metric exceeds target)
      cumulativeMetric += metricPath[pIdx + 1];
      if (hasCumulativeTarget && pIdx === numPeriods - 1) {
        if (cumulativeMetric < cumulativeTarget) {
          periodPayoff = 0; // Cumulative target not met
        }
      }

      // Discount and accumulate
      const discountedPayoff = periodPayoff * Math.exp(-effectivePayoffDiscount * discountT);
      totalPayoff += discountedPayoff;
      cumulativePayments += periodPayoff;
      periodPayoffs.push(periodPayoff);
      periodResults[pIdx].push(discountedPayoff);
    }

    // Clawback evaluation (at end of all periods)
    if (hasClawback && cumulativeMetric < clawbackThreshold) {
      const clawback = Math.min(
        cumulativePayments * clawbackRate,
        clawbackCap > 0 ? clawbackCap : Infinity
      );
      totalPayoff -= clawback * Math.exp(-effectivePayoffDiscount * (numPeriods));
      clawbackAmount = clawback;
    }

    allResults.push(totalPayoff);
    if (sim < 100) pathData.push({ metricPath, periodPayoffs, totalPayoff });
  }

  // Sort and compute statistics
  allResults.sort((a, b) => a - b);
  const mean = allResults.reduce((a, b) => a + b, 0) / allResults.length;
  const variance = allResults.reduce((a, b) => a + (b - mean) ** 2, 0) / (allResults.length - 1);
  const stdErr = Math.sqrt(variance / allResults.length);

  // Histogram
  const numBins = 50;
  const minVal = allResults[0], maxVal = allResults[allResults.length - 1];
  const binWidth = (maxVal - minVal) / numBins || 1;
  const histogram = Array.from({ length: numBins }, (_, i) => ({
    x: minVal + i * binWidth + binWidth / 2, count: 0
  }));
  allResults.forEach(v => {
    const idx = Math.min(Math.floor((v - minVal) / binWidth), numBins - 1);
    if (idx >= 0) histogram[idx].count++;
  });

  // Per-period statistics
  const periodStats = periodResults.map((pr, i) => {
    if (pr.length === 0) return { mean: 0, p25: 0, p50: 0, p75: 0 };
    pr.sort((a, b) => a - b);
    return {
      mean: pr.reduce((a, b) => a + b, 0) / pr.length,
      p25: pr[Math.floor(0.25 * pr.length)],
      p50: pr[Math.floor(0.50 * pr.length)],
      p75: pr[Math.floor(0.75 * pr.length)],
    };
  });

  // Convergence
  const convergence = [];
  let sum = 0;
  const step = Math.max(1, Math.floor(allResults.length / 100));
  for (let i = 0; i < allResults.length; i++) {
    sum += allResults[i];
    if (i % step === 0 && i > 0) convergence.push({ n: i, mean: sum / (i + 1) });
  }

  return {
    fairValue: mean,
    stdError: stdErr,
    ci95: [mean - 1.96 * stdErr, mean + 1.96 * stdErr],
    percentiles: {
      p5: allResults[Math.floor(0.05 * allResults.length)],
      p10: allResults[Math.floor(0.10 * allResults.length)],
      p25: allResults[Math.floor(0.25 * allResults.length)],
      p50: allResults[Math.floor(0.50 * allResults.length)],
      p75: allResults[Math.floor(0.75 * allResults.length)],
      p90: allResults[Math.floor(0.90 * allResults.length)],
      p95: allResults[Math.floor(0.95 * allResults.length)],
    },
    histogram, convergence, periodStats,
    probPayoff: (allResults.filter(r => r > 0).length / allResults.length * 100).toFixed(1),
    numPeriods,
  };
};

// ============================================================
// SENSITIVITY ENGINE
// ============================================================
const runSensitivity = (baseConfig, paramKey, range, steps = 12) => {
  const data = [];
  for (let i = 0; i <= steps; i++) {
    const val = range[0] + (range[1] - range[0]) * i / steps;
    const cfg = { ...baseConfig, [paramKey]: val };
    const r = runMultiPeriodMC({ ...cfg, numPaths: 5000 });
    data.push({ value: val, fairValue: r.fairValue });
  }
  return data;
};

// ============================================================
// DOCUMENT EXTRACTION (Claude API)
// ============================================================
const extractFromDocument = async (text, mode) => {
  const systemPrompts = {
    backtest: `You are a financial data extraction specialist. Extract earnout/contingent consideration from SEC filings.
Return ONLY valid JSON, no markdown:
{"earnouts":[{"name":"string","acquisitionDate":"string or null","maxPayout":number or null,"initialFairValue":number or null,"currentFairValue":number or null,"priorFairValue":number or null,"fairValueChange":number or null,"metric":"string or null","structure":"linear|binary|tiered|milestone|percentage|cagr|unknown","threshold":number or null,"participationRate":number or null,"cap":number or null,"floor":number or null,"fixedPayment":number or null,"measurementPeriods":[{"year":number,"target":number or null,"label":"string"}],"hasCatchUp":boolean,"hasClawback":boolean,"hasAcceleration":boolean,"accelerationTrigger":"string or null","hasCumulativeTarget":boolean,"cumulativeTarget":number or null,"multiYearCap":number or null,"methodology":"Monte Carlo|probability-weighted|DCF|unknown","discountRate":number or null,"volatility":number or null,"riskFreeRate":number or null,"projectedMetric":number or null,"level3Rollforward":{"openingBalance":number or null,"additions":number or null,"fairValueChanges":number or null,"payments":number or null,"closingBalance":number or null},"confidenceScore":number}],"reportingPeriod":"string","companyName":"string","filingType":"10-K|10-Q"}
Extract EVERY earnout. Use null for undisclosed values.`,

    live_ppa: `You are a valuation report extraction specialist. Extract earnout terms from a PPA valuation report.
Return ONLY valid JSON, no markdown:
{"earnout":{"name":"string","metric":"string","metricDefinition":"string","structure":"linear|binary|tiered|milestone|percentage|cagr|multi-metric","periods":[{"year":number,"yearFromNow":number,"threshold":number or null,"cap":number or null,"floor":number or null,"fixedPayment":number or null,"participationRate":number or null,"projectedMetric":number or null,"tiers":[{"lower":number,"upper":number,"rate":number}] or null}],"hasCatchUp":boolean,"catchUpDescription":"string or null","hasClawback":boolean,"clawbackThreshold":number or null,"clawbackRate":number or null,"clawbackCap":number or null,"hasAcceleration":boolean,"accelerationTrigger":"string or null","accelerationTreatment":"string or null","hasCumulativeTarget":boolean,"cumulativeTarget":number or null,"hasMultiYearCap":boolean,"multiYearCap":number or null,"hasCarryForward":boolean,"isMultiMetric":boolean,"secondMetric":{"name":"string","threshold":number,"currentValue":number,"growthRate":number,"volatility":number} or null,"metricCorrelation":number or null,"paymentTiming":"string","paymentDelay":number or null,"isEscrowed":boolean,"methodology":"Monte Carlo|probability-weighted|DCF","assumptions":{"currentMetric":number,"metricGrowthRate":number or null,"volatility":number,"discountRate":number,"riskFreeRate":number,"creditAdjustment":number or null,"comparableCompanies":["string"] or null},"initialFairValue":number or null,"currency":"string","confidenceScore":number,"ambiguities":["string"],"alternativeInterpretations":[{"clause":"string","interpretation1":"string","interpretation2":"string"}] or null}}`
  };

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" },
      body: JSON.stringify({ model: CLAUDE_MODEL, max_tokens: 4096, system: systemPrompts[mode], messages: [{ role: "user", content: `Extract all earnout information:\n\n${text.substring(0, 80000)}` }] })
    });
    const data = await response.json();
    const txt = data.content?.map(c => c.text || "").join("") || "";
    return JSON.parse(txt.replace(/```json|```/g, "").trim());
  } catch (err) { console.error("Extraction error:", err); return null; }
};

const verifyExtraction = async (text, extracted) => {
  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" },
      body: JSON.stringify({ model: CLAUDE_MODEL, max_tokens: 2048,
        system: `Verify extracted earnout terms. Return ONLY JSON: {"verified":boolean,"errors":[{"field":"string","issue":"string"}],"missingTerms":[{"term":"string"}],"overallConfidence":number,"recommendation":"proceed|review_needed|high_risk"}`,
        messages: [{ role: "user", content: `Document:\n${text.substring(0, 40000)}\n\nExtracted:\n${JSON.stringify(extracted, null, 2)}\n\nVerify.` }] })
    });
    const data = await response.json();
    return JSON.parse((data.content?.map(c => c.text || "").join("") || "").replace(/```json|```/g, "").trim());
  } catch (err) { return { verified: false, overallConfidence: 0, recommendation: "review_needed", errors: [], missingTerms: [] }; }
};

// ============================================================
// EXCEL GENERATION
// ============================================================
const generateExcel = async (params, results, sensitivities) => {
  const XLSX = await import("xlsx");
  const wb = XLSX.utils.book_new();
  const sum = [["ValuProEarnout — Valuation Summary"], ["Date", new Date().toISOString().split("T")[0]], [],
    ["EARNOUT TERMS"], ["Metric", params.metric || "EBITDA"], ["Structure", params.periods?.[0]?.structure || "linear"],
    ["Number of Periods", params.periods?.length || 1], ["Multi-Year Cap", params.multiYearCap || "None"],
    ["Catch-Up", params.hasCatchUp ? "Yes" : "No"], ["Clawback", params.hasClawback ? "Yes" : "No"],
    ["Acceleration", params.hasAcceleration ? "Yes" : "No"], [],
    ["PER-PERIOD DETAIL"]];
  (params.periods || []).forEach((p, i) => {
    sum.push([`Period ${i + 1}`, `Year ${p.yearFromNow || i + 1}`]);
    sum.push(["  Structure", p.structure]); sum.push(["  Threshold", p.threshold]);
    sum.push(["  Participation Rate", p.participationRate]); sum.push(["  Cap", p.cap || "None"]);
    sum.push(["  Projected Metric", p.projectedMetric]); sum.push([]);
  });
  sum.push([], ["ASSUMPTIONS"], ["Current Metric", params.currentMetric], ["Growth Rate", params.metricGrowthRate],
    ["Volatility", params.volatility], ["Discount Rate", params.discountRate], ["Risk-Free Rate", params.riskFreeRate],
    ["Credit Adj", params.creditAdj || 0], [], ["RESULTS"], ["Fair Value", results.fairValue],
    ["95% CI Low", results.ci95[0]], ["95% CI High", results.ci95[1]], ["Std Error", results.stdError],
    ["Prob of Payoff", `${results.probPayoff}%`], [],
    ["PERCENTILES"], ...Object.entries(results.percentiles).map(([k, v]) => [k.replace("p", "") + "th", v]));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(sum), "Summary");

  if (sensitivities) {
    const sd = [["Sensitivity Analysis"], []];
    for (const [label, data] of Object.entries(sensitivities)) {
      sd.push([label], ["Input", "Fair Value"]); data.forEach(d => sd.push([d.value, d.fairValue])); sd.push([]);
    }
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(sd), "Sensitivity");
  }

  if (results.periodStats) {
    const pd = [["Per-Period Analysis"], ["Period", "Mean FV", "P25", "P50", "P75"]];
    results.periodStats.forEach((ps, i) => pd.push([`Period ${i + 1}`, ps.mean, ps.p25, ps.p50, ps.p75]));
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(pd), "Period Detail");
  }

  const dist = [["Distribution"], ["Bin", "Count"]];
  results.histogram.forEach(h => dist.push([h.x, h.count]));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(dist), "Distribution");

  const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([wbout], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `ValuProEarnout_${new Date().toISOString().split("T")[0]}.xlsx`; a.click();
};

// ============================================================
// MEMO GENERATION
// ============================================================
const generateMemo = (params, results, sensitivities, format = "pdf") => {
  const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const periods = params.periods || [];
  const features = [
    params.hasCatchUp && "catch-up provisions", params.hasClawback && "clawback provisions",
    params.hasAcceleration && "acceleration clauses", params.hasCumulativeTarget && "cumulative target",
    params.hasMultiYearCap && "multi-year cap", params.hasCarryForward && "carry-forward",
    params.isMultiMetric && "multi-metric conditions", params.isEscrowed && "escrowed funds",
  ].filter(Boolean);

  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>ValuProEarnout Memo</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:Georgia,serif;font-size:11pt;line-height:1.6;color:#1a1a1a;max-width:8.5in;margin:0 auto;padding:1in}
h1{font-size:16pt;font-weight:700;margin-bottom:4pt}h2{font-size:13pt;font-weight:700;margin-top:18pt;margin-bottom:6pt;border-bottom:1px solid #ccc;padding-bottom:4pt}
p{margin-bottom:8pt;text-align:justify}table{width:100%;border-collapse:collapse;margin:10pt 0;font-size:10pt}th,td{border:1px solid #ccc;padding:4pt 8pt;text-align:left}
th{background:#f5f5f5;font-weight:700}td.n{text-align:right;font-family:Courier,monospace}.hdr{display:flex;justify-content:space-between;margin-bottom:20pt;padding-bottom:10pt;border-bottom:2px solid #1a1a1a}
.ftr{margin-top:30pt;padding-top:10pt;border-top:1px solid #ccc;font-size:9pt;color:#666}@media print{body{padding:0.5in}}</style></head><body>
<div class="hdr"><div><h1>Earnout Fair Value Measurement</h1><p style="font-size:10pt;color:#666">Methodology Memorandum — ${date}</p></div><div style="text-align:right"><strong>ValuProEarnout</strong><br><span style="font-size:9pt;color:#666">Automated Remeasurement Platform</span></div></div>

<h2>1. Overview</h2>
<p>This memorandum documents the fair value measurement of contingent consideration (earnout) in accordance with ASC 820 and ASC 805. The earnout liability is classified as Level 3 within the fair value hierarchy. The earnout spans ${periods.length} measurement period${periods.length > 1 ? "s" : ""}${features.length > 0 ? " and includes " + features.join(", ") : ""}.</p>

<h2>2. Earnout Terms</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Performance Metric</td><td>${params.metric || "EBITDA"}</td></tr>
<tr><td>Number of Measurement Periods</td><td>${periods.length}</td></tr>
<tr><td>Multi-Year Cap</td><td class="n">${params.multiYearCap ? "$" + (params.multiYearCap / 1e6).toFixed(1) + "M" : "None"}</td></tr>
<tr><td>Catch-Up Provisions</td><td>${params.hasCatchUp ? "Yes" : "No"}</td></tr>
<tr><td>Clawback Provisions</td><td>${params.hasClawback ? "Yes" : "No"}</td></tr>
<tr><td>Acceleration Clauses</td><td>${params.hasAcceleration ? "Yes" : "No"}</td></tr>
<tr><td>Escrowed</td><td>${params.isEscrowed ? "Yes" : "No"}</td></tr></table>

<h2>3. Per-Period Structure</h2>
<table><tr><th>Period</th><th>Structure</th><th>Threshold</th><th>Rate/Payment</th><th>Cap</th><th>Projected Metric</th></tr>
${periods.map((p, i) => `<tr><td>Year ${p.yearFromNow || i + 1}</td><td style="text-transform:capitalize">${p.structure}</td><td class="n">${p.threshold ? "$" + (p.threshold / 1e6).toFixed(1) + "M" : "—"}</td><td class="n">${p.participationRate ? (p.participationRate * 100).toFixed(0) + "%" : p.fixedPayment ? "$" + (p.fixedPayment / 1e6).toFixed(1) + "M" : "—"}</td><td class="n">${p.cap ? "$" + (p.cap / 1e6).toFixed(1) + "M" : "None"}</td><td class="n">${p.projectedMetric ? "$" + (p.projectedMetric / 1e6).toFixed(1) + "M" : "—"}</td></tr>`).join("")}
</table>

<h2>4. Valuation Methodology</h2>
<p>Fair value was estimated using a multi-period Monte Carlo simulation under the risk-neutral framework. The underlying performance metric was modeled as a stochastic process with annual resolution across ${periods.length} measurement periods using ${MC_PATHS.toLocaleString()} simulation paths. ${params.hasCatchUp ? "Path dependency was modeled explicitly — shortfalls in earlier periods reduce effective thresholds in subsequent periods per the catch-up provisions." : ""} ${params.hasClawback ? "Clawback provisions were evaluated at the end of all measurement periods based on cumulative metric performance." : ""} ${params.isMultiMetric ? "Correlated simulation of two performance metrics was performed to evaluate the joint probability of both metrics exceeding their respective thresholds." : ""}</p>

<h2>5. Key Assumptions</h2>
<table><tr><th>Assumption</th><th>Value</th><th>Source</th></tr>
<tr><td>Current Metric Value</td><td class="n">$${(params.currentMetric / 1e6).toFixed(1)}M</td><td>Latest actuals / management forecast</td></tr>
<tr><td>Metric Growth Rate</td><td class="n">${(params.metricGrowthRate * 100).toFixed(1)}%</td><td>Management projection</td></tr>
<tr><td>Volatility</td><td class="n">${(params.volatility * 100).toFixed(1)}%</td><td>Comparable company equity volatility</td></tr>
<tr><td>Discount Rate</td><td class="n">${(params.discountRate * 100).toFixed(1)}%</td><td>Risk-adjusted rate</td></tr>
<tr><td>Risk-Free Rate</td><td class="n">${(params.riskFreeRate * 100).toFixed(1)}%</td><td>U.S. Treasury yield</td></tr>
<tr><td>Credit Risk Adjustment</td><td class="n">${((params.creditAdj || 0) * 100).toFixed(1)}%</td><td>Counterparty assessment</td></tr></table>

<h2>6. Fair Value Conclusion</h2>
<table><tr><th>Measure</th><th>Value</th></tr>
<tr><td><strong>Fair Value (Mean)</strong></td><td class="n"><strong>$${(results.fairValue / 1e6).toFixed(2)}M</strong></td></tr>
<tr><td>95% Confidence Interval</td><td class="n">$${(results.ci95[0] / 1e6).toFixed(2)}M — $${(results.ci95[1] / 1e6).toFixed(2)}M</td></tr>
<tr><td>Standard Error</td><td class="n">$${(results.stdError / 1e6).toFixed(3)}M</td></tr>
<tr><td>Probability of Any Payoff</td><td class="n">${results.probPayoff}%</td></tr></table>

${results.periodStats ? `<h2>7. Per-Period Fair Value Decomposition</h2>
<table><tr><th>Period</th><th>Mean FV</th><th>P25</th><th>Median</th><th>P75</th></tr>
${results.periodStats.map((ps, i) => `<tr><td>Period ${i + 1}</td><td class="n">$${(ps.mean / 1e6).toFixed(2)}M</td><td class="n">$${(ps.p25 / 1e6).toFixed(2)}M</td><td class="n">$${(ps.p50 / 1e6).toFixed(2)}M</td><td class="n">$${(ps.p75 / 1e6).toFixed(2)}M</td></tr>`).join("")}
</table>` : ""}

<h2>8. ASC 820 Fair Value Hierarchy</h2>
<p>The earnout liability is classified as <strong>Level 3</strong>. Significant unobservable inputs include management's projected performance metrics, estimated volatility, and the discount rate. The methodology is consistent with the initial purchase price allocation and prior reporting periods.</p>

<div class="ftr"><p>Generated by ValuProEarnout — ${date}</p></div></body></html>`;

  if (format === "pdf" || format === "both") {
    const win = window.open("", "_blank"); win.document.write(html); win.document.close(); setTimeout(() => win.print(), 500);
  }
  if (format === "docx" || format === "both") {
    const blob = new Blob([`<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word'><head><meta charset="utf-8"></head><body>${html}</body></html>`], { type: "application/msword" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `ValuProEarnout_Memo_${new Date().toISOString().split("T")[0]}.doc`; a.click();
  }
};

// ============================================================
// FORMATTING
// ============================================================
const fmt = (n, d = 0) => { if (n == null || isNaN(n)) return "—"; if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(2)}B`; if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(2)}M`; if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(1)}K`; return `$${n.toFixed(d)}`; };
const fmtPct = (n) => n == null ? "—" : `${(n * 100).toFixed(1)}%`;

// ============================================================
// INLINE SVG ICONS
// ============================================================
const Icon = ({ name, size = 16, color = "currentColor" }) => {
  const d = { target: `<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>`, upload: `<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>`, file: `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>`, check: `<polyline points="20 6 9 17 4 12"/>`, alert: `<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>`, refresh: `<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>`, download: `<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>`, bar: `<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>`, sliders: `<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>`, dollar: `<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>`, shield: `<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>`, sun: `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>`, moon: `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>`, arrowRight: `<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>`, info: `<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>`, activity: `<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>`, layers: `<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>`, scale: `<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22V8"/><path d="M20 8L12 16 4 8"/>`, book: `<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>`, plus: `<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>`, chevronRight: `<polyline points="9 18 15 12 9 6"/>`, brain: `<path d="M12 2a7 7 0 0 1 7 7c0 2.4-1.2 4.5-3 5.7V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.3C6.2 13.5 5 11.4 5 9a7 7 0 0 1 7-7z"/>`, sparkles: `<path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z"/>`, fileSearch: `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><circle cx="11.5" cy="14.5" r="2.5"/><line x1="13.25" y1="16.25" x2="15" y2="18"/>`, trendUp: `<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>`, percent: `<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>`, hash: `<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>`, calendar: `<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>` };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" dangerouslySetInnerHTML={{ __html: d[name] || "" }} />;
};

// ============================================================
// UI COMPONENTS
// ============================================================
const AnimatedValue = ({ value, prefix = "$" }) => {
  const [display, setDisplay] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    const start = display, end = value, duration = 800, startTime = Date.now();
    const animate = () => { const p = Math.min((Date.now() - startTime) / duration, 1); setDisplay(start + (end - start) * (1 - Math.pow(1 - p, 3))); if (p < 1) ref.current = requestAnimationFrame(animate); };
    ref.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(ref.current);
  }, [value]);
  const f = Math.abs(display) >= 1e6 ? `${(display / 1e6).toFixed(2)}M` : Math.abs(display) >= 1e3 ? `${(display / 1e3).toFixed(1)}K` : display.toFixed(0);
  return <span>{prefix}{f}</span>;
};

const Histogram = ({ data, fairValue, theme }) => {
  const maxCount = Math.max(...data.map(d => d.count));
  const tc = theme === "dark";
  return (<div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 140, padding: "0 4px" }}>
    {data.map((bin, i) => {
      const h = maxCount > 0 ? (bin.count / maxCount) * 100 : 0;
      const isFV = fairValue >= bin.x - (data[1]?.x - data[0]?.x) / 2 && fairValue < bin.x + (data[1]?.x - data[0]?.x) / 2;
      return <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: "100%" }}>
        <div style={{ width: "100%", height: `${h}%`, backgroundColor: isFV ? "#d97706" : tc ? "#3b82f6" : "#2563eb", borderRadius: "2px 2px 0 0", minHeight: bin.count > 0 ? 2 : 0, opacity: isFV ? 1 : 0.65, transition: "height 0.3s" }} />
      </div>;
    })}
  </div>);
};

const ParamSlider = ({ label, value, onChange, min, max, step, format = "number", tooltip, theme }) => {
  const tc = theme === "dark";
  const dv = format === "percent" ? `${(value * 100).toFixed(1)}%` : format === "currency" ? fmt(value) : format === "years" ? `${value.toFixed(1)} yrs` : `${value.toFixed(step < 1 ? 1 : 0)}`;
  return (<div style={{ marginBottom: 13 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
      <span style={{ fontSize: 12, color: tc ? "#8896b0" : "#6b7280", display: "flex", alignItems: "center", gap: 4 }}>{label}{tooltip && <span title={tooltip} style={{ cursor: "help", opacity: 0.4 }}><Icon name="info" size={10} /></span>}</span>
      <span style={{ fontSize: 12, fontFamily: "'IBM Plex Mono',monospace", color: tc ? "#e8edf5" : "#111827", fontWeight: 600 }}>{dv}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#2563eb", height: 4, cursor: "pointer" }} />
  </div>);
};

const TornadoChart = ({ sensitivities, baseValue, theme }) => {
  if (!sensitivities) return null;
  const tc = theme === "dark";
  const entries = Object.entries(sensitivities).map(([label, data]) => { const v = data.map(d => d.fairValue); return { label, min: Math.min(...v), max: Math.max(...v) }; }).sort((a, b) => (b.max - b.min) - (a.max - a.min));
  const gMin = Math.min(...entries.map(e => e.min)), gMax = Math.max(...entries.map(e => e.max)), range = gMax - gMin || 1;
  return (<div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
    {entries.map((e, i) => { const lp = ((e.min - gMin) / range) * 100, wp = ((e.max - e.min) / range) * 100, bp = ((baseValue - gMin) / range) * 100;
      return <div key={i}><div style={{ fontSize: 11, color: tc ? "#8896b0" : "#6b7280", marginBottom: 2 }}>{e.label}</div>
        <div style={{ position: "relative", height: 22, background: tc ? "#1e2d4a" : "#e5e7eb", borderRadius: 4 }}>
          <div style={{ position: "absolute", left: `${lp}%`, width: `${wp}%`, height: "100%", background: "linear-gradient(90deg,#dc2626,#2563eb,#059669)", borderRadius: 4, opacity: 0.75 }} />
          <div style={{ position: "absolute", left: `${bp}%`, top: 0, bottom: 0, width: 2, background: "#d97706" }} />
          <div style={{ position: "absolute", left: 4, top: 3, fontSize: 10, color: tc ? "#e8edf5" : "#111827", fontFamily: "'IBM Plex Mono',monospace" }}>{fmt(e.min)}</div>
          <div style={{ position: "absolute", right: 4, top: 3, fontSize: 10, color: tc ? "#e8edf5" : "#111827", fontFamily: "'IBM Plex Mono',monospace" }}>{fmt(e.max)}</div>
        </div></div>; })}
  </div>);
};

// ============================================================
// MAIN APPLICATION
// ============================================================
export default function ValuProEarnout() {
  const [theme, setTheme] = useState("light");
  const [view, setView] = useState("landing");
  const [mode, setMode] = useState(null);

  const [files, setFiles] = useState([]);
  const [docText, setDocText] = useState("");
  const [extractedData, setExtractedData] = useState(null);
  const [verification, setVerification] = useState(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");

  // Multi-period earnout params
  const [params, setParams] = useState({
    metric: "Adjusted EBITDA",
    currentMetric: 15e6,
    metricGrowthRate: 0.08,
    volatility: 0.40,
    discountRate: 0.12,
    riskFreeRate: 0.043,
    creditAdj: 0.01,
    payoffDiscountRate: null, // null = auto-calculated from riskFreeRate + creditAdj
    isEscrowed: false,
    // Periods
    periods: [
      { year: 1, yearFromNow: 1, structure: "binary", threshold: 18e6, participationRate: 0, fixedPayment: 5e6, cap: 5e6, floor: 0, projectedMetric: 18e6, tiers: null },
      { year: 2, yearFromNow: 2, structure: "binary", threshold: 20e6, participationRate: 0, fixedPayment: 5e6, cap: 5e6, floor: 0, projectedMetric: 20e6, tiers: null },
      { year: 3, yearFromNow: 3, structure: "binary", threshold: 22e6, participationRate: 0, fixedPayment: 5e6, cap: 5e6, floor: 0, projectedMetric: 22e6, tiers: null },
    ],
    // Path dependency features
    hasCatchUp: false, hasCumulativeTarget: false, cumulativeTarget: 0,
    hasMultiYearCap: true, multiYearCap: 15e6, hasCarryForward: false,
    hasClawback: false, clawbackThreshold: 0, clawbackRate: 0, clawbackCap: 0,
    hasAcceleration: false, accelerationProb: 0.05, accelerationTreatment: "max", accelerationPercentile: 75,
    isMultiMetric: false, secondMetric: null, metricCorrelation: 0.5,
    paymentDelay: 120,
  });

  const [results, setResults] = useState(null);
  const [sensitivities, setSensitivities] = useState(null);
  const [backtestComparison, setBacktestComparison] = useState(null);

  const tc = theme === "dark";
  const c = {
    bg: tc ? "#0c1222" : "#f8f9fb", card: tc ? "#151e30" : "#ffffff", cardBorder: tc ? "#1e2d4a" : "#e5e7eb",
    cardShadow: tc ? "none" : "0 1px 3px rgba(0,0,0,0.04)", accent: "#2563eb",
    accentLight: tc ? "rgba(37,99,235,0.12)" : "rgba(37,99,235,0.06)",
    success: "#059669", warning: "#d97706", danger: "#dc2626",
    text: tc ? "#e8edf5" : "#111827", textMuted: tc ? "#8896b0" : "#6b7280", textDim: tc ? "#556480" : "#9ca3af",
    headerBg: tc ? "rgba(12,18,34,0.85)" : "rgba(255,255,255,0.92)", inputBg: tc ? "#1a2540" : "#f3f4f6",
  };
  const cardStyle = { background: c.card, border: `1px solid ${c.cardBorder}`, borderRadius: 12, padding: 20, boxShadow: c.cardShadow };

  const runValuation = useCallback((p = params) => {
    setTimeout(() => {
      const r = runMultiPeriodMC(p);
      setResults(r);
      const sens = {};
      sens["Volatility"] = runSensitivity(p, "volatility", [0.15, 0.70]);
      sens["Metric Discount Rate"] = runSensitivity(p, "discountRate", [0.06, 0.20]);
      sens["Current Metric"] = runSensitivity(p, "currentMetric", [p.currentMetric * 0.5, p.currentMetric * 1.5]);
      sens["Growth Rate"] = runSensitivity(p, "metricGrowthRate", [0, 0.20]);
      setSensitivities(sens);
    }, 50);
  }, [params]);

  const handleFileUpload = (e) => {
    const nf = Array.from(e.target.files); setFiles(prev => [...prev, ...nf]);
    nf.forEach(f => { const r = new FileReader(); r.onload = (ev) => setDocText(prev => prev + "\n" + ev.target.result); r.readAsText(f); });
  };

  const runPipeline = async () => {
    setView("processing"); setProgress(0);
    setStage("Analyzing document..."); setProgress(10); await new Promise(r => setTimeout(r, 400));
    setStage(mode === "backtest" ? "Extracting earnout disclosures..." : "Extracting earnout terms..."); setProgress(25);
    const extracted = await extractFromDocument(docText, mode === "backtest" ? "backtest" : "live_ppa");
    setExtractedData(extracted); setProgress(50);
    setStage("Verifying extraction..."); const verif = await verifyExtraction(docText, extracted); setVerification(verif); setProgress(65);
    setStage("Configuring model..."); setProgress(75);

    if (mode === "backtest" && extracted?.earnouts?.length > 0) {
      const e = extracted.earnouts[0];
      const numPeriods = e.measurementPeriods?.length || 3;
      const newPeriods = Array.from({ length: numPeriods }, (_, i) => ({
        year: i + 1, yearFromNow: e.measurementPeriods?.[i]?.year || (i + 1),
        structure: e.structure || "binary", threshold: e.threshold || 0,
        participationRate: e.participationRate || 0, fixedPayment: e.fixedPayment || (e.maxPayout ? e.maxPayout / numPeriods : 5e6),
        cap: e.cap || (e.maxPayout ? e.maxPayout / numPeriods : null), floor: e.floor || 0,
        projectedMetric: e.projectedMetric || params.currentMetric * Math.pow(1.08, i + 1), tiers: null,
      }));
      const np = { ...params, metric: e.metric || params.metric, currentMetric: e.projectedMetric || params.currentMetric,
        volatility: e.volatility || params.volatility, discountRate: e.discountRate || params.discountRate,
        riskFreeRate: e.riskFreeRate || params.riskFreeRate, periods: newPeriods,
        hasCatchUp: e.hasCatchUp || false, hasClawback: e.hasClawback || false,
        hasAcceleration: e.hasAcceleration || false, hasMultiYearCap: e.multiYearCap ? true : false,
        multiYearCap: e.multiYearCap || e.maxPayout || 15e6,
      };
      setParams(np); setStage("Running Monte Carlo..."); setProgress(85);
      await new Promise(r => setTimeout(r, 200));
      const res = runMultiPeriodMC(np); setResults(res);
      if (e.currentFairValue || e.initialFairValue) {
        const reported = e.currentFairValue || e.initialFairValue;
        setBacktestComparison({ reported, computed: res.fairValue, gap: Math.abs(res.fairValue - reported) / reported * 100 });
      }
    } else if (mode === "live" && extracted?.earnout) {
      const e = extracted.earnout; const a = e.assumptions || {};
      const newPeriods = (e.periods || []).map((p, i) => ({
        year: i + 1, yearFromNow: p.yearFromNow || (i + 1), structure: p.structure || e.structure || "linear",
        threshold: p.threshold || 0, participationRate: p.participationRate || 0, fixedPayment: p.fixedPayment || 0,
        cap: p.cap || null, floor: p.floor || 0, projectedMetric: p.projectedMetric || 0, tiers: p.tiers || null,
      }));
      if (newPeriods.length === 0) newPeriods.push({ ...params.periods[0] });
      const np = { ...params, metric: e.metric || params.metric, currentMetric: a.currentMetric || params.currentMetric,
        metricGrowthRate: a.metricGrowthRate || params.metricGrowthRate, volatility: a.volatility || params.volatility,
        discountRate: a.discountRate || params.discountRate, riskFreeRate: a.riskFreeRate || params.riskFreeRate,
        creditAdj: a.creditAdjustment || params.creditAdj, periods: newPeriods,
        hasCatchUp: e.hasCatchUp || false, hasClawback: e.hasClawback || false,
        hasAcceleration: e.hasAcceleration || false, hasMultiYearCap: e.hasMultiYearCap || false,
        multiYearCap: e.multiYearCap || 0, hasCarryForward: e.hasCarryForward || false,
        hasCumulativeTarget: e.hasCumulativeTarget || false, cumulativeTarget: e.cumulativeTarget || 0,
        isMultiMetric: e.isMultiMetric || false, secondMetric: e.secondMetric || null,
        isEscrowed: e.isEscrowed || false, paymentDelay: e.paymentDelay || 120,
      };
      setParams(np); setStage("Running Monte Carlo..."); setProgress(85);
      await new Promise(r => setTimeout(r, 200)); runValuation(np);
    }
    setProgress(95); setStage("Generating sensitivity..."); await new Promise(r => setTimeout(r, 300));
    setProgress(100); setTimeout(() => setView("results"), 400);
  };

  // ---- GRANT THORNTON DEMO ----
  const runGTDemo = () => {
    setMode("backtest");
    setView("processing"); setProgress(0);
    const gtParams = {
      metric: "Adjusted EBITDA",
      currentMetric: 12e6, // FY18 base EBITDA ~$12M
      metricGrowthRate: 0.12, // Implied by management projections ($12M → $17M over 3 yrs ≈ 12% CAGR)
      volatility: 0.40, // GT stated: 40% EBITDA volatility
      discountRate: 0.10, // GT stated: 10% EBITDA discount rate (metric risk premium)
      riskFreeRate: 0.025, // ~2018 UST rate
      creditAdj: 0.02, // Implied credit spread (risk-adjusted rate 4.5% = riskFree 2.5% + credit 2%)
      payoffDiscountRate: 0.045, // GT stated: 4.5% risk-adjusted discount rate for payoffs
      isEscrowed: false,
      periods: [
        { year: 1, yearFromNow: 1, structure: "binary", threshold: 14e6, participationRate: 0, fixedPayment: 5e6, cap: 5e6, floor: 0, projectedMetric: 14e6, tiers: null },
        { year: 2, yearFromNow: 2, structure: "binary", threshold: 15.5e6, participationRate: 0, fixedPayment: 5e6, cap: 5e6, floor: 0, projectedMetric: 15.5e6, tiers: null },
        { year: 3, yearFromNow: 3, structure: "binary", threshold: 17e6, participationRate: 0, fixedPayment: 5e6, cap: 5e6, floor: 0, projectedMetric: 17e6, tiers: null },
      ],
      hasCatchUp: false, hasCumulativeTarget: false, cumulativeTarget: 0,
      hasMultiYearCap: true, multiYearCap: 15e6, hasCarryForward: false,
      hasClawback: false, clawbackThreshold: 0, clawbackRate: 0, clawbackCap: 0,
      hasAcceleration: false, accelerationProb: 0, accelerationTreatment: "max", accelerationPercentile: 75,
      isMultiMetric: false, secondMetric: null, metricCorrelation: 0.5,
      paymentDelay: 90,
    };

    // Simulate the processing stages quickly (no API call needed)
    const stages = [
      { s: "Loading Grant Thornton example...", p: 15, d: 300 },
      { s: "Terms pre-loaded: 3-year binary earnout, $5M/year", p: 35, d: 400 },
      { s: "Assumptions: 40% vol, 10% discount (per GT)", p: 55, d: 350 },
      { s: `Running Monte Carlo (${MC_PATHS.toLocaleString()} paths)...`, p: 80, d: 500 },
      { s: "Generating sensitivity analysis...", p: 95, d: 400 },
    ];

    let delay = 0;
    stages.forEach(({ s, p, d }) => {
      delay += d;
      setTimeout(() => { setStage(s); setProgress(p); }, delay);
    });

    setTimeout(() => {
      setParams(gtParams);
      const res = runMultiPeriodMC(gtParams);
      setResults(res);

      // GT expected range: ~$8M–$12M
      setBacktestComparison({
        reported: 10e6, // GT midpoint estimate
        computed: res.fairValue,
        gap: Math.abs(res.fairValue - 10e6) / 10e6 * 100,
        isDemo: true,
      });

      // Run sensitivities
      const sens = {};
      sens["Volatility"] = runSensitivity(gtParams, "volatility", [0.15, 0.70]);
      sens["Discount Rate"] = runSensitivity(gtParams, "discountRate", [0.06, 0.20]);
      sens["Current Metric"] = runSensitivity(gtParams, "currentMetric", [gtParams.currentMetric * 0.5, gtParams.currentMetric * 1.5]);
      sens["Growth Rate"] = runSensitivity(gtParams, "metricGrowthRate", [0, 0.20]);
      setSensitivities(sens);

      setProgress(100);
      setTimeout(() => setView("results"), 300);
    }, delay + 500);
  };

  const resetAll = () => { setView("landing"); setResults(null); setSensitivities(null); setExtractedData(null); setDocText(""); setFiles([]); setBacktestComparison(null); setMode(null); };

  // ============================================================
  // RENDER
  // ============================================================
  const fontLink = <><link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
    <style>{`*{box-sizing:border-box}body{font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;margin:0}::selection{background:#2563eb;color:white}
input[type=range]{-webkit-appearance:none;background:${c.cardBorder};border-radius:99px}input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#2563eb;cursor:pointer}
.vg{display:grid}.vf{display:flex}@media(max-width:768px){.r-stack{grid-template-columns:1fr!important}.r-col{flex-direction:column!important}.r-wrap{flex-wrap:wrap}.r-p{padding:16px!important}.r-h1{font-size:28px!important}}`}</style></>;

  const themeBtn = <button onClick={() => setTheme(t => t === "dark" ? "light" : "dark")} style={{ padding: "5px 10px", background: c.inputBg, border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.textMuted, cursor: "pointer", display: "flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 500 }}><Icon name={tc ? "sun" : "moon"} size={12} />{tc ? "Light" : "Dark"}</button>;

  const hdr = (back = false) => (
    <header className="vf" style={{ padding: "10px 28px", justifyContent: "space-between", alignItems: "center", borderBottom: `1px solid ${c.cardBorder}`, background: c.headerBg, backdropFilter: "blur(16px)", position: "sticky", top: 0, zIndex: 100 }}>
      <div className="vf" style={{ alignItems: "center", gap: 12 }}>
        {back && <button onClick={resetAll} style={{ background: "none", border: "none", color: c.textMuted, cursor: "pointer", transform: "rotate(180deg)", padding: 2 }}><Icon name="chevronRight" size={15} /></button>}
        <div className="vf" style={{ alignItems: "center", gap: 8, cursor: "pointer" }} onClick={() => setView("landing")}>
          <div style={{ width: 28, height: 28, borderRadius: 7, background: "linear-gradient(135deg,#2563eb,#1d4ed8)", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon name="scale" size={14} color="white" /></div>
          <span style={{ fontSize: 15, fontWeight: 600, color: c.text }}>ValuPro<span style={{ color: c.accent }}>Earnout</span></span>
        </div>
      </div>
      <div className="vf" style={{ gap: 8, alignItems: "center" }}>
        {mode && <span style={{ padding: "3px 10px", borderRadius: 5, background: mode === "backtest" ? "rgba(5,150,105,0.08)" : c.accentLight, fontSize: 10, color: mode === "backtest" ? c.success : c.accent, fontWeight: 500 }}>{mode === "backtest" ? "Backtest" : "Live"}</span>}
        {themeBtn}
      </div>
    </header>
  );

  // ---- LANDING ----
  if (view === "landing") return (
    <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',system-ui,sans-serif", color: c.text }}>
      {fontLink}{hdr()}
      <main style={{ maxWidth: 720, margin: "0 auto", padding: "80px 28px 48px", textAlign: "center" }}>
        <div style={{ display: "inline-flex", padding: "4px 12px", borderRadius: 5, background: c.accentLight, fontSize: 11, fontWeight: 500, color: c.accent, marginBottom: 24, gap: 4, alignItems: "center" }}>
          <Icon name="sparkles" size={11} color={c.accent} /> Multi-Period Earnout Remeasurement
        </div>
        <h1 className="r-h1" style={{ fontSize: 40, fontWeight: 700, lineHeight: 1.15, marginBottom: 16, letterSpacing: "-1px", fontFamily: "'Source Serif 4',Georgia,serif", color: c.text }}>
          Stop overpaying for<br /><span style={{ color: c.accent }}>quarterly earnout updates</span>
        </h1>
        <p style={{ fontSize: 15, color: c.textMuted, maxWidth: 500, margin: "0 auto", lineHeight: 1.7, marginBottom: 48 }}>
          Multi-period Monte Carlo with path dependency, catch-ups, clawbacks, and acceleration. Upload once, remeasure every quarter.
        </p>
        <div className="vg r-stack" style={{ gridTemplateColumns: "1fr 1fr", gap: 14, maxWidth: 580, margin: "0 auto 48px" }}>
          {[{ id: "backtest", icon: "target", color: c.success, title: "Backtest", desc: "Upload SEC 10-K/10-Q to validate against reported values", tags: ["10-K / 10-Q", "Validation"] },
            { id: "live", icon: "activity", color: c.accent, title: "Live Valuation", desc: "Upload PPA report + forecast for quarterly remeasurement", tags: ["PPA Report", "Quarterly"] }
          ].map(item => (
            <div key={item.id} onClick={() => { setMode(item.id); setView(item.id === "backtest" ? "backtest_upload" : "live_upload"); }}
              style={{ ...cardStyle, cursor: "pointer", padding: 26, textAlign: "left", transition: "all 0.2s" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = item.color; }} onMouseLeave={e => { e.currentTarget.style.borderColor = c.cardBorder; }}>
              <div style={{ width: 32, height: 32, borderRadius: 7, background: item.id === "backtest" ? "rgba(5,150,105,0.06)" : c.accentLight, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 14 }}><Icon name={item.icon} size={16} color={item.color} /></div>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 5, color: c.text }}>{item.title}</h3>
              <p style={{ fontSize: 12, color: c.textMuted, lineHeight: 1.6, margin: 0 }}>{item.desc}</p>
              <div className="vf" style={{ gap: 5, marginTop: 12 }}>{item.tags.map(t => <span key={t} style={{ padding: "2px 7px", borderRadius: 4, background: item.id === "backtest" ? "rgba(5,150,105,0.04)" : c.accentLight, fontSize: 10, color: item.color, fontWeight: 500 }}>{t}</span>)}</div>
            </div>
          ))}
        </div>
        <div className="vf r-wrap" style={{ gap: 32, justifyContent: "center" }}>
          {[{ n: "50K", l: "MC Paths" }, { n: "Multi-Period", l: "Path Dependent" }, { n: "ASC 820", l: "Compliant" }, { n: "Catch-Up", l: "& Clawback" }].map((s, i) => (
            <div key={i} style={{ textAlign: "center" }}><div style={{ fontSize: 14, fontWeight: 600, color: c.accent, fontFamily: "'IBM Plex Mono',monospace" }}>{s.n}</div><div style={{ fontSize: 10, color: c.textDim, marginTop: 1 }}>{s.l}</div></div>
          ))}
        </div>
      </main>
      <footer style={{ padding: "14px 28px", borderTop: `1px solid ${c.cardBorder}`, display: "flex", justifyContent: "space-between", fontSize: 10, color: c.textDim }}>
        <span>ValuProEarnout</span><div className="vf" style={{ gap: 4, alignItems: "center" }}><Icon name="shield" size={10} color={c.textDim} /> Encrypted</div>
      </footer>
    </div>
  );

  // ---- UPLOAD ----
  if (view === "backtest_upload" || view === "live_upload") {
    const isBT = view === "backtest_upload";
    const docs = isBT ? [{ label: "SEC 10-K Filing", desc: "Annual report with earnout fair value, Level 3 rollforward" }, { label: "SEC 10-Q (optional)", desc: "Quarterly report for additional data points" }]
      : [{ label: "PPA Valuation Report", desc: "Initial earnout valuation from your PPA firm" }, { label: "Management Forecast", desc: "Latest projections for the earnout metric" }];
    return (
      <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',system-ui,sans-serif", color: c.text }}>{fontLink}{hdr(true)}
        <div className="r-p" style={{ maxWidth: 580, margin: "40px auto", padding: "0 28px" }}>
          <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 5, fontFamily: "'Source Serif 4',Georgia,serif" }}>{isBT ? "Upload SEC Filing" : "Upload Documents"}</h2>
          <p style={{ color: c.textMuted, marginBottom: 28, fontSize: 13 }}>{isBT ? "Upload 10-K or 10-Q to extract earnout disclosures." : "Upload PPA report and latest forecast."}</p>
          {docs.map((d, i) => (<div key={i} style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: c.text, marginBottom: 5 }}>{d.label}</div>
            <label style={{ ...cardStyle, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 80, cursor: "pointer", borderStyle: "dashed", borderWidth: 2 }}>
              <input type="file" accept=".txt,.htm,.html,.pdf,.doc,.docx,.csv" onChange={handleFileUpload} style={{ display: "none" }} />
              <Icon name="upload" size={20} color={c.accent} /><span style={{ fontSize: 11, color: c.textMuted, marginTop: 6 }}>{d.desc}</span>
            </label>
          </div>))}
          {files.length > 0 && files.map((f, i) => (<div key={i} className="vf" style={{ alignItems: "center", gap: 8, padding: "7px 12px", background: c.accentLight, borderRadius: 6, marginBottom: 5 }}>
            <Icon name="file" size={13} color={c.accent} /><span style={{ fontSize: 11, flex: 1, color: c.text }}>{f.name}</span><Icon name="check" size={13} color={c.success} />
          </div>))}

          {/* GT Demo — backtest only */}
          {isBT && (
            <div style={{ marginTop: 20, marginBottom: 20 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                <div style={{ flex: 1, height: 1, background: c.cardBorder }} />
                <span style={{ fontSize: 10, color: c.textDim, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.06em" }}>or try a demo</span>
                <div style={{ flex: 1, height: 1, background: c.cardBorder }} />
              </div>
              <div onClick={runGTDemo}
                style={{ ...cardStyle, cursor: "pointer", padding: 18, transition: "all 0.2s", borderColor: "rgba(5,150,105,0.15)", background: tc ? "rgba(5,150,105,0.03)" : "rgba(5,150,105,0.02)" }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = c.success; e.currentTarget.style.transform = "translateY(-1px)"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(5,150,105,0.15)"; e.currentTarget.style.transform = "none"; }}>
                <div className="vf" style={{ alignItems: "center", gap: 12 }}>
                  <div style={{ width: 36, height: 36, borderRadius: 8, background: "rgba(5,150,105,0.08)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <Icon name="target" size={17} color={c.success} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: c.text, marginBottom: 2 }}>Run Grant Thornton Example</div>
                    <div style={{ fontSize: 11, color: c.textMuted, lineHeight: 1.5 }}>
                      3-year binary earnout • $5M/year if EBITDA target met • $15M max
                    </div>
                    <div className="vf" style={{ gap: 4, marginTop: 6, flexWrap: "wrap" }}>
                      {["40% Volatility", "10% Discount Rate", "No API Key Needed", "Instant Results"].map(t => (
                        <span key={t} style={{ padding: "2px 6px", borderRadius: 3, background: "rgba(5,150,105,0.06)", fontSize: 9, color: c.success, fontWeight: 500 }}>{t}</span>
                      ))}
                    </div>
                  </div>
                  <Icon name="chevronRight" size={16} color={c.success} />
                </div>
              </div>
            </div>
          )}

          <div className="vf r-col" style={{ gap: 10, marginTop: isBT ? 0 : 24 }}>
            <button onClick={() => { setView("landing"); setMode(null); setFiles([]); setDocText(""); }} style={{ padding: "10px 20px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 8, color: c.text, cursor: "pointer", fontSize: 12 }}>Back</button>
            <button onClick={runPipeline} disabled={!docText} style={{ flex: 1, padding: "10px 20px", background: docText ? "linear-gradient(135deg,#2563eb,#1d4ed8)" : c.cardBorder, border: "none", borderRadius: 8, color: "white", cursor: docText ? "pointer" : "not-allowed", fontSize: 13, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
              <Icon name="brain" size={15} color="white" />{isBT ? "Run Backtest" : "Run Valuation"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---- PROCESSING ----
  if (view === "processing") return (
    <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',system-ui,sans-serif", color: c.text }}>{fontLink}{hdr()}
      <div className="r-p" style={{ maxWidth: 480, margin: "60px auto", padding: "0 24px", textAlign: "center" }}>
        <div style={{ width: 60, height: 60, borderRadius: "50%", background: c.accentLight, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 24px" }}>
          <div style={{ animation: "spin 2s linear infinite" }}><Icon name="refresh" size={24} color={c.accent} /></div>
        </div>
        <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
        <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4, fontFamily: "'Source Serif 4',Georgia,serif" }}>{mode === "backtest" ? "Running Backtest" : "Processing"}</h2>
        <p style={{ fontSize: 13, color: c.accent, marginBottom: 24, fontWeight: 500 }}>{stage}</p>
        <div style={{ width: "100%", height: 4, background: c.cardBorder, borderRadius: 100, overflow: "hidden", marginBottom: 28 }}>
          <div style={{ width: `${progress}%`, height: "100%", background: "linear-gradient(90deg,#2563eb,#60a5fa)", borderRadius: 100, transition: "width 0.5s" }} />
        </div>
        {[{ l: "Document ingestion", t: 10 }, { l: "Term extraction (multi-pass)", t: 25 }, { l: "Adversarial verification", t: 50 }, { l: "Model configuration", t: 65 },
          { l: `Multi-period Monte Carlo (${MC_PATHS.toLocaleString()} paths)`, t: 85 }, { l: "Sensitivity analysis", t: 95 }].map((s, i) => (
          <div key={i} className="vf" style={{ alignItems: "center", gap: 8, padding: "5px 0", opacity: progress >= s.t ? 1 : 0.3 }}>
            {progress > s.t + 5 ? <Icon name="check" size={14} color={c.success} /> : progress >= s.t ? <div style={{ width: 14, height: 14, borderRadius: "50%", border: `2px solid ${c.accent}`, borderTopColor: "transparent", animation: "spin 1s linear infinite" }} /> : <div style={{ width: 14, height: 14, borderRadius: "50%", border: `1px solid ${c.textDim}` }} />}
            <span style={{ fontSize: 12, color: progress >= s.t ? c.text : c.textDim }}>{s.l}</span>
          </div>))}
      </div>
    </div>
  );

  // ---- RESULTS ----
  if (view === "results" && results) return (
    <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',system-ui,sans-serif", color: c.text }}>{fontLink}{hdr(true)}
      <div className="r-p" style={{ padding: "24px 28px", maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div className="vf r-col" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, gap: 12 }}>
          <div>
            <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 3, fontFamily: "'Source Serif 4',Georgia,serif" }}>{mode === "backtest" ? "Backtest Results" : "Earnout Remeasurement"}</h2>
            <p style={{ fontSize: 12, color: c.textMuted }}>{params.metric} • {params.periods.length} periods • {new Date().toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}</p>
          </div>
          <div className="vf r-wrap" style={{ gap: 6 }}>
            <button onClick={() => generateExcel(params, results, sensitivities)} style={{ padding: "7px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 5 }}><Icon name="download" size={13} />Excel</button>
            <button onClick={() => generateMemo(params, results, sensitivities, "pdf")} style={{ padding: "7px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 5 }}><Icon name="file" size={13} />PDF</button>
            <button onClick={() => generateMemo(params, results, sensitivities, "docx")} style={{ padding: "7px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 5 }}><Icon name="file" size={13} />Word</button>
            <button onClick={resetAll} style={{ padding: "7px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 5 }}><Icon name="plus" size={13} />New</button>
          </div>
        </div>

        {/* Backtest banner */}
        {backtestComparison && (
          <div style={{ ...cardStyle, marginBottom: 14, padding: 16, background: backtestComparison.gap < 5 ? "rgba(5,150,105,0.04)" : backtestComparison.gap < 10 ? "rgba(217,119,6,0.04)" : "rgba(220,38,38,0.04)", borderColor: backtestComparison.gap < 5 ? "rgba(5,150,105,0.15)" : backtestComparison.gap < 10 ? "rgba(217,119,6,0.15)" : "rgba(220,38,38,0.15)" }}>
            <div className="vf r-col" style={{ alignItems: "center", gap: 14 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: backtestComparison.gap < 5 ? c.success : backtestComparison.gap < 10 ? c.warning : c.danger, marginBottom: 3 }}>
                  {backtestComparison.isDemo ? "Grant Thornton Benchmark" : "Backtest"}: {backtestComparison.gap.toFixed(1)}% gap
                </div>
                <div style={{ fontSize: 11, color: c.textMuted }}>
                  {backtestComparison.isDemo
                    ? `GT expected range $8M–$12M (midpoint $10M) — ValuPro ${fmt(backtestComparison.computed)} • 3-year binary, 40% vol, 10% discount`
                    : `Reported ${fmt(backtestComparison.reported)} — ValuPro ${fmt(backtestComparison.computed)}`}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="vg r-stack" style={{ gridTemplateColumns: "280px 1fr", gap: 16 }}>
          {/* LEFT: Params */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Features summary */}
            <div style={cardStyle}>
              <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="layers" size={13} color={c.accent} /> Earnout Features</h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {[{ l: `${params.periods.length} Periods`, on: true }, { l: "Catch-Up", on: params.hasCatchUp }, { l: "Clawback", on: params.hasClawback },
                  { l: "Acceleration", on: params.hasAcceleration }, { l: "Multi-Year Cap", on: params.hasMultiYearCap }, { l: "Carry-Forward", on: params.hasCarryForward },
                  { l: "Cumulative Target", on: params.hasCumulativeTarget }, { l: "Multi-Metric", on: params.isMultiMetric }, { l: "Escrowed", on: params.isEscrowed },
                ].filter(f => f.on).map(f => <span key={f.l} style={{ padding: "2px 7px", borderRadius: 4, background: c.accentLight, fontSize: 10, color: c.accent, fontWeight: 500 }}>{f.l}</span>)}
              </div>
            </div>

            {/* Per-period projections */}
            <div style={cardStyle}>
              <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="calendar" size={13} color={c.accent} /> Period Projections</h3>
              {params.periods.map((p, i) => (
                <div key={i} style={{ marginBottom: 10, paddingBottom: 8, borderBottom: i < params.periods.length - 1 ? `1px solid ${c.cardBorder}` : "none" }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: c.text, marginBottom: 4 }}>Year {p.yearFromNow || i + 1} — <span style={{ textTransform: "capitalize", color: c.accent }}>{p.structure}</span></div>
                  <ParamSlider theme={theme} label={`Yr ${p.yearFromNow} Projected ${params.metric}`} value={p.projectedMetric || params.currentMetric} onChange={v => { const np = [...params.periods]; np[i] = { ...np[i], projectedMetric: v }; setParams(pr => ({ ...pr, periods: np })); }} min={5e6} max={50e6} step={500000} format="currency" />
                  <ParamSlider theme={theme} label={`Yr ${p.yearFromNow} Threshold`} value={p.threshold || 0} onChange={v => { const np = [...params.periods]; np[i] = { ...np[i], threshold: v }; setParams(pr => ({ ...pr, periods: np })); }} min={0} max={50e6} step={500000} format="currency" />
                </div>
              ))}
            </div>

            {/* Global assumptions */}
            <div style={cardStyle}>
              <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="sliders" size={13} color={c.accent} /> Assumptions</h3>
              <ParamSlider theme={theme} label="Current Metric" value={params.currentMetric} onChange={v => setParams(p => ({ ...p, currentMetric: v }))} min={5e6} max={50e6} step={500000} format="currency" tooltip="Latest actual or projected metric value" />
              <ParamSlider theme={theme} label="Metric Growth Rate" value={params.metricGrowthRate} onChange={v => setParams(p => ({ ...p, metricGrowthRate: v }))} min={-0.10} max={0.25} step={0.01} format="percent" tooltip="Expected annual growth rate of the metric" />
              <ParamSlider theme={theme} label="Volatility" value={params.volatility} onChange={v => setParams(p => ({ ...p, volatility: v }))} min={0.10} max={0.75} step={0.01} format="percent" tooltip="Comparable company equity or metric volatility" />
              <ParamSlider theme={theme} label="Metric Discount Rate" value={params.discountRate} onChange={v => setParams(p => ({ ...p, discountRate: v }))} min={0.05} max={0.25} step={0.005} format="percent" tooltip="Rate to discount the metric itself (e.g., WACC for EBITDA). Determines risk premium in risk-neutral simulation." />
              <ParamSlider theme={theme} label="Risk-Free Rate" value={params.riskFreeRate} onChange={v => setParams(p => ({ ...p, riskFreeRate: v }))} min={0.01} max={0.08} step={0.001} format="percent" tooltip="US Treasury yield matching earnout duration" />
              <ParamSlider theme={theme} label="Credit Adj." value={params.creditAdj || 0} onChange={v => setParams(p => ({ ...p, creditAdj: v }))} min={0} max={0.05} step={0.005} format="percent" tooltip="Counterparty credit risk premium (zero if escrowed)" />
              <div style={{ fontSize: 10, color: c.textDim, padding: "6px 0 2px", borderTop: `1px solid ${c.cardBorder}`, marginTop: 4 }}>
                Payoff discount: {fmtPct(params.payoffDiscountRate != null ? params.payoffDiscountRate : (params.riskFreeRate + (params.isEscrowed ? 0 : (params.creditAdj || 0))))} (Rf + Credit)
                <br />Risk-neutral drift: {fmtPct(params.metricGrowthRate - (params.discountRate - params.riskFreeRate))} (Growth − Risk Premium)
              </div>
              <button onClick={() => runValuation(params)} style={{ width: "100%", padding: "9px", background: "linear-gradient(135deg,#2563eb,#1d4ed8)", border: "none", borderRadius: 7, color: "white", cursor: "pointer", fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", gap: 5, marginTop: 8 }}>
                <Icon name="refresh" size={13} color="white" /> Recalculate
              </button>
            </div>
          </div>

          {/* RIGHT: Results */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* KPIs */}
            <div className="vg r-stack" style={{ gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
              {[{ l: "Fair Value", v: results.fairValue, c: c.accent }, { l: "95% CI Low", v: results.ci95[0], c: c.danger }, { l: "95% CI High", v: results.ci95[1], c: c.success }, { l: "Prob. Payoff", v: null, d: `${results.probPayoff}%`, c: c.text }].map((k, i) => (
                <div key={i} style={{ ...cardStyle, padding: 14 }}>
                  <div style={{ fontSize: 10, color: c.textMuted, marginBottom: 4 }}>{k.l}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'IBM Plex Mono',monospace", color: k.c }}>{k.d || <AnimatedValue value={k.v} />}</div>
                </div>
              ))}
            </div>

            {/* Per-period breakdown */}
            {results.periodStats && results.periodStats.length > 1 && (
              <div style={cardStyle}>
                <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 12, display: "flex", alignItems: "center", gap: 5 }}><Icon name="calendar" size={13} color={c.accent} /> Per-Period Fair Value</h3>
                <div className="vf" style={{ gap: 8 }}>
                  {results.periodStats.map((ps, i) => (
                    <div key={i} style={{ flex: 1, textAlign: "center", padding: "10px 8px", background: c.accentLight, borderRadius: 8 }}>
                      <div style={{ fontSize: 10, color: c.textMuted, marginBottom: 4 }}>Period {i + 1}</div>
                      <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'IBM Plex Mono',monospace", color: c.accent }}>{fmt(ps.mean)}</div>
                      <div style={{ fontSize: 9, color: c.textDim, marginTop: 2 }}>P50: {fmt(ps.p50)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Charts */}
            <div className="vg r-stack" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={cardStyle}>
                <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="bar" size={13} color={c.accent} /> Distribution</h3>
                <Histogram data={results.histogram} fairValue={results.fairValue} theme={theme} />
                <div className="vf" style={{ justifyContent: "space-between", marginTop: 5, fontSize: 9, fontFamily: "'IBM Plex Mono',monospace", color: c.textDim }}>
                  <span>{fmt(results.histogram[0]?.x)}</span><span style={{ color: c.warning }}>FV: {fmt(results.fairValue)}</span><span>{fmt(results.histogram[results.histogram.length - 1]?.x)}</span>
                </div>
              </div>
              <div style={cardStyle}>
                <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="activity" size={13} color={c.accent} /> Sensitivity</h3>
                <TornadoChart sensitivities={sensitivities} baseValue={results.fairValue} theme={theme} />
              </div>
            </div>

            {/* Percentiles */}
            <div style={cardStyle}>
              <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="hash" size={13} color={c.accent} /> Percentiles</h3>
              <div className="vg r-stack" style={{ gridTemplateColumns: "repeat(7,1fr)", gap: 6 }}>
                {Object.entries(results.percentiles).map(([k, v]) => (
                  <div key={k} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: c.textDim, marginBottom: 3 }}>{k.replace("p", "")}th</div>
                    <div style={{ height: 48, display: "flex", alignItems: "flex-end", justifyContent: "center", marginBottom: 3 }}>
                      <div style={{ width: "80%", height: `${Math.min(100, (v / (results.percentiles.p95 || 1)) * 100)}%`, background: `linear-gradient(180deg,${c.accent},rgba(37,99,235,0.2))`, borderRadius: "3px 3px 0 0", minHeight: 3 }} />
                    </div>
                    <div style={{ fontSize: 10, fontFamily: "'IBM Plex Mono',monospace", color: c.text }}>{fmt(v)}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Methodology */}
            <div style={{ ...cardStyle, background: tc ? "rgba(37,99,235,0.03)" : "rgba(37,99,235,0.02)", borderColor: tc ? "rgba(37,99,235,0.1)" : "rgba(37,99,235,0.06)" }}>
              <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 5, display: "flex", alignItems: "center", gap: 5 }}><Icon name="book" size={13} color={c.accent} /> Methodology</h3>
              <p style={{ fontSize: 11, color: c.textMuted, lineHeight: 1.65, margin: 0 }}>
                Multi-period Monte Carlo simulation ({MC_PATHS.toLocaleString()} paths) under risk-neutral framework per ASC 820/IFRS 13.
                {params.periods.length} measurement periods modeled with annual GBM metric evolution.
                {params.hasCatchUp ? " Path-dependent catch-up provisions modeled explicitly." : ""}
                {params.hasClawback ? " Clawback evaluated at terminal period." : ""}
                {params.hasAcceleration ? " Stochastic acceleration trigger modeled per-period." : ""}
                {params.isMultiMetric ? " Correlated dual-metric simulation with joint threshold evaluation." : ""}
                {" "}Level 3 fair value hierarchy classification.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return <div style={{ minHeight: "100vh", background: c.bg }}>{fontLink}{hdr()}</div>;
}
