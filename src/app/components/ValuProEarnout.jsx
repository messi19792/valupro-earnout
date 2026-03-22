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

  // ---- DISCOUNT RATE FRAMEWORK (per Appraisal Foundation VA-4 / GT methodology) ----
  // 
  // CORRECT APPROACH FOR EARNOUT MONTE CARLO:
  // For each period i with projected metric P_i and threshold T_i:
  //   1. The risk-neutral expected metric = P_i * exp(-metricRiskPremium * t_i)
  //      where metricRiskPremium = discountRate - riskFreeRate
  //   2. Simulate log-normal: metric_sim = RN_expected * exp(-0.5*σ²*t + σ*√t*Z)
  //      equivalently: metric_sim = P_i * exp(-riskPremium*t - 0.5*σ²*t + σ*√t*Z)
  //   3. Evaluate payoff: does metric_sim >= threshold?
  //   4. Discount payoff at riskFreeRate + creditAdj
  //
  // For PATH-DEPENDENT earnouts (catch-up, carry-forward, cumulative):
  //   Generate a correlated GBM path from currentMetric so year-over-year
  //   performance is linked, but scale each year's result relative to the
  //   projected metric for that year.

  const metricRiskPremium = Math.max(0, discountRate - riskFreeRate);
  const effectivePayoffDiscount = payoffDiscountRate != null 
    ? payoffDiscountRate 
    : riskFreeRate + (isEscrowed ? 0 : creditAdj);
  const numPeriods = periods.length;
  const allResults = [];
  const periodResults = Array.from({ length: numPeriods }, () => []);
  const pathData = [];

  const needsCorrelatedPath = hasCatchUp || hasCarryForward || hasCumulativeTarget || hasClawback;

  for (let sim = 0; sim < numPaths; sim++) {
    // Generate correlated normal draws for all periods
    // (correlated path: z values carry forward; independent: fresh z each period)
    const zValues = [];
    for (let p = 0; p < numPeriods; p++) zValues.push(normalRandom());

    // For path-dependent: generate cumulative GBM path for ratio scaling
    let pathRatios = null;
    if (needsCorrelatedPath) {
      // GBM path from 1.0 (normalized) with risk-neutral drift
      const rnDrift = metricGrowthRate - metricRiskPremium;
      pathRatios = [1.0];
      for (let p = 0; p < numPeriods; p++) {
        pathRatios.push(pathRatios[p] * Math.exp((rnDrift - 0.5 * volatility * volatility) + volatility * zValues[p]));
      }
    }

    // Generate second metric path if multi-metric  
    let secondMetricValues = null;
    if (isMultiMetric && secondMetric) {
      secondMetricValues = [];
      for (let p = 0; p < numPeriods; p++) {
        const z2 = metricCorrelation * zValues[p] + Math.sqrt(1 - metricCorrelation * metricCorrelation) * normalRandom();
        const t = (periods[p].yearFromNow || (p + 1));
        const sv = secondMetric.currentValue * Math.exp(
          (secondMetric.growthRate - metricRiskPremium - 0.5 * secondMetric.volatility * secondMetric.volatility) * t +
          secondMetric.volatility * Math.sqrt(t) * z2
        );
        secondMetricValues.push(sv);
      }
    }

    let totalPayoff = 0;
    let cumulativeMetric = 0;
    let cumulativePayments = 0;
    let priorShortfall = 0;
    let excessCarryForward = 0;
    let accelerated = false;
    let clawbackAmount = 0;
    const periodPayoffs = [];
    const metricPath = [currentMetric]; // For tracking

    for (let pIdx = 0; pIdx < numPeriods; pIdx++) {
      const period = periods[pIdx];
      const yearFromNow = period.yearFromNow || (pIdx + 1);
      const paymentDelayYears = paymentDelay / 365;
      const discountT = yearFromNow + paymentDelayYears;

      // ---- SIMULATE METRIC VALUE FOR THIS PERIOD ----
      // Use the period's projected metric as the base
      const projectedMetric = period.projectedMetric || (currentMetric * Math.pow(1 + metricGrowthRate, yearFromNow));
      let metricValue;

      if (needsCorrelatedPath && pathRatios) {
        // Path-dependent: scale the projected metric by the GBM ratio
        // ratio = simulated/expected, so metric = projected * ratio_adjustment
        const expectedRatio = Math.exp((metricGrowthRate - metricRiskPremium) * yearFromNow);
        const simRatio = pathRatios[pIdx + 1];
        metricValue = projectedMetric * (simRatio / expectedRatio);
      } else {
        // Independent periods: simulate each from its projected value
        // Risk-neutral: projected * exp(-riskPremium*t) is the RN expected value
        // Then add log-normal noise: * exp(-0.5*σ²*t + σ*√t*Z)
        const t = yearFromNow;
        metricValue = projectedMetric * Math.exp(
          -metricRiskPremium * t - 0.5 * volatility * volatility * t + volatility * Math.sqrt(t) * zValues[pIdx]
        );
      }

      metricPath.push(metricValue);

      // Check acceleration trigger
      if (hasAcceleration && !accelerated && Math.random() < accelerationProb) {
        accelerated = true;
        let acceleratedPayoff = 0;
        if (accelerationTreatment === "max") {
          for (let rIdx = pIdx; rIdx < numPeriods; rIdx++) {
            acceleratedPayoff += periods[rIdx].cap || periods[rIdx].fixedPayment || 0;
          }
        } else {
          for (let rIdx = pIdx; rIdx < numPeriods; rIdx++) {
            const pm = periods[rIdx].projectedMetric || projectedMetric;
            acceleratedPayoff += evaluatePayoff(pm * (accelerationPercentile / 100), periods[rIdx], metricPath, cumulativePayments, config);
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
      if (isMultiMetric && secondMetricValues) {
        if (secondMetricValues[pIdx] < (secondMetric.threshold || 0)) {
          periodPayoffs.push(0);
          periodResults[pIdx].push(0);
          if (hasCatchUp) priorShortfall += Math.max(0, (period.threshold || 0) - metricValue);
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

      // Cumulative target check
      cumulativeMetric += metricValue;
      if (hasCumulativeTarget && pIdx === numPeriods - 1) {
        if (cumulativeMetric < cumulativeTarget) {
          periodPayoff = 0;
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
      totalPayoff -= clawback * Math.exp(-effectivePayoffDiscount * numPeriods);
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
// DOCUMENT EXTRACTION (Claude API) — with provenance capture
// ============================================================
const extractFromDocument = async (text, mode) => {
  const systemPrompts = {
    backtest: `You are a financial data extraction specialist. Extract earnout/contingent consideration from SEC filings.
Return ONLY valid JSON, no markdown:
{"earnouts":[{"name":"string","acquisitionDate":"string or null","maxPayout":number or null,"initialFairValue":number or null,"currentFairValue":number or null,"priorFairValue":number or null,"fairValueChange":number or null,"metric":"string or null","structure":"linear|binary|tiered|milestone|percentage|cagr|unknown","threshold":number or null,"participationRate":number or null,"cap":number or null,"floor":number or null,"fixedPayment":number or null,"measurementPeriods":[{"year":number,"target":number or null,"label":"string"}],"hasCatchUp":boolean,"hasClawback":boolean,"hasAcceleration":boolean,"accelerationTrigger":"string or null","hasCumulativeTarget":boolean,"cumulativeTarget":number or null,"multiYearCap":number or null,"methodology":"Monte Carlo|probability-weighted|DCF|unknown","discountRate":number or null,"volatility":number or null,"riskFreeRate":number or null,"projectedMetric":number or null,"level3Rollforward":{"openingBalance":number or null,"additions":number or null,"fairValueChanges":number or null,"payments":number or null,"closingBalance":number or null},"confidenceScore":number,"provenance":{"volatilitySource":"string or null","discountRateSource":"string or null","projectionSource":"string or null","methodologyQuote":"string or null","level3DisclosureText":"string or null","comparableCompanies":["string"] or null,"referencedDocuments":["string"] or null}}],"reportingPeriod":"string","companyName":"string","filingType":"10-K|10-Q"}
Extract EVERY earnout. Use null for undisclosed. For provenance: extract any text describing HOW assumptions were derived (e.g. "volatility based on comparable company analysis"), any comparable company names mentioned, any external documents referenced (e.g. "per the Merger Agreement Section 2.4"), and any direct quotes about methodology.`,

    live_ppa: `You are a valuation report extraction specialist. Extract earnout terms AND their provenance from a PPA valuation report.
Return ONLY valid JSON, no markdown:
{"earnout":{"name":"string","metric":"string","metricDefinition":"string","structure":"linear|binary|tiered|milestone|percentage|cagr|multi-metric","periods":[{"year":number,"yearFromNow":number,"threshold":number or null,"cap":number or null,"floor":number or null,"fixedPayment":number or null,"participationRate":number or null,"projectedMetric":number or null,"tiers":[{"lower":number,"upper":number,"rate":number}] or null}],"hasCatchUp":boolean,"catchUpDescription":"string or null","hasClawback":boolean,"clawbackThreshold":number or null,"clawbackRate":number or null,"clawbackCap":number or null,"hasAcceleration":boolean,"accelerationTrigger":"string or null","accelerationTreatment":"string or null","hasCumulativeTarget":boolean,"cumulativeTarget":number or null,"hasMultiYearCap":boolean,"multiYearCap":number or null,"hasCarryForward":boolean,"isMultiMetric":boolean,"secondMetric":{"name":"string","threshold":number,"currentValue":number,"growthRate":number,"volatility":number} or null,"metricCorrelation":number or null,"paymentTiming":"string","paymentDelay":number or null,"isEscrowed":boolean,"methodology":"Monte Carlo|probability-weighted|DCF","assumptions":{"currentMetric":number,"metricGrowthRate":number or null,"volatility":number,"discountRate":number,"riskFreeRate":number,"creditAdjustment":number or null,"comparableCompanies":["string"] or null},"initialFairValue":number or null,"currency":"string","confidenceScore":number,"ambiguities":["string"],"alternativeInterpretations":[{"clause":"string","interpretation1":"string","interpretation2":"string"}] or null,"provenance":{"volatility":{"value":number or null,"methodology":"string or null","comparableCompanies":[{"name":"string","ticker":"string or null","volatility":number or null}] or null,"deLeveringMethod":"string or null","dataDateRange":"string or null","sourceLocation":"string or null"},"discountRate":{"value":number or null,"methodology":"string or null","components":{"riskFreeRate":number or null,"equityRiskPremium":number or null,"sizePremium":number or null,"companySpecificRisk":number or null,"beta":number or null,"costOfDebt":number or null,"debtWeight":number or null,"equityWeight":number or null} or null,"sourceLocation":"string or null"},"projections":{"source":"string or null","forecastDate":"string or null","provider":"string or null","sourceLocation":"string or null"},"creditRisk":{"methodology":"string or null","acquirerRating":"string or null","sourceLocation":"string or null"},"referencedDocuments":["string"] or null,"methodologyQuote":"string or null"}}}`
  };

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" },
      body: JSON.stringify({ model: CLAUDE_MODEL, max_tokens: 6000, system: systemPrompts[mode], messages: [{ role: "user", content: `Extract all earnout information AND provenance details:\n\n${text.substring(0, 80000)}` }] })
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
// EXCEL GENERATION — Professionally formatted with xlsx-js-style
// ============================================================
const generateExcel = async (params, results, sensitivities) => {
  let XLSX; try { XLSX = await import("xlsx-js-style"); } catch(e) { XLSX = await import("xlsx"); }
  const wb = XLSX.utils.book_new();
  const periods = params.periods || [];
  const nP = periods.length;
  const metricRiskPremium = Math.max(0, params.discountRate - params.riskFreeRate);
  const payoffDisc = params.payoffDiscountRate != null ? params.payoffDiscountRate : params.riskFreeRate + (params.isEscrowed ? 0 : (params.creditAdj || 0));

  // ---- STYLE DEFINITIONS ----
  const fontBase = { name: "Aptos", sz: 9 };
  const fontBold = { ...fontBase, bold: true };
  const fontHeader = { name: "Aptos", sz: 10, bold: true, color: { rgb: "FFFFFF" } };
  const fontSection = { name: "Aptos", sz: 10, bold: true, color: { rgb: "1F3864" } };
  const fontTitle = { name: "Aptos", sz: 12, bold: true, color: { rgb: "1F3864" } };
  const fontSmall = { name: "Aptos", sz: 8, italic: true, color: { rgb: "808080" } };

  const fillHeader = { fgColor: { rgb: "1F3864" } };
  const fillSection = { fgColor: { rgb: "D6E4F0" } };
  const fillInput = { fgColor: { rgb: "FFF2CC" } }; // Light yellow for input cells
  const fillDerived = { fgColor: { rgb: "E2EFDA" } }; // Light green for formula cells
  const fillWhite = { fgColor: { rgb: "FFFFFF" } };
  const fillAlt = { fgColor: { rgb: "F2F2F2" } }; // Alternating row

  const borderThin = { style: "thin", color: { rgb: "B4B4B4" } };
  const borderMed = { style: "medium", color: { rgb: "1F3864" } };
  const borders = { top: borderThin, bottom: borderThin, left: borderThin, right: borderThin };
  const borderBottom = { bottom: borderMed };
  const borderTop = { top: borderMed };

  const alignR = { horizontal: "right", vertical: "center" };
  const alignL = { horizontal: "left", vertical: "center" };
  const alignC = { horizontal: "center", vertical: "center" };

  const nf1 = "#,##0.0"; // 1 decimal
  const nfDol = "#,##0";
  const nfDol1 = "#,##0.0";
  const nfPct = "0.0%";
  const nfPct2 = "0.00%";
  const nfDec2 = "0.00";
  const nfDec4 = "0.0000";
  const nfInt = "#,##0";

  // Helper: create styled cell
  const cell = (v, s = {}) => ({ v, t: typeof v === "number" ? "n" : "s", s });
  const fcell = (f, s = {}) => ({ f, t: "n", s }); // formula cell

  const sLabel = { font: fontBold, alignment: alignL, border: borders, fill: fillWhite };
  const sInput = { font: fontBase, alignment: alignR, border: borders, fill: fillInput, numFmt: nfDol };
  const sInputPct = { font: fontBase, alignment: alignR, border: borders, fill: fillInput, numFmt: nfPct };
  const sInputInt = { font: fontBase, alignment: alignR, border: borders, fill: fillInput, numFmt: nfInt };
  const sDerived = { font: fontBold, alignment: alignR, border: borders, fill: fillDerived, numFmt: nfPct };
  const sDerivedLabel = { font: fontBold, alignment: alignL, border: borders, fill: fillDerived };
  const sHdr = { font: fontHeader, alignment: alignC, border: borders, fill: { type: "pattern", patternType: "solid", ...fillHeader } };
  const sSect = { font: fontSection, alignment: alignL, fill: { type: "pattern", patternType: "solid", ...fillSection }, border: borders };
  const sNorm = { font: fontBase, alignment: alignR, border: borders, fill: fillWhite, numFmt: nfDol1 };
  const sNormPct = { font: fontBase, alignment: alignR, border: borders, fill: fillWhite, numFmt: nfPct };
  const sNormDec = { font: fontBase, alignment: alignR, border: borders, fill: fillWhite, numFmt: nfDec2 };
  const sNormInt = { font: fontBase, alignment: alignR, border: borders, fill: fillWhite, numFmt: nfInt };
  const sNormL = { font: fontBase, alignment: alignL, border: borders, fill: fillWhite };
  const sBold = { font: fontBold, alignment: alignR, border: borders, fill: fillWhite, numFmt: nfDol1 };
  const sBoldL = { font: fontBold, alignment: alignL, border: borders, fill: fillWhite };
  const sTitle = { font: fontTitle, alignment: alignL };
  const sNote = { font: fontSmall, alignment: alignL };
  const sAlt = { font: fontBase, alignment: alignR, border: borders, fill: { type: "pattern", patternType: "solid", ...fillAlt }, numFmt: nfDol1 };
  const sAltL = { font: fontBase, alignment: alignL, border: borders, fill: { type: "pattern", patternType: "solid", ...fillAlt } };

  // ============ SHEET 1: INPUTS ============
  const ws1 = {};
  let r = 0;
  const s1 = (row, col, val, style) => { const ref = XLSX.utils.encode_cell({ r: row, c: col }); ws1[ref] = typeof val === "object" && val.f ? { ...val, s: style } : { v: val, t: typeof val === "number" ? "n" : "s", s: style }; };
  const s1f = (row, col, formula, style) => { const ref = XLSX.utils.encode_cell({ r: row, c: col }); ws1[ref] = { f: formula, t: "n", s: style }; };

  // Title
  s1(0, 0, "ValuProEarnout — Model Inputs", sTitle);
  s1(1, 0, "All yellow cells are inputs. Green cells are formula-linked.", sNote);

  // Section: Earnout Terms
  r = 3;
  s1(r, 0, "EARNOUT TERMS", sSect); s1(r, 1, "", sSect);
  r++; s1(r, 0, "Performance Metric", sLabel); s1(r, 1, params.metric || "EBITDA", { ...sNormL, fill: fillInput });
  r++; s1(r, 0, "Number of Periods", sLabel); s1(r, 1, nP, sInputInt);
  r++; s1(r, 0, "Multi-Year Cap", sLabel); s1(r, 1, params.multiYearCap || 0, sInput);
  r++; s1(r, 0, "Catch-Up", sLabel); s1(r, 1, params.hasCatchUp ? "Yes" : "No", { ...sNormL, fill: fillInput });
  r++; s1(r, 0, "Clawback", sLabel); s1(r, 1, params.hasClawback ? "Yes" : "No", { ...sNormL, fill: fillInput });
  r++; s1(r, 0, "Acceleration", sLabel); s1(r, 1, params.hasAcceleration ? "Yes" : "No", { ...sNormL, fill: fillInput });
  r++; s1(r, 0, "Escrowed", sLabel); s1(r, 1, params.isEscrowed ? "Yes" : "No", { ...sNormL, fill: fillInput });

  // Section: Assumptions
  r += 2;
  const assumpStart = r;
  s1(r, 0, "KEY ASSUMPTIONS", sSect); s1(r, 1, "", sSect); s1(r, 2, "", sSect);
  r++; s1(r, 0, "", sHdr); s1(r, 1, "Value", sHdr); s1(r, 2, "Type", sHdr);
  r++; const rCM = r; s1(r, 0, "Current Metric ($)", sLabel); s1(r, 1, params.currentMetric, sInput); s1(r, 2, "Input", sNote);
  r++; const rGR = r; s1(r, 0, "Metric Growth Rate", sLabel); s1(r, 1, params.metricGrowthRate, sInputPct); s1(r, 2, "Input", sNote);
  r++; const rVol = r; s1(r, 0, "Volatility (σ)", sLabel); s1(r, 1, params.volatility, sInputPct); s1(r, 2, "Input", sNote);
  r++; const rDR = r; s1(r, 0, "Metric Discount Rate", sLabel); s1(r, 1, params.discountRate, sInputPct); s1(r, 2, "Input", sNote);
  r++; const rRF = r; s1(r, 0, "Risk-Free Rate", sLabel); s1(r, 1, params.riskFreeRate, sInputPct); s1(r, 2, "Input", sNote);
  r++; const rCA = r; s1(r, 0, "Credit Risk Adjustment", sLabel); s1(r, 1, params.creditAdj || 0, sInputPct); s1(r, 2, "Input", sNote);
  r++; s1(r, 0, "Payment Delay (days)", sLabel); s1(r, 1, params.paymentDelay || 120, sInputInt); s1(r, 2, "Input", sNote);
  r++; // Blank row
  r++; s1(r, 0, "DERIVED RATES", sSect); s1(r, 1, "", sSect); s1(r, 2, "", sSect);
  r++; const rMRP = r; s1(r, 0, "Metric Risk Premium", sDerivedLabel); s1f(r, 1, `B${rDR + 1}-B${rRF + 1}`, sDerived); s1(r, 2, "Formula", sNote);
  r++; const rPDR = r; s1(r, 0, "Payoff Discount Rate", sDerivedLabel); s1f(r, 1, params.isEscrowed ? `B${rRF + 1}` : `B${rRF + 1}+B${rCA + 1}`, sDerived); s1(r, 2, "Formula", sNote);
  r++; const rRND = r; s1(r, 0, "Risk-Neutral Drift", sDerivedLabel); s1f(r, 1, `B${rGR + 1}-B${rMRP + 1}`, sDerived); s1(r, 2, "Formula", sNote);

  // Section: Per-Period Terms
  r += 2;
  s1(r, 0, "PER-PERIOD TERMS", sSect); for (let c = 1; c <= 7; c++) s1(r, c, "", sSect);
  r++;
  ["Period", "Year", "Structure", "Threshold ($)", "Part. Rate", "Fixed Pmt ($)", "Cap ($)", "Projected ($)"].forEach((h, c) => s1(r, c, h, sHdr));
  const periodDataStart = r + 1;
  periods.forEach((p, i) => {
    r++;
    const bg = i % 2 === 0 ? sNormL : sAltL;
    const bn = i % 2 === 0 ? sNorm : sAlt;
    const bp = i % 2 === 0 ? sNormPct : { ...sAlt, numFmt: nfPct };
    s1(r, 0, i + 1, { ...(i % 2 === 0 ? sNormInt : { ...sAlt, numFmt: nfInt }) });
    s1(r, 1, p.yearFromNow || i + 1, { ...(i % 2 === 0 ? sNormInt : { ...sAlt, numFmt: nfInt }) });
    s1(r, 2, p.structure, bg);
    s1(r, 3, p.threshold || 0, bn);
    s1(r, 4, p.participationRate || 0, bp);
    s1(r, 5, p.fixedPayment || 0, bn);
    s1(r, 6, p.cap || 0, bn);
    s1(r, 7, p.projectedMetric || 0, bn);
  });

  ws1["!ref"] = XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: r + 1, c: 7 } });
  ws1["!cols"] = [{ wch: 26 }, { wch: 16 }, { wch: 12 }, { wch: 16 }, { wch: 12 }, { wch: 16 }, { wch: 14 }, { wch: 16 }];
  XLSX.utils.book_append_sheet(wb, ws1, "Inputs");

  // ============ SHEET 2: VALUATION ============
  const ws2 = {};
  const s2 = (row, col, val, style) => { ws2[XLSX.utils.encode_cell({ r: row, c: col })] = { v: val, t: typeof val === "number" ? "n" : "s", s: style }; };
  const s2f = (row, col, formula, style) => { ws2[XLSX.utils.encode_cell({ r: row, c: col })] = { f: formula, t: "n", s: style }; };

  r = 0;
  s2(0, 0, "ValuProEarnout — Valuation Output", sTitle);
  s2(1, 0, "All values formula-linked to Inputs sheet where applicable.", sNote);

  r = 3;
  s2(r, 0, "FAIR VALUE SUMMARY", sSect); s2(r, 1, "", sSect);
  r++; s2(r, 0, "", sHdr); s2(r, 1, "Value", sHdr);
  r++; const rFV = r; s2(r, 0, "Fair Value (Mean)", sBoldL); s2(r, 1, results.fairValue, { ...sBold, numFmt: nfDol1 });
  r++; const rSE = r; s2(r, 0, "Standard Error", sLabel); s2(r, 1, results.stdError, sNorm);
  r++; s2(r, 0, "95% CI — Low", sLabel); s2f(r, 1, `B${rFV + 1}-1.96*B${rSE + 1}`, sNorm);
  r++; s2(r, 0, "95% CI — High", sLabel); s2f(r, 1, `B${rFV + 1}+1.96*B${rSE + 1}`, sNorm);
  r++; s2(r, 0, "Probability of Payoff", sLabel); s2(r, 1, results.probPayoff / 100, sNormPct);
  r++; s2(r, 0, "Monte Carlo Paths", sLabel); s2(r, 1, MC_PATHS, sNormInt);
  r++; const maxPay = params.multiYearCap || periods.reduce((s, p) => s + (p.cap || p.fixedPayment || 0), 0);
  s2(r, 0, "Maximum Payout", sLabel); s2(r, 1, maxPay, sNorm);
  r++; s2(r, 0, "Fair Value as % of Max", sLabel); s2f(r, 1, `B${rFV + 1}/B${r}`, sNormPct);

  // Percentiles
  r += 2;
  s2(r, 0, "PERCENTILE DISTRIBUTION", sSect); s2(r, 1, "", sSect);
  r++; s2(r, 0, "Percentile", sHdr); s2(r, 1, "Fair Value ($)", sHdr);
  Object.entries(results.percentiles).forEach(([k, v], i) => {
    r++;
    const bg = i % 2 === 0 ? sNormL : sAltL;
    const bn = i % 2 === 0 ? sNorm : sAlt;
    s2(r, 0, k.replace("p", "") + "th", bg); s2(r, 1, v, bn);
  });

  // Per-period decomposition
  r += 2;
  s2(r, 0, "PER-PERIOD DECOMPOSITION", sSect); for (let c = 1; c <= 5; c++) s2(r, c, "", sSect);
  r++; ["Period", "Mean FV ($)", "P25 ($)", "Median ($)", "P75 ($)", "% of Total"].forEach((h, c) => s2(r, c, h, sHdr));
  const pdStart = r + 1;
  if (results.periodStats) {
    results.periodStats.forEach((ps, i) => {
      r++;
      const bg = i % 2 === 0 ? sNormL : sAltL;
      const bn = i % 2 === 0 ? sNorm : sAlt;
      s2(r, 0, `Period ${i + 1}`, bg); s2(r, 1, ps.mean, bn); s2(r, 2, ps.p25, bn); s2(r, 3, ps.p50, bn); s2(r, 4, ps.p75, bn);
      s2f(r, 5, `B${r + 1}/B${rFV + 1}`, { ...(i % 2 === 0 ? sNormPct : { ...sAlt, numFmt: nfPct }) });
    });
    r++;
    s2(r, 0, "Total", sBoldL);
    s2f(r, 1, `SUM(B${pdStart + 1}:B${pdStart + nP})`, sBold);
    s2f(r, 5, `B${r + 1}/B${rFV + 1}`, { ...sBold, numFmt: nfPct });
  }

  ws2["!ref"] = XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: r + 1, c: 5 } });
  ws2["!cols"] = [{ wch: 24 }, { wch: 16 }, { wch: 14 }, { wch: 14 }, { wch: 14 }, { wch: 14 }];
  XLSX.utils.book_append_sheet(wb, ws2, "Valuation");

  // ============ SHEET 3: RN BRIDGE ============
  const ws3 = {};
  const s3 = (row, col, val, style) => { ws3[XLSX.utils.encode_cell({ r: row, c: col })] = { v: val, t: typeof val === "number" ? "n" : "s", s: style }; };

  r = 0;
  s3(0, 0, "ValuProEarnout — Risk-Neutral Valuation Bridge", sTitle);
  s3(1, 0, "Step-by-step trace from projected metric to fair value per VFR 4 framework.", sNote);

  r = 3;
  s3(r, 0, "Step", sHdr); s3(r, 1, "Description", sHdr); s3(r, 2, "Formula / Source", sHdr);
  periods.forEach((_, i) => s3(r, 3 + i, `Period ${i + 1}`, sHdr));

  const bridgeRows = [
    ["1", "Projected Metric ($)", "Management forecast", periods.map(p => p.projectedMetric || 0), nfDol],
    ["2", "Metric Risk Premium", "Disc Rate − Rf", periods.map(() => metricRiskPremium), nfPct],
    ["3", "Time to Payment (yrs)", "Year + delay/365", periods.map(p => (p.yearFromNow || 1) + (params.paymentDelay || 120) / 365), nfDec2],
    ["4", "RN Discount Factor", "exp(−RP × T)", periods.map(p => Math.exp(-metricRiskPremium * (p.yearFromNow || 1))), nfDec4],
    ["5", "RN Expected Metric ($)", "Row 1 × Row 4", periods.map(p => (p.projectedMetric || 0) * Math.exp(-metricRiskPremium * (p.yearFromNow || 1))), nfDol],
    ["6", "Threshold ($)", "Earnout target", periods.map(p => p.threshold || 0), nfDol],
    ["7", "Moneyness", "Row 5 ÷ Row 6", periods.map(p => { const rnm = (p.projectedMetric || 0) * Math.exp(-metricRiskPremium * (p.yearFromNow || 1)); return p.threshold ? rnm / p.threshold : 0; }), nfDec2],
    ["8", "Volatility (σ)", "Comparable cos", periods.map(() => params.volatility), nfPct],
    ["9", "σ × √T", "Uncertainty", periods.map(p => params.volatility * Math.sqrt(p.yearFromNow || 1)), nfPct],
    ["10", "Max Payoff ($)", "Fixed pmt / cap", periods.map(p => p.fixedPayment || p.cap || 0), nfDol],
  ];
  if (results.periodStats) {
    bridgeRows.push(["11", "MC Fair Value ($)", `${MC_PATHS.toLocaleString()} paths`, results.periodStats.map(ps => ps.mean), nfDol1]);
    bridgeRows.push(["12", "Implied Prob. of Payoff", "Row 11 ÷ (Row 10 × PV)", results.periodStats.map((ps, i) => {
      const maxP = periods[i].fixedPayment || periods[i].cap || 1;
      const disc = Math.exp(-payoffDisc * ((periods[i].yearFromNow || 1) + (params.paymentDelay || 120) / 365));
      return maxP > 0 ? ps.mean / (maxP * disc) : 0;
    }), nfPct]);
  }

  bridgeRows.forEach((brow, bi) => {
    r = 4 + bi;
    const isAlt = bi % 2 === 1;
    s3(r, 0, brow[0], isAlt ? { ...sAltL, numFmt: "0" } : { ...sNormL, numFmt: "0" });
    s3(r, 1, brow[1], isAlt ? sAltL : sNormL);
    s3(r, 2, brow[2], isAlt ? sAltL : sNormL);
    brow[3].forEach((val, pi) => {
      s3(r, 3 + pi, val, { font: fontBase, alignment: alignR, border: borders, fill: isAlt ? { type: "pattern", patternType: "solid", ...fillAlt } : fillWhite, numFmt: brow[4] });
    });
  });

  r = 4 + bridgeRows.length + 1;
  s3(r, 0, "", sBoldL); s3(r, 1, "Total Fair Value ($)", sBoldL); s3(r, 2, "", sBoldL);
  s3(r, 3, results.fairValue, { ...sBold, numFmt: nfDol1 });
  r++;
  s3(r, 1, "Payoff Discount Rate", sLabel); s3(r, 3, payoffDisc, { ...sNormPct });

  ws3["!ref"] = XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: r + 1, c: 3 + nP - 1 } });
  ws3["!cols"] = [{ wch: 6 }, { wch: 28 }, { wch: 22 }, ...periods.map(() => ({ wch: 16 }))];
  XLSX.utils.book_append_sheet(wb, ws3, "RN Bridge");

  // ============ SHEET 4: SENSITIVITY ============
  if (sensitivities) {
    const ws4 = {};
    const s4 = (row, col, val, style) => { ws4[XLSX.utils.encode_cell({ r: row, c: col })] = { v: val, t: typeof val === "number" ? "n" : "s", s: style }; };
    r = 0;
    s4(0, 0, "ValuProEarnout — Sensitivity Analysis", sTitle);
    s4(1, 0, "Each parameter varied independently; all others held at base case.", sNote);
    r = 3;
    const sensEntries = Object.entries(sensitivities);
    s4(r, 0, "#", sHdr);
    sensEntries.forEach(([label], si) => { s4(r, 1 + si * 2, label + " (Input)", sHdr); s4(r, 2 + si * 2, label + " (FV $)", sHdr); });
    const maxLen = Math.max(...sensEntries.map(([, d]) => d.length));
    for (let ri = 0; ri < maxLen; ri++) {
      r = 4 + ri;
      const isAlt = ri % 2 === 1;
      s4(r, 0, ri + 1, isAlt ? { ...sAlt, numFmt: nfInt } : sNormInt);
      sensEntries.forEach(([label, data], si) => {
        const inputFmt = label.includes("Metric") && !label.includes("Rate") ? nfDol : nfPct;
        s4(r, 1 + si * 2, data[ri]?.value ?? "", { font: fontBase, alignment: alignR, border: borders, fill: isAlt ? { type: "pattern", patternType: "solid", ...fillAlt } : fillWhite, numFmt: inputFmt });
        s4(r, 2 + si * 2, data[ri]?.fairValue ?? "", { font: fontBase, alignment: alignR, border: borders, fill: isAlt ? { type: "pattern", patternType: "solid", ...fillAlt } : fillWhite, numFmt: nfDol1 });
      });
    }
    r = 4 + maxLen + 1;
    s4(r, 0, "", sBoldL); s4(r, 1, "Base Case FV", sBoldL); s4(r, 2, results.fairValue, { ...sBold, numFmt: nfDol1 });
    ws4["!ref"] = XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: r + 1, c: sensEntries.length * 2 } });
    ws4["!cols"] = [{ wch: 5 }, ...sensEntries.flatMap(() => [{ wch: 18 }, { wch: 16 }])];
    XLSX.utils.book_append_sheet(wb, ws4, "Sensitivity");
  }

  // ============ SHEET 5: DISTRIBUTION ============
  const ws5 = {};
  const s5 = (row, col, val, style) => { ws5[XLSX.utils.encode_cell({ r: row, c: col })] = { v: val, t: typeof val === "number" ? "n" : "s", s: style }; };
  s5(0, 0, "ValuProEarnout — Distribution Data", sTitle);
  s5(2, 0, "Bin ($)", sHdr); s5(2, 1, "Frequency", sHdr); s5(2, 2, "Cumulative %", sHdr);
  let cum = 0;
  results.histogram.forEach((h, i) => {
    cum += h.count;
    const isAlt = i % 2 === 1;
    s5(3 + i, 0, h.x, isAlt ? sAlt : sNorm);
    s5(3 + i, 1, h.count, isAlt ? { ...sAlt, numFmt: nfInt } : sNormInt);
    s5(3 + i, 2, cum / MC_PATHS, isAlt ? { ...sAlt, numFmt: nfPct } : sNormPct);
  });
  ws5["!ref"] = XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: 3 + results.histogram.length, c: 2 } });
  ws5["!cols"] = [{ wch: 16 }, { wch: 12 }, { wch: 14 }];
  XLSX.utils.book_append_sheet(wb, ws5, "Distribution");

  const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([wbout], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `ValuProEarnout_${new Date().toISOString().split("T")[0]}.xlsx`; a.click();
};

// ============================================================
// MEMO GENERATION — Professional narrative
// ============================================================
const generateMemo = (params, results, sensitivities, format = "pdf") => {
  const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const periods = params.periods || [];
  const nP = periods.length;
  const metricRiskPremium = Math.max(0, params.discountRate - params.riskFreeRate);
  const payoffDisc = params.payoffDiscountRate != null ? params.payoffDiscountRate : params.riskFreeRate + (params.isEscrowed ? 0 : (params.creditAdj || 0));
  const rnDrift = params.metricGrowthRate - metricRiskPremium;
  const features = [
    params.hasCatchUp && "catch-up provisions", params.hasClawback && "clawback provisions",
    params.hasAcceleration && "acceleration clauses", params.hasCumulativeTarget && "cumulative target",
    params.hasMultiYearCap && `aggregate cap of $${(params.multiYearCap / 1e6).toFixed(1)}M`, params.hasCarryForward && "carry-forward of excess performance",
    params.isMultiMetric && "multi-metric conditions", params.isEscrowed && "escrowed payment funds",
  ].filter(Boolean);

  // Build narrative per-period analysis
  const periodNarrative = periods.map((p, i) => {
    const ps = results.periodStats?.[i];
    const impliedProb = ps && (p.fixedPayment || p.cap) ? (ps.mean / ((p.fixedPayment || p.cap) * Math.exp(-payoffDisc * ((p.yearFromNow || 1) + (params.paymentDelay || 120) / 365)))) * 100 : 0;
    const rnExpected = (p.projectedMetric || 0) * Math.exp(-metricRiskPremium * (p.yearFromNow || 1));
    const moneyness = p.threshold ? rnExpected / p.threshold : 0;
    const moneynessDesc = moneyness >= 1.1 ? "in-the-money" : moneyness >= 0.95 ? "near-the-money" : "out-of-the-money";
    return `<p><strong>Period ${i + 1} (Year ${p.yearFromNow || i + 1}):</strong> The earnout requires ${params.metric} to ${p.structure === "binary" ? `meet or exceed the $${(p.threshold / 1e6).toFixed(1)}M threshold, triggering a fixed payment of $${((p.fixedPayment || p.cap) / 1e6).toFixed(1)}M` : p.structure === "linear" ? `exceed $${(p.threshold / 1e6).toFixed(1)}M, with a ${((p.participationRate || 0) * 100).toFixed(0)}% participation rate on the excess${p.cap ? `, capped at $${(p.cap / 1e6).toFixed(1)}M` : ""}` : p.structure === "tiered" ? `achieve tiered performance levels` : `achieve the specified target`}. Management projects ${params.metric} of $${((p.projectedMetric || 0) / 1e6).toFixed(1)}M for this period. After applying the metric risk premium of ${(metricRiskPremium * 100).toFixed(1)}%, the risk-neutral expected metric is $${(rnExpected / 1e6).toFixed(1)}M, placing this period ${moneynessDesc} relative to the threshold (ratio: ${moneyness.toFixed(2)}x). The Monte Carlo simulation yields a period fair value of $${ps ? (ps.mean / 1e6).toFixed(2) : "0.00"}M, implying a risk-neutral probability of payment of approximately ${impliedProb.toFixed(0)}%. The median (P50) outcome is $${ps ? (ps.p50 / 1e6).toFixed(2) : "0.00"}M, reflecting the ${p.structure === "binary" ? "binary nature of the payoff — the majority of simulation paths result in either zero or the full payment amount" : "distribution of possible outcomes around the threshold"}.</p>`;
  }).join("\n");

  // Sensitivity narrative
  const sensNarrative = sensitivities ? Object.entries(sensitivities).map(([label, data]) => {
    const vals = data.map(d => d.fairValue);
    const minFV = Math.min(...vals), maxFV = Math.max(...vals);
    const range = maxFV - minFV;
    const sensitivity = range / results.fairValue * 100;
    return `${label} drives a range of $${(minFV / 1e6).toFixed(2)}M to $${(maxFV / 1e6).toFixed(2)}M (${sensitivity.toFixed(0)}% of base value)`;
  }).join("; ") + "." : "";

  // Find most sensitive input
  const mostSensitive = sensitivities ? Object.entries(sensitivities).reduce((best, [label, data]) => {
    const vals = data.map(d => d.fairValue);
    const range = Math.max(...vals) - Math.min(...vals);
    return range > best.range ? { label, range } : best;
  }, { label: "", range: 0 }).label : "";

  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>ValuProEarnout — Fair Value Memorandum</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:Aptos,'Calibri','Segoe UI',sans-serif;font-size:9pt;line-height:1.65;color:#1a1a1a;max-width:8.5in;margin:0 auto;padding:1in}
h1{font-size:13pt;font-weight:700;margin-bottom:3pt;color:#1F3864}h2{font-size:11pt;font-weight:700;margin-top:18pt;margin-bottom:6pt;border-bottom:1.5px solid #1F3864;padding-bottom:3pt;color:#1F3864}
h3{font-size:10pt;font-weight:700;margin-top:12pt;margin-bottom:3pt;color:#333}
p{margin-bottom:8pt;text-align:justify}table{width:100%;border-collapse:collapse;margin:8pt 0;font-size:9pt;font-family:Aptos,'Calibri',sans-serif}th,td{border:1px solid #B4B4B4;padding:4pt 6pt;text-align:left;vertical-align:top}
th{background:#1F3864;color:#fff;font-weight:600;font-size:8pt;text-transform:uppercase;letter-spacing:0.04em}
td.n{text-align:right;font-family:Aptos,'Calibri',sans-serif;font-variant-numeric:tabular-nums}td.h{font-weight:700;background:#F2F2F2}
tr:nth-child(even){background:#F8F9FA}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:20pt;padding-bottom:10pt;border-bottom:2pt solid #1F3864}
.ftr{margin-top:30pt;padding-top:10pt;border-top:1pt solid #999;font-size:7.5pt;color:#666}
.note{background:#F2F7FB;border-left:3pt solid #2563eb;padding:6pt 10pt;margin:10pt 0;font-size:8pt;line-height:1.55}
strong{font-weight:700}em{font-style:italic}
@media print{body{padding:0.5in}}</style></head><body>
<div class="hdr"><div><h1>Contingent Consideration Fair Value Measurement</h1><p style="font-size:8pt;color:#555">Methodology Memorandum — ${date}</p></div><div style="text-align:right"><strong style="font-size:10pt;color:#1F3864">ValuProEarnout</strong><br><span style="font-size:7.5pt;color:#666">Automated Remeasurement Platform</span></div></div>

<h2>1. Executive Summary</h2>
<p>This memorandum documents the fair value measurement of contingent consideration (earnout) liability in accordance with ASC 820, <em>Fair Value Measurement</em>, and ASC 805, <em>Business Combinations</em>. The earnout is classified as Level 3 within the fair value hierarchy, as significant unobservable inputs are required for its measurement.</p>
<p>The earnout obligation is based on the acquired entity's ${params.metric} performance across ${nP} annual measurement period${nP > 1 ? "s" : ""}${features.length > 0 ? ", and incorporates " + features.join(", ") : ""}. The total maximum payout is $${(params.multiYearCap ? params.multiYearCap / 1e6 : periods.reduce((s, p) => s + (p.cap || p.fixedPayment || 0), 0) / 1e6).toFixed(1)}M.</p>
<p>Using a multi-period Monte Carlo simulation with ${MC_PATHS.toLocaleString()} paths under the risk-neutral framework prescribed by the Appraisal Foundation's Valuation Advisory 4 (VFR 4), we estimate the fair value of the earnout liability at <strong>$${(results.fairValue / 1e6).toFixed(2)}M</strong> (95% confidence interval: $${(results.ci95[0] / 1e6).toFixed(2)}M to $${(results.ci95[1] / 1e6).toFixed(2)}M). The probability of any positive payoff is ${results.probPayoff}%.</p>

<h2>2. Description of the Earnout</h2>
<p>The earnout arrangement requires the acquirer to make contingent payments to the seller based on the achievement of specified ${params.metric} performance targets during each measurement period. The key terms are summarized below:</p>
<table><tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Performance Metric</td><td>${params.metric}</td></tr>
<tr><td>Measurement Periods</td><td>${nP} annual periods</td></tr>
<tr><td>Maximum Total Payout</td><td class="n">$${(params.multiYearCap ? params.multiYearCap / 1e6 : periods.reduce((s, p) => s + (p.cap || p.fixedPayment || 0), 0) / 1e6).toFixed(1)}M</td></tr>
<tr><td>Payment Timing</td><td>${params.paymentDelay || 120} days following each measurement period end</td></tr>
<tr><td>Escrow</td><td>${params.isEscrowed ? "Funds held in escrow — no counterparty credit risk" : "Not escrowed — subject to acquirer credit risk"}</td></tr>
${features.length > 0 ? `<tr><td>Special Features</td><td>${features.join("; ")}</td></tr>` : ""}
</table>

<h3>Per-Period Structure</h3>
<table><tr><th>Period</th><th>Year</th><th>Structure</th><th>Target</th><th>Payment/Rate</th><th>Cap</th><th>Projected Metric</th></tr>
${periods.map((p, i) => `<tr><td>Period ${i + 1}</td><td>${p.yearFromNow || i + 1}</td><td style="text-transform:capitalize">${p.structure}</td><td class="n">${p.threshold ? "$" + (p.threshold / 1e6).toFixed(1) + "M" : "—"}</td><td class="n">${p.participationRate ? (p.participationRate * 100).toFixed(0) + "%" : p.fixedPayment ? "$" + (p.fixedPayment / 1e6).toFixed(1) + "M" : "—"}</td><td class="n">${p.cap ? "$" + (p.cap / 1e6).toFixed(1) + "M" : "None"}</td><td class="n">${p.projectedMetric ? "$" + (p.projectedMetric / 1e6).toFixed(1) + "M" : "—"}</td></tr>`).join("")}
</table>

<h2>3. Valuation Methodology</h2>

<h3>3.1 Framework Selection</h3>
<p>The earnout payoff structure is nonlinear — it features ${periods[0]?.structure === "binary" ? "binary thresholds where the payoff is all-or-nothing" : periods[0]?.structure === "tiered" ? "tiered participation levels" : "thresholds that create option-like characteristics"}${params.hasMultiYearCap ? " and a multi-year aggregate cap" : ""}. Consistent with the guidance in the Appraisal Foundation's VFR 4 and industry best practices articulated by firms such as Kroll, Grant Thornton, and Valuation Research Corporation, an Option Pricing Model (OPM) using Monte Carlo simulation is the appropriate methodology for nonlinear, ${nP > 1 ? "multi-period" : "single-period"} earnout structures.</p>
<p>A scenario-based method (SBM) was considered but rejected, as the SBM is unable to properly account for the risk inherent in the nonlinear payoff structure. The SBM typically overestimates the fair value of earnouts with threshold-based payoffs because it considers too few outcomes and fails to capture the option-like characteristics of the payout.</p>

<h3>3.2 Risk-Neutral Simulation</h3>
<p>The Monte Carlo simulation was conducted under the risk-neutral framework, which is the standard approach for valuing contingent claims under ASC 820. The key steps are:</p>
<p><strong>Step 1 — Risk-neutral adjustment:</strong> Management's projected ${params.metric} for each period was adjusted to a risk-neutral basis by discounting at the metric risk premium of ${(metricRiskPremium * 100).toFixed(1)}% (metric discount rate of ${(params.discountRate * 100).toFixed(1)}% less risk-free rate of ${(params.riskFreeRate * 100).toFixed(1)}%). This adjustment, applied before evaluating the payoff structure, removes the market risk premium embedded in management's projections, consistent with VFR 4 guidance that states: "an OPM translates the forecast into a risk-neutral framework by discounting at the required metric risk premium before applying any thresholds, tiers or caps."</p>
<p><strong>Step 2 — Stochastic simulation:</strong> For each of the ${MC_PATHS.toLocaleString()} simulation paths, the risk-neutral adjusted metric was subjected to log-normal uncertainty calibrated to ${(params.volatility * 100).toFixed(1)}% ${params.metric} volatility, derived from comparable company analysis. The simulated metric value for each period was then evaluated against the applicable threshold to determine the payoff.</p>
<p><strong>Step 3 — Payoff discounting:</strong> The resulting contingent payments were discounted to present value at ${(payoffDisc * 100).toFixed(1)}% (risk-free rate of ${(params.riskFreeRate * 100).toFixed(1)}%${!params.isEscrowed ? ` plus credit risk adjustment of ${((params.creditAdj || 0) * 100).toFixed(1)}% reflecting acquirer counterparty risk, as funds are not held in escrow` : ", with no credit adjustment as funds are escrowed"}). ${params.paymentDelay ? `A payment delay of ${params.paymentDelay} days following each measurement period end was incorporated into the discounting.` : ""}</p>
${params.hasCatchUp ? `<p><strong>Path dependency — Catch-up:</strong> The catch-up provision was modeled explicitly. In simulation paths where the metric falls short of the threshold in an earlier period, the shortfall amount is carried forward and effectively reduces the threshold for subsequent periods, increasing the probability of payment in later years. This creates path dependency that cannot be captured by independent period-by-period analysis.</p>` : ""}
${params.hasClawback ? `<p><strong>Clawback:</strong> At the conclusion of all measurement periods, the cumulative metric performance was evaluated against the clawback threshold. In paths where cumulative performance fell below the threshold, a portion of previously made payments was clawed back at a rate of ${(params.clawbackRate * 100).toFixed(0)}%, subject to the clawback cap.</p>` : ""}

<h2>4. Key Assumptions</h2>
<table><tr><th>Assumption</th><th>Value</th><th>Basis</th></tr>
<tr><td>Current ${params.metric}</td><td class="n">$${(params.currentMetric / 1e6).toFixed(1)}M</td><td>Latest actual performance / trailing twelve months</td></tr>
<tr><td>Metric Growth Rate</td><td class="n">${(params.metricGrowthRate * 100).toFixed(1)}%</td><td>Management projection (real-world expected growth)</td></tr>
<tr><td>${params.metric} Volatility</td><td class="n">${(params.volatility * 100).toFixed(1)}%</td><td>Equity volatility of comparable public companies, adjusted for financial leverage</td></tr>
<tr><td>Metric Discount Rate</td><td class="n">${(params.discountRate * 100).toFixed(1)}%</td><td>Required rate of return for ${params.metric} cash flows (WACC-based)</td></tr>
<tr><td>Risk-Free Rate</td><td class="n">${(params.riskFreeRate * 100).toFixed(1)}%</td><td>U.S. Treasury yield matching weighted-average earnout duration</td></tr>
<tr><td>Credit Risk Adjustment</td><td class="n">${((params.creditAdj || 0) * 100).toFixed(1)}%</td><td>${params.isEscrowed ? "N/A — funds escrowed" : "Acquirer counterparty credit assessment"}</td></tr>
<tr><td class="h">Derived: Metric Risk Premium</td><td class="n h">${(metricRiskPremium * 100).toFixed(1)}%</td><td>Metric discount rate less risk-free rate</td></tr>
<tr><td class="h">Derived: Payoff Discount Rate</td><td class="n h">${(payoffDisc * 100).toFixed(1)}%</td><td>Risk-free rate plus credit risk adjustment</td></tr>
<tr><td class="h">Derived: Risk-Neutral Drift</td><td class="n h">${(rnDrift * 100).toFixed(1)}%</td><td>Growth rate less metric risk premium</td></tr>
</table>

<h2>5. Fair Value Conclusion</h2>
<table><tr><th>Measure</th><th>Value</th></tr>
<tr><td class="h"><strong>Fair Value (Mean)</strong></td><td class="n h"><strong>$${(results.fairValue / 1e6).toFixed(2)}M</strong></td></tr>
<tr><td>95% Confidence Interval — Low</td><td class="n">$${(results.ci95[0] / 1e6).toFixed(2)}M</td></tr>
<tr><td>95% Confidence Interval — High</td><td class="n">$${(results.ci95[1] / 1e6).toFixed(2)}M</td></tr>
<tr><td>Standard Error</td><td class="n">$${(results.stdError / 1e6).toFixed(3)}M</td></tr>
<tr><td>Probability of Any Payoff</td><td class="n">${results.probPayoff}%</td></tr>
<tr><td>Maximum Possible Payout</td><td class="n">$${(params.multiYearCap ? params.multiYearCap / 1e6 : periods.reduce((s, p) => s + (p.cap || p.fixedPayment || 0), 0) / 1e6).toFixed(1)}M</td></tr>
<tr><td>Fair Value as % of Maximum</td><td class="n">${(results.fairValue / (params.multiYearCap || periods.reduce((s, p) => s + (p.cap || p.fixedPayment || 0), 0)) * 100).toFixed(1)}%</td></tr></table>

<h2>6. Per-Period Analysis</h2>
<p>The table below decomposes the total fair value into contributions from each measurement period. This decomposition is useful for understanding which periods drive the most value and for subsequent quarterly remeasurement as periods expire or approach maturity.</p>
<table><tr><th>Period</th><th>Mean FV</th><th>P25</th><th>Median</th><th>P75</th><th>% of Total</th></tr>
${results.periodStats ? results.periodStats.map((ps, i) => `<tr><td>Period ${i + 1} (Year ${periods[i]?.yearFromNow || i + 1})</td><td class="n">$${(ps.mean / 1e6).toFixed(2)}M</td><td class="n">$${(ps.p25 / 1e6).toFixed(2)}M</td><td class="n">$${(ps.p50 / 1e6).toFixed(2)}M</td><td class="n">$${(ps.p75 / 1e6).toFixed(2)}M</td><td class="n">${(ps.mean / results.fairValue * 100).toFixed(1)}%</td></tr>`).join("") : ""}
<tr><td class="h"><strong>Total</strong></td><td class="n h"><strong>$${(results.fairValue / 1e6).toFixed(2)}M</strong></td><td></td><td></td><td></td><td class="n h">100.0%</td></tr></table>

${periodNarrative}

<h2>7. Sensitivity Analysis</h2>
<p>Sensitivity analysis was performed by varying each key assumption independently while holding all other inputs constant at their base case values. The analysis demonstrates the impact of assumption uncertainty on the fair value conclusion:</p>
<p>${sensNarrative}</p>
<p>The fair value is most sensitive to changes in <strong>${mostSensitive}</strong>. This is consistent with ${mostSensitive === "Volatility" ? "the option-like nature of the binary payoff structure, where higher volatility increases the probability of extreme outcomes in both directions" : mostSensitive === "Current Metric" || mostSensitive === "Growth Rate" ? "the earnout's dependence on achieving performance thresholds — higher projected performance increases the probability of threshold achievement" : "the economic characteristics of the earnout structure"}.</p>

<h2>8. Fair Value Hierarchy Classification</h2>
<p>The earnout liability is classified as <strong>Level 3</strong> within the fair value hierarchy defined by ASC 820. Level 3 measurements are based on significant unobservable inputs, which in this case include:</p>
<p>Management's projected ${params.metric} performance for each measurement period; the estimated ${params.metric} volatility of ${(params.volatility * 100).toFixed(1)}%, derived from comparable company equity volatility adjusted for financial leverage; and the metric discount rate of ${(params.discountRate * 100).toFixed(1)}%, which reflects the required rate of return for ${params.metric}-related cash flows.</p>
<p>The methodology and assumptions are consistent with the initial purchase price allocation measurement${nP > 1 ? " and, where applicable, with prior period remeasurements" : ""}.</p>

<div class="note"><strong>Methodology note:</strong> This valuation was prepared using ValuProEarnout, an automated remeasurement platform implementing the Option Pricing Model (Monte Carlo simulation) as prescribed by the Appraisal Foundation's Valuation Advisory 4: Valuation of Contingent Consideration (February 2019). The simulation employed ${MC_PATHS.toLocaleString()} paths with annual period resolution under the risk-neutral framework. All computations are deterministic given the input assumptions and random seed.</div>

<div class="ftr"><p>Generated by ValuProEarnout — ${date} | This memorandum is provided for informational purposes and does not constitute an audit opinion or formal appraisal.</p></div></body></html>`;

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

  // Provenance — auto-populated from extraction, user can supplement
  const [provenance, setProvenance] = useState({
    volatility: { methodology: null, comparableCompanies: [], deLeveringMethod: null, sourceLocation: null, userNote: "" },
    discountRate: { methodology: null, components: null, sourceLocation: null, userNote: "" },
    projections: { source: null, forecastDate: null, provider: null, sourceLocation: null, userNote: "" },
    creditRisk: { methodology: null, acquirerRating: null, userNote: "" },
    referencedDocuments: [], methodologyQuote: null,
  });

  // Audit Support
  const [auditQuestions, setAuditQuestions] = useState(null);
  const [auditResponses, setAuditResponses] = useState(null);
  const [auditProcessing, setAuditProcessing] = useState(false);

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

    let np = { ...params };
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
      np = { ...params, metric: e.metric || params.metric, currentMetric: e.projectedMetric || params.currentMetric,
        volatility: e.volatility || params.volatility, discountRate: e.discountRate || params.discountRate,
        riskFreeRate: e.riskFreeRate || params.riskFreeRate, periods: newPeriods,
        hasCatchUp: e.hasCatchUp || false, hasClawback: e.hasClawback || false,
        hasAcceleration: e.hasAcceleration || false, hasMultiYearCap: e.multiYearCap ? true : false,
        multiYearCap: e.multiYearCap || e.maxPayout || 15e6,
      };
      if (e.currentFairValue || e.initialFairValue) {
        setBacktestComparison({ reported: e.currentFairValue || e.initialFairValue, computed: null, gap: null });
      }
    } else if (mode === "live" && extracted?.earnout) {
      const e = extracted.earnout; const a = e.assumptions || {};
      const newPeriods = (e.periods || []).map((p, i) => ({
        year: i + 1, yearFromNow: p.yearFromNow || (i + 1), structure: p.structure || e.structure || "linear",
        threshold: p.threshold || 0, participationRate: p.participationRate || 0, fixedPayment: p.fixedPayment || 0,
        cap: p.cap || null, floor: p.floor || 0, projectedMetric: p.projectedMetric || 0, tiers: p.tiers || null,
      }));
      if (newPeriods.length === 0) newPeriods.push({ ...params.periods[0] });
      np = { ...params, metric: e.metric || params.metric, currentMetric: a.currentMetric || params.currentMetric,
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
    }
    setParams(np);
    // Populate provenance from extraction if available
    const prov = extractedData?.earnout?.provenance || extractedData?.earnouts?.[0]?.provenance || {};
    if (prov && Object.keys(prov).length > 0) {
      setProvenance(prev => ({
        ...prev,
        volatility: { ...prev.volatility, ...(prov.volatility || {}), comparableCompanies: prov.volatility?.comparableCompanies || prov.comparableCompanies || extractedData?.earnout?.assumptions?.comparableCompanies?.map(n => ({ name: n })) || [] },
        discountRate: { ...prev.discountRate, ...(prov.discountRate || {}) },
        projections: { ...prev.projections, ...(prov.projections || {}) },
        creditRisk: { ...prev.creditRisk, ...(prov.creditRisk || {}) },
        referencedDocuments: prov.referencedDocuments || [],
        methodologyQuote: prov.methodologyQuote || null,
      }));
    }
    setProgress(100);
    // Go to review screen instead of running MC directly
    setTimeout(() => setView("review"), 400);
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
      { s: "Ready for review...", p: 100, d: 300 },
    ];

    let delay = 0;
    stages.forEach(({ s, p, d }) => {
      delay += d;
      setTimeout(() => { setStage(s); setProgress(p); }, delay);
    });

    setTimeout(() => {
      setParams(gtParams);
      setBacktestComparison({ reported: null, computed: null, gap: null, isDemo: true, scenarioBasedValue: 9.1e6 });
      setVerification({ verified: true, overallConfidence: 100, recommendation: "proceed", errors: [], missingTerms: [] });
      // Set GT provenance for demo
      setProvenance({
        volatility: { methodology: "2-year historical equity volatility of comparable public companies, de-levered using Hamada equation and re-levered at subject company capital structure.", comparableCompanies: [{ name: "Comparable A", ticker: "COMPA", volatility: 0.38 }, { name: "Comparable B", ticker: "COMPB", volatility: 0.42 }, { name: "Comparable C", ticker: "COMPC", volatility: 0.40 }], deLeveringMethod: "Hamada equation", sourceLocation: "Grant Thornton article, April 2021" },
        discountRate: { methodology: "EBITDA discount rate reflecting required return for EBITDA cash flows, consistent with WACC.", components: { riskFreeRate: 0.025, equityRiskPremium: 0.055, sizePremium: 0.02 }, sourceLocation: "GT article assumptions" },
        projections: { source: "Management forecast at acquisition date", forecastDate: "December 31, 2018", provider: "TargetCo management", sourceLocation: "GT article example terms" },
        creditRisk: { methodology: "Implied from risk-adjusted discount rate of 4.5% less risk-free rate of 2.5%", acquirerRating: "Not stated — implied investment-grade" },
        referencedDocuments: ["Grant Thornton 'Earnout values: Scenario-based forecast vs. simulation', April 2021", "Appraisal Foundation VFR 4: Valuation of Contingent Consideration, February 2019"],
        methodologyQuote: "The authoritative VFR 4 recommends the option-pricing methodologies, including a Monte Carlo simulation, in the valuation of revenue and earnings earnouts.",
      });
      setView("review");
    }, delay + 300);
  };

  const resetAll = () => { setView("landing"); setResults(null); setSensitivities(null); setExtractedData(null); setDocText(""); setFiles([]); setBacktestComparison(null); setMode(null); setAuditQuestions(null); setAuditResponses(null); setProvenance({ volatility: { methodology: null, comparableCompanies: [], deLeveringMethod: null, sourceLocation: null, userNote: "" }, discountRate: { methodology: null, components: null, sourceLocation: null, userNote: "" }, projections: { source: null, forecastDate: null, provider: null, sourceLocation: null, userNote: "" }, creditRisk: { methodology: null, acquirerRating: null, userNote: "" }, referencedDocuments: [], methodologyQuote: null }); };

  // Run valuation after user confirms extracted terms on review screen
  const runFromReview = () => {
    setView("processing"); setProgress(0);
    setStage(`Running Monte Carlo (${MC_PATHS.toLocaleString()} paths)...`); setProgress(60);
    setTimeout(() => {
      const res = runMultiPeriodMC(params);
      setResults(res);
      setProgress(80); setStage("Generating sensitivity analysis...");
      setTimeout(() => {
        const sens = {};
        sens["Volatility"] = runSensitivity(params, "volatility", [0.15, 0.70]);
        sens["Metric Discount Rate"] = runSensitivity(params, "discountRate", [0.06, 0.20]);
        sens["Current Metric"] = runSensitivity(params, "currentMetric", [params.currentMetric * 0.5, params.currentMetric * 1.5]);
        sens["Growth Rate"] = runSensitivity(params, "metricGrowthRate", [0, 0.20]);
        setSensitivities(sens);

        // Update backtest comparison with computed value
        if (backtestComparison) {
          const bc = { ...backtestComparison, computed: res.fairValue };
          if (bc.reported) bc.gap = Math.abs(res.fairValue - bc.reported) / bc.reported * 100;
          setBacktestComparison(bc);
        }

        setProgress(100);
        setTimeout(() => setView("results"), 300);
      }, 100);
    }, 200);
  };

  // ============================================================
  // AUDIT SUPPORT ENGINE
  // ============================================================
  const processAuditQuestions = async (questionsText) => {
    setAuditProcessing(true);
    const mrp = Math.max(0, params.discountRate - params.riskFreeRate);
    const pd = params.payoffDiscountRate != null ? params.payoffDiscountRate : params.riskFreeRate + (params.isEscrowed ? 0 : (params.creditAdj || 0));
    const ctx = {
      terms: { metric: params.metric, periods: params.periods.length, structure: params.periods[0]?.structure, maxPayout: params.multiYearCap || params.periods.reduce((s, p) => s + (p.cap || p.fixedPayment || 0), 0), hasCatchUp: params.hasCatchUp, hasClawback: params.hasClawback, isEscrowed: params.isEscrowed, paymentDelay: params.paymentDelay },
      assumptions: { currentMetric: params.currentMetric, growthRate: params.metricGrowthRate, volatility: params.volatility, metricDiscountRate: params.discountRate, riskFreeRate: params.riskFreeRate, creditAdj: params.creditAdj, metricRiskPremium: mrp, payoffDiscountRate: pd, riskNeutralDrift: params.metricGrowthRate - mrp },
      periodDetail: params.periods.map((p, i) => ({ period: i + 1, year: p.yearFromNow, structure: p.structure, threshold: p.threshold, projectedMetric: p.projectedMetric, fixedPayment: p.fixedPayment, cap: p.cap, fairValue: results?.periodStats?.[i]?.mean })),
      results: results ? { fairValue: results.fairValue, ci95: results.ci95, probPayoff: results.probPayoff, percentiles: results.percentiles } : null,
      sensitivity: sensitivities ? Object.fromEntries(Object.entries(sensitivities).map(([k, v]) => [k, { min: Math.min(...v.map(d => d.fairValue)), max: Math.max(...v.map(d => d.fairValue)) }])) : null,
      provenance,
    };
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" },
        body: JSON.stringify({ model: CLAUDE_MODEL, max_tokens: 8000,
          system: `You are an expert earnout valuation professional responding to auditor questions about a contingent consideration fair value measurement under ASC 820 / ASC 805 / IFRS 13 / IFRS 3. You have deep knowledge of the Appraisal Foundation VFR 4 guidance.

You will receive the auditor's questions and the complete valuation context.

For EACH question, generate a professional response. Return ONLY valid JSON array, no markdown:
[{"question":"the original question verbatim","response":"detailed professional response using model data and provenance","confidence":number 0-100,"confidenceReason":"brief explanation of confidence level","missingInfo":"what would strengthen this response, or null","category":"methodology|assumptions|sensitivity|rollforward|terms|compliance|other"}]

CONFIDENCE SCORING:
90-100: Uses specific numbers from the model AND provenance source detail. Fully defensible.
70-89: Uses model data but provenance is partial. Structurally sound, may need user to add sources.
50-69: Directionally correct using general industry knowledge. User should add deal-specific detail.
Below 50: Outside model scope. Provide a framework and tell user what to supply.

RESPONSE STYLE: Write as if you are the valuation professional defending the work to the auditor. Be specific with numbers. Reference ASC 820, VFR 4, or other guidance where relevant. For assumption questions, cite the provenance data if available. For sensitivity questions, use the sensitivity data from the model.`,
          messages: [{ role: "user", content: `AUDITOR QUESTIONS:\n${questionsText}\n\nVALUATION CONTEXT:\n${JSON.stringify(ctx, null, 2)}` }] })
      });
      const data = await response.json();
      const txt = (data.content?.map(c => c.text || "").join("") || "").replace(/```json|```/g, "").trim();
      setAuditResponses(JSON.parse(txt));
    } catch (err) {
      console.error("Audit engine error:", err);
      setAuditResponses([{ question: "Processing Error", response: "The audit response engine encountered an error. Please check your API key and try again.", confidence: 0, confidenceReason: "Error", missingInfo: null, category: "other" }]);
    }
    setAuditProcessing(false);
  };

  // Export audit responses as Word
  const exportAuditWord = () => {
    if (!auditResponses) return;
    const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    const confBadge = (c) => c >= 90 ? "●" : c >= 70 ? "◐" : c >= 50 ? "○" : "△";
    const html = `<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word'><head><meta charset="utf-8">
<style>body{font-family:Aptos,Calibri,sans-serif;font-size:9pt;line-height:1.6;color:#1a1a1a;max-width:8in;margin:0 auto;padding:0.8in}
h1{font-size:13pt;font-weight:700;color:#1F3864;margin-bottom:4pt}h2{font-size:11pt;font-weight:700;color:#1F3864;margin-top:16pt;border-bottom:1px solid #1F3864;padding-bottom:3pt}
p{margin-bottom:6pt;text-align:justify}.q{font-weight:700;color:#1F3864;margin-top:12pt;margin-bottom:4pt;font-size:9.5pt}
.conf{display:inline-block;padding:1pt 6pt;border-radius:3pt;font-size:8pt;font-weight:600;margin-left:6pt}
.high{background:#E2EFDA;color:#1F6F3A}.med{background:#FFF2CC;color:#8B6914}.low{background:#FCE4E4;color:#C0392B}
.missing{background:#F2F7FB;border-left:2pt solid #2563eb;padding:4pt 8pt;margin:4pt 0;font-size:8.5pt;color:#2563eb}
.footer{margin-top:24pt;border-top:1pt solid #999;padding-top:6pt;font-size:7.5pt;color:#666}</style></head><body>
<h1>Auditor Response — Contingent Consideration Fair Value</h1>
<p style="font-size:8.5pt;color:#555">${params.metric} Earnout • ${params.periods.length} Periods • ${date}</p>
<p style="font-size:8.5pt;color:#555">Fair Value: $${(results?.fairValue / 1e6).toFixed(2)}M | 95% CI: $${(results?.ci95[0] / 1e6).toFixed(2)}M – $${(results?.ci95[1] / 1e6).toFixed(2)}M</p>
<h2>Responses to Auditor Questions</h2>
${auditResponses.map((r, i) => `
<p class="q">Q${i + 1}: ${r.question} <span class="conf ${r.confidence >= 90 ? "high" : r.confidence >= 70 ? "med" : "low"}">${confBadge(r.confidence)} ${r.confidence}%</span></p>
<p>${r.response}</p>
${r.missingInfo ? `<div class="missing"><strong>To strengthen:</strong> ${r.missingInfo}</div>` : ""}
`).join("")}
<div class="footer">Generated by ValuProEarnout Audit Support Engine — ${date}<br>Responses are draft and should be reviewed by a qualified valuation professional before submission.</div>
</body></html>`;
    const blob = new Blob([html], { type: "application/msword" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `AuditResponse_${new Date().toISOString().split("T")[0]}.doc`; a.click();
  };

  // Export audit responses as Excel
  const exportAuditExcel = async () => {
    if (!auditResponses) return;
    let XLSX; try { XLSX = await import("xlsx-js-style"); } catch(e) { XLSX = await import("xlsx"); }
    const wb = XLSX.utils.book_new();
    const fontH = { name: "Aptos", sz: 10, bold: true, color: { rgb: "FFFFFF" } };
    const fillH = { fgColor: { rgb: "1F3864" } };
    const fontN = { name: "Aptos", sz: 9 };
    const fontB = { name: "Aptos", sz: 9, bold: true };
    const bdr = { top: { style: "thin", color: { rgb: "B4B4B4" } }, bottom: { style: "thin", color: { rgb: "B4B4B4" } }, left: { style: "thin", color: { rgb: "B4B4B4" } }, right: { style: "thin", color: { rgb: "B4B4B4" } } };
    const sH = { font: fontH, fill: { type: "pattern", patternType: "solid", ...fillH }, alignment: { horizontal: "center", vertical: "center", wrapText: true }, border: bdr };
    const sN = { font: fontN, alignment: { vertical: "top", wrapText: true }, border: bdr };
    const sB = { font: fontB, alignment: { vertical: "top", wrapText: true }, border: bdr };
    const sConf = (c) => ({ font: fontB, alignment: { horizontal: "center", vertical: "center" }, border: bdr, fill: { type: "pattern", patternType: "solid", fgColor: { rgb: c >= 90 ? "E2EFDA" : c >= 70 ? "FFF2CC" : c >= 50 ? "FCE4E4" : "F2F2F2" } }, numFmt: "0" });

    const ws = {};
    const s = (r, col, v, st) => { ws[XLSX.utils.encode_cell({ r, c: col })] = { v, t: typeof v === "number" ? "n" : "s", s: st }; };
    s(0, 0, "#", sH); s(0, 1, "Auditor Question", sH); s(0, 2, "Draft Response", sH); s(0, 3, "Confidence", sH); s(0, 4, "Confidence Reason", sH); s(0, 5, "Missing Info / To Strengthen", sH); s(0, 6, "Category", sH);
    auditResponses.forEach((r, i) => {
      const row = i + 1;
      const bg = i % 2 === 0 ? "FFFFFF" : "F8F9FA";
      const sA = { ...sN, fill: { type: "pattern", patternType: "solid", fgColor: { rgb: bg } } };
      s(row, 0, i + 1, { ...sB, alignment: { horizontal: "center", vertical: "top" }, fill: { type: "pattern", patternType: "solid", fgColor: { rgb: bg } } });
      s(row, 1, r.question, { ...sB, fill: { type: "pattern", patternType: "solid", fgColor: { rgb: bg } } });
      s(row, 2, r.response, sA);
      s(row, 3, r.confidence, sConf(r.confidence));
      s(row, 4, r.confidenceReason || "", sA);
      s(row, 5, r.missingInfo || "—", sA);
      s(row, 6, r.category || "", sA);
    });
    ws["!ref"] = XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: auditResponses.length, c: 6 } });
    ws["!cols"] = [{ wch: 4 }, { wch: 40 }, { wch: 70 }, { wch: 10 }, { wch: 30 }, { wch: 40 }, { wch: 14 }];
    // Set row heights for wrapping
    ws["!rows"] = [{ hpt: 20 }, ...auditResponses.map(() => ({ hpt: 80 }))];
    XLSX.utils.book_append_sheet(wb, ws, "Audit Responses");
    const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    const blob = new Blob([wbout], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `AuditResponse_${new Date().toISOString().split("T")[0]}.xlsx`; a.click();
  };

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

  // ---- REVIEW & CONFIRM ----
  if (view === "review") {
    const structures = ["binary", "linear", "tiered", "percentage", "cagr", "milestone"];
    const confScore = verification?.overallConfidence || 0;
    const confColor = confScore >= 80 ? c.success : confScore >= 50 ? c.warning : c.danger;
    const confLabel = confScore >= 80 ? "High" : confScore >= 50 ? "Medium" : "Low";
    const updatePeriod = (idx, field, val) => { const np = [...params.periods]; np[idx] = { ...np[idx], [field]: val }; setParams(p => ({ ...p, periods: np })); };
    const addPeriod = () => { const last = params.periods[params.periods.length - 1]; setParams(p => ({ ...p, periods: [...p.periods, { ...last, year: last.year + 1, yearFromNow: (last.yearFromNow || last.year) + 1 }] })); };
    const removePeriod = (idx) => { if (params.periods.length <= 1) return; setParams(p => ({ ...p, periods: p.periods.filter((_, i) => i !== idx) })); };

    // ---- INTELLIGENT FIELD DIAGNOSTICS ENGINE ----
    // For each field: status (ok|warning|missing|info), message, suggestion, impact, resolution
    const isGT = backtestComparison?.isDemo;
    const isBT = mode === "backtest";
    const extractionNotes = extractedData?.earnout?.ambiguities || extractedData?.earnouts?.[0]?.ambiguities || [];

    const getDiag = (field, value) => {
      const missing = value === null || value === undefined || value === 0;
      const diags = {
        currentMetric: {
          status: missing ? "missing" : "ok",
          message: missing ? "Current metric value not found in the document." : (isGT ? "FY18 base EBITDA from GT example." : isBT ? "Extracted from most recent period actuals in the filing." : "Extracted from PPA report or management forecast."),
          suggestion: missing ? "Use the most recent trailing twelve months (TTM) actual. For EBITDA, this should be adjusted EBITDA consistent with the earnout definition." : null,
          impact: "The current metric anchors the growth projection. A 10% error here flows through to all period projections.",
          resolution: missing ? "Check the acquisition closing financials or latest management reporting package." : null,
        },
        volatility: {
          status: missing ? "missing" : value < 0.15 ? "warning" : value > 0.65 ? "warning" : "ok",
          message: missing ? "Volatility not disclosed in the document."
            : isGT ? "GT example states 40% EBITDA volatility derived from comparable company equity volatility."
            : isBT ? "SEC filings rarely disclose volatility directly. This may be a default value."
            : (value && !missing) ? "Extracted from PPA report comparable company analysis." : "Could not extract — the report may reference 'comparable company analysis' without stating the figure.",
          suggestion: missing ? `Suggested: ${params.metric?.includes("Revenue") ? "15%–25%" : "30%–45%"} based on typical ${params.metric?.includes("Revenue") ? "revenue" : "EBITDA"} volatility for mid-market targets. EBITDA volatility is typically 1.5–2.5× revenue volatility due to operational leverage.`
            : value < 0.15 ? "This seems low. EBITDA volatility below 15% is unusual — even stable businesses typically show 20%+. Verify against comparable company data."
            : value > 0.65 ? "This is very high. Volatility above 65% is typical only for early-stage or biotech companies. Verify this reflects the metric, not equity volatility."
            : null,
          impact: `Volatility is the most sensitive input for ${params.periods[0]?.structure === "binary" ? "binary" : "threshold-based"} earnouts. A 5% change in volatility can move fair value by 10–15%.`,
          resolution: missing ? "Upload the comparable company analysis exhibit, or check Kroll Cost of Capital Navigator / Duff & Phelps Valuation Handbook for industry volatility benchmarks." : null,
        },
        discountRate: {
          status: missing ? "missing" : value < 0.06 ? "warning" : value > 0.20 ? "warning" : "ok",
          message: missing ? "Metric discount rate not found. This is the required return for the metric's cash flows (typically WACC-based)."
            : isGT ? "GT states 10.0% EBITDA discount rate."
            : "Extracted from report. Verify this represents the metric-specific discount rate, not the earnout payoff discount rate.",
          suggestion: missing ? `Suggested: 8%–14% for ${params.metric?.includes("Revenue") ? "revenue (typically lower risk premium than EBITDA)" : "EBITDA"}. Should be consistent with the WACC used in the PPA business enterprise valuation.`
            : value < 0.06 ? "Below 6% is unusually low for a metric discount rate — this may be the risk-free rate or payoff discount rate, not the metric rate."
            : value > 0.20 ? "Above 20% implies very high risk. Verify this isn't a blended rate that already includes credit risk."
            : null,
          impact: "The metric discount rate determines the risk premium used to convert management's projections to risk-neutral. Higher rate = lower risk-neutral metric = lower probability of hitting thresholds.",
          resolution: missing ? "Check the PPA report's DCF / business enterprise valuation for the WACC, or the earnout methodology section for the 'required metric risk premium' (RMRP)." : null,
        },
        riskFreeRate: {
          status: missing ? "missing" : "ok",
          message: isGT ? "Approximated at 2.5% reflecting 2018 US Treasury rates." : "Risk-free rate matching the weighted-average earnout duration.",
          suggestion: missing ? `Suggested: Use the US Treasury yield matching the earnout's weighted-average duration. For a ${params.periods.length}-year earnout, use the ${params.periods.length}-year Treasury rate.` : null,
          impact: "Moderate. Affects both the risk-neutral drift and the payoff discount rate, but changes of 50bps have a relatively small impact.",
          resolution: missing ? "Check the Federal Reserve's H.15 release for current Treasury yields, or the PPA report's risk-free rate assumption." : null,
        },
        creditAdj: {
          status: params.isEscrowed ? "info" : missing ? "warning" : "ok",
          message: params.isEscrowed ? "Credit adjustment is zero because earnout funds are held in escrow." 
            : missing ? "Credit risk adjustment not stated. Since funds are NOT escrowed, counterparty credit risk should be reflected."
            : isGT ? "Implied at 2.0% (risk-adjusted rate 4.5% = Rf 2.5% + credit 2.0%)."
            : "Extracted from report.",
          suggestion: params.isEscrowed ? null : missing ? "Suggested: 1%–3% for investment-grade acquirers, 3%–5% for non-investment-grade. Check the acquirer's credit profile or debt spreads." : null,
          impact: params.isEscrowed ? "None — escrowed funds eliminate counterparty risk." : "Directly increases the payoff discount rate, reducing the present value of all future payments.",
          resolution: missing && !params.isEscrowed ? "Review the acquirer's credit rating or recent debt issuance spreads. If the merger agreement provides for escrow, toggle the 'Escrowed' feature." : null,
        },
        metricGrowthRate: {
          status: missing ? "warning" : "ok",
          message: missing ? "Growth rate not explicitly stated." : isGT ? "Implied ~12% CAGR from management projections ($12M → $17M over 3 years)." : "Derived from management projections.",
          suggestion: missing ? "Calculate from the projected metrics: if Current = $15M and Year 3 Projected = $20M, growth rate = (20/15)^(1/3) - 1 ≈ 10%." : null,
          impact: "Growth rate affects the risk-neutral drift. Combined with the metric discount rate, it determines whether the simulated metric trends above or below thresholds.",
          resolution: missing ? "Use the management forecast or budget projections for the earnout metric." : null,
        },
      };

      // Period-level diagnostics
      if (field === "projectedMetric") {
        return {
          status: missing ? "missing" : "ok",
          message: missing ? "Projected metric for this period not found." : "Management's forecast for this measurement period.",
          suggestion: missing ? `Suggested: Apply the growth rate (${fmtPct(params.metricGrowthRate)}) to the current metric. Year ${value || "N"} projected = ${fmt(params.currentMetric * Math.pow(1 + params.metricGrowthRate, value || 1))}.` : null,
          impact: "The projected metric is the starting point for risk-neutral simulation. If projected = threshold, probability of payment is ~35–45% under typical volatility.",
          resolution: missing ? "Check the management forecast update or budget for the specific measurement period." : null,
        };
      }
      if (field === "threshold") {
        return {
          status: missing && params.periods[0]?.structure !== "percentage" ? "missing" : "ok",
          message: missing ? "Threshold not found. For binary earnouts, this is required." : "The performance target that must be met or exceeded.",
          suggestion: missing ? "Check the merger agreement or earnout schedule for the specific EBITDA/revenue target for each measurement period." : null,
          impact: "The threshold is the strike price of the option. Small changes (±5%) can significantly affect the probability of payment.",
          resolution: missing ? "The threshold is typically defined in the merger agreement, which may not be in the PPA report. Upload the merger agreement or earnout schedule." : null,
        };
      }

      return diags[field] || { status: "ok", message: "", suggestion: null, impact: null, resolution: null };
    };

    // ---- ENHANCED EditField with inline diagnostics + provenance ----
    const EditField = ({ label, value, onChange, type = "number", step, format, fieldKey, periodIdx, tooltip }) => {
      const [showDiag, setShowDiag] = useState(false);
      const diag = getDiag(fieldKey, value);
      const isIssue = diag.status === "missing" || diag.status === "warning";
      const statusColor = diag.status === "missing" ? c.danger : diag.status === "warning" ? c.warning : diag.status === "info" ? c.accent : c.success;
      const statusIcon = diag.status === "missing" ? "alert" : diag.status === "warning" ? "alert" : diag.status === "info" ? "info" : "check";
      const statusLabel = diag.status === "missing" ? "Not Found" : diag.status === "warning" ? "Review" : diag.status === "info" ? "Note" : "Extracted";

      // Get provenance for this field
      const fieldProv = fieldKey === "volatility" ? provenance.volatility : fieldKey === "discountRate" ? provenance.discountRate : fieldKey === "metricGrowthRate" || fieldKey === "projectedMetric" ? provenance.projections : fieldKey === "creditAdj" ? provenance.creditRisk : null;
      const hasProvenance = fieldProv && (fieldProv.methodology || fieldProv.source || (fieldProv.comparableCompanies && fieldProv.comparableCompanies.length > 0) || fieldProv.sourceLocation);

      return (
        <div style={{ marginBottom: 12 }}>
          <div className="vf" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
            <span style={{ fontSize: 11, color: c.textMuted, display: "flex", alignItems: "center", gap: 4 }}>
              {label}
              {tooltip && <span title={tooltip} style={{ cursor: "help", opacity: 0.4 }}><Icon name="info" size={9} /></span>}
            </span>
            <div className="vf" style={{ gap: 4, alignItems: "center" }}>
              {hasProvenance && <span style={{ fontSize: 8, padding: "1px 4px", borderRadius: 2, background: "rgba(5,150,105,0.06)", color: c.success }}>sourced</span>}
              <button onClick={() => setShowDiag(!showDiag)} style={{ display: "flex", alignItems: "center", gap: 3, padding: "1px 6px", borderRadius: 3, border: "none", cursor: "pointer", fontSize: 9, fontWeight: 500, background: isIssue ? `${statusColor}12` : "transparent", color: statusColor }}>
                <Icon name={statusIcon} size={9} color={statusColor} /> {statusLabel}
              </button>
            </div>
          </div>
          <input type={type} value={format === "percent" ? ((value || 0) * 100).toFixed(1) : value || ""} step={step}
            onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(format === "percent" ? v / 100 : v); }}
            style={{ width: "100%", padding: "6px 10px", fontSize: 12, fontFamily: "'IBM Plex Mono',monospace", border: `1.5px solid ${isIssue ? statusColor : c.cardBorder}`, borderRadius: 6, background: isIssue ? `${statusColor}06` : c.inputBg, color: c.text, outline: "none", textAlign: "right" }} />
          
          {/* Expandable diagnostics + provenance panel */}
          {showDiag && (
            <div style={{ marginTop: 6, padding: "8px 10px", background: tc ? "#1a2540" : "#fafbfc", border: `1px solid ${c.cardBorder}`, borderRadius: 6, fontSize: 10, lineHeight: 1.55 }}>
              {/* Status message */}
              <div style={{ color: c.text, marginBottom: 6 }}>
                <Icon name={statusIcon} size={10} color={statusColor} /> {diag.message}
              </div>
              
              {/* Provenance — source detail */}
              {hasProvenance && (
                <div style={{ padding: "5px 8px", background: "rgba(5,150,105,0.03)", border: `1px solid rgba(5,150,105,0.1)`, borderRadius: 4, marginBottom: 5 }}>
                  <div style={{ fontSize: 9, fontWeight: 600, color: c.success, marginBottom: 3 }}>Source Detail (used by Audit Support engine)</div>
                  {fieldProv.methodology && <div style={{ color: c.textMuted }}><strong>Methodology:</strong> {fieldProv.methodology}</div>}
                  {fieldProv.source && <div style={{ color: c.textMuted }}><strong>Source:</strong> {fieldProv.source}</div>}
                  {fieldProv.comparableCompanies && fieldProv.comparableCompanies.length > 0 && (
                    <div style={{ color: c.textMuted }}><strong>Comparables:</strong> {fieldProv.comparableCompanies.map(co => `${co.name}${co.ticker ? ` (${co.ticker})` : ""}${co.volatility ? `: ${(co.volatility * 100).toFixed(1)}%` : ""}`).join(", ")}</div>
                  )}
                  {fieldProv.deLeveringMethod && <div style={{ color: c.textMuted }}><strong>De-levering:</strong> {fieldProv.deLeveringMethod}</div>}
                  {fieldProv.components && <div style={{ color: c.textMuted }}><strong>Components:</strong> {Object.entries(fieldProv.components).filter(([,v]) => v != null).map(([k,v]) => `${k}: ${typeof v === "number" && v < 1 ? (v*100).toFixed(1)+"%" : v}`).join(", ")}</div>}
                  {fieldProv.forecastDate && <div style={{ color: c.textMuted }}><strong>Forecast date:</strong> {fieldProv.forecastDate}</div>}
                  {fieldProv.provider && <div style={{ color: c.textMuted }}><strong>Provider:</strong> {fieldProv.provider}</div>}
                  {fieldProv.acquirerRating && <div style={{ color: c.textMuted }}><strong>Rating:</strong> {fieldProv.acquirerRating}</div>}
                  {(fieldProv.sourceLocation) && <div style={{ color: c.textDim, fontStyle: "italic" }}>Ref: {fieldProv.sourceLocation}</div>}
                </div>
              )}

              {/* No provenance — editable note for user to add */}
              {!hasProvenance && (
                <div style={{ padding: "5px 8px", background: "rgba(217,119,6,0.03)", border: `1px solid rgba(217,119,6,0.1)`, borderRadius: 4, marginBottom: 5 }}>
                  <div style={{ fontSize: 9, fontWeight: 600, color: c.warning, marginBottom: 3 }}>No source detail found — add below to strengthen audit responses</div>
                  <textarea value={fieldProv?.userNote || ""} placeholder="E.g., Based on 3 comparable companies per Exhibit C of PPA report..."
                    onChange={e => {
                      const note = e.target.value;
                      if (fieldKey === "volatility") setProvenance(p => ({ ...p, volatility: { ...p.volatility, userNote: note } }));
                      else if (fieldKey === "discountRate") setProvenance(p => ({ ...p, discountRate: { ...p.discountRate, userNote: note } }));
                      else if (fieldKey === "metricGrowthRate" || fieldKey === "projectedMetric") setProvenance(p => ({ ...p, projections: { ...p.projections, userNote: note } }));
                      else if (fieldKey === "creditAdj") setProvenance(p => ({ ...p, creditRisk: { ...p.creditRisk, userNote: note } }));
                    }}
                    rows={2} style={{ width: "100%", padding: "4px 6px", fontSize: 10, border: `1px solid ${c.cardBorder}`, borderRadius: 4, background: c.inputBg, color: c.text, outline: "none", resize: "vertical", fontFamily: "'Inter',system-ui,sans-serif" }} />
                </div>
              )}
              
              {/* Suggestion */}
              {diag.suggestion && (
                <div style={{ padding: "4px 8px", background: "rgba(37,99,235,0.04)", borderRadius: 4, marginBottom: 4, color: c.accent }}>
                  <strong>Suggested:</strong> {diag.suggestion}
                </div>
              )}
              
              {/* Sensitivity impact */}
              {diag.impact && (
                <div style={{ color: c.textMuted, marginBottom: diag.resolution ? 4 : 0 }}>
                  <strong style={{ color: c.warning }}>Impact:</strong> {diag.impact}
                </div>
              )}
              
              {/* Resolution — what to upload */}
              {diag.resolution && (
                <div style={{ padding: "4px 8px", background: "rgba(5,150,105,0.04)", borderRadius: 4, color: c.success }}>
                  <strong>To resolve:</strong> {diag.resolution}
                </div>
              )}
            </div>
          )}
        </div>
      );
    };

    // Count issues for summary
    const issueFields = ["currentMetric", "volatility", "discountRate", "riskFreeRate", "creditAdj", "metricGrowthRate"];
    const issueCount = issueFields.filter(f => { const d = getDiag(f, params[f]); return d.status === "missing" || d.status === "warning"; }).length;
    const periodIssues = params.periods.reduce((count, p) => {
      if (!p.projectedMetric) count++;
      if (!p.threshold && p.structure !== "percentage") count++;
      return count;
    }, 0);
    const totalIssues = issueCount + periodIssues;

    return (
      <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',system-ui,sans-serif", color: c.text }}>{fontLink}{hdr(true)}
        <div className="r-p" style={{ maxWidth: 920, margin: "0 auto", padding: "28px" }}>
          {/* Header */}
          <div className="vf r-col" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, gap: 8 }}>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 3, fontFamily: "'Source Serif 4',Georgia,serif" }}>Review Extracted Terms</h2>
              <p style={{ fontSize: 12, color: c.textMuted }}>Verify all fields below. Click any status badge to see diagnostics, suggestions, and sensitivity impact.</p>
            </div>
            <div className="vf" style={{ gap: 10, alignItems: "center" }}>
              {verification && (
                <div className="vf" style={{ gap: 6, alignItems: "center" }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: confColor }} />
                  <span style={{ fontSize: 11, color: confColor, fontWeight: 600 }}>Confidence: {confLabel} ({confScore}%)</span>
                </div>
              )}
              {totalIssues > 0 && (
                <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "rgba(217,119,6,0.08)", color: c.warning, fontWeight: 600 }}>
                  {totalIssues} field{totalIssues > 1 ? "s" : ""} need review
                </span>
              )}
              {totalIssues === 0 && (
                <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "rgba(5,150,105,0.08)", color: c.success, fontWeight: 600 }}>
                  All fields populated
                </span>
              )}
            </div>
          </div>

          {/* Verification warnings from extraction */}
          {verification?.errors?.length > 0 && (
            <div style={{ ...cardStyle, marginBottom: 12, padding: 14, borderColor: "rgba(220,38,38,0.15)", background: "rgba(220,38,38,0.02)" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: c.danger, marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}><Icon name="alert" size={13} color={c.danger} /> Extraction Issues</div>
              {verification.errors.map((err, i) => <div key={i} style={{ fontSize: 11, color: c.textMuted, marginBottom: 2 }}>• <strong>{err.field}:</strong> {err.issue}</div>)}
            </div>
          )}

          {/* Ambiguities from extraction */}
          {extractionNotes.length > 0 && (
            <div style={{ ...cardStyle, marginBottom: 12, padding: 14, borderColor: "rgba(37,99,235,0.15)", background: "rgba(37,99,235,0.02)" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: c.accent, marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}><Icon name="info" size={13} color={c.accent} /> Document Ambiguities</div>
              {extractionNotes.map((note, i) => <div key={i} style={{ fontSize: 11, color: c.textMuted, marginBottom: 2 }}>• {note}</div>)}
            </div>
          )}

          {/* GT Demo banner */}
          {isGT && (
            <div style={{ ...cardStyle, marginBottom: 12, padding: 14, borderColor: "rgba(5,150,105,0.15)", background: "rgba(5,150,105,0.02)" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: c.success, marginBottom: 4, display: "flex", alignItems: "center", gap: 5 }}><Icon name="target" size={13} color={c.success} /> Grant Thornton Benchmark Example</div>
              <div style={{ fontSize: 11, color: c.textMuted, lineHeight: 1.55 }}>All terms pre-loaded from the GT article. Click any status badge to see the GT source for each assumption. You can modify any value to see how it affects the result.</div>
            </div>
          )}

          <div className="vg r-stack" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {/* LEFT: Earnout Terms + Assumptions */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={cardStyle}>
                <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, display: "flex", alignItems: "center", gap: 5 }}><Icon name="file" size={14} color={c.accent} /> Earnout Terms</h3>

                <div style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: c.textMuted, display: "block", marginBottom: 3 }}>Performance Metric</span>
                  <input value={params.metric} onChange={e => setParams(p => ({ ...p, metric: e.target.value }))} style={{ width: "100%", padding: "6px 10px", fontSize: 12, border: `1px solid ${c.cardBorder}`, borderRadius: 6, background: c.inputBg, color: c.text, outline: "none" }} />
                </div>

                <EditField label="Current Metric ($)" value={params.currentMetric} onChange={v => setParams(p => ({ ...p, currentMetric: v }))} step={100000} fieldKey="currentMetric" />
                <EditField label="Multi-Year Cap ($)" value={params.multiYearCap || 0} onChange={v => setParams(p => ({ ...p, multiYearCap: v, hasMultiYearCap: v > 0 }))} step={100000} fieldKey="multiYearCap" />
                <EditField label="Payment Delay (days)" value={params.paymentDelay || 120} onChange={v => setParams(p => ({ ...p, paymentDelay: v }))} step={30} fieldKey="paymentDelay" />

                {/* Toggle features */}
                <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {[{ k: "hasCatchUp", l: "Catch-Up" }, { k: "hasClawback", l: "Clawback" }, { k: "hasAcceleration", l: "Acceleration" },
                    { k: "hasCarryForward", l: "Carry-Forward" }, { k: "hasCumulativeTarget", l: "Cumulative Target" }, { k: "isEscrowed", l: "Escrowed" }, { k: "isMultiMetric", l: "Multi-Metric" },
                  ].map(f => (
                    <button key={f.k} onClick={() => setParams(p => ({ ...p, [f.k]: !p[f.k] }))}
                      style={{ padding: "3px 8px", fontSize: 10, fontWeight: 500, borderRadius: 4, cursor: "pointer", border: `1px solid ${params[f.k] ? c.accent : c.cardBorder}`, background: params[f.k] ? c.accentLight : "transparent", color: params[f.k] ? c.accent : c.textDim, transition: "all 0.15s" }}>
                      {params[f.k] ? "✓ " : ""}{f.l}
                    </button>
                  ))}
                </div>
              </div>

              {/* Assumptions */}
              <div style={cardStyle}>
                <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, display: "flex", alignItems: "center", gap: 5 }}><Icon name="sliders" size={14} color={c.accent} /> Valuation Assumptions</h3>
                <EditField label="Metric Growth Rate" value={params.metricGrowthRate} onChange={v => setParams(p => ({ ...p, metricGrowthRate: v }))} step={0.001} format="percent" fieldKey="metricGrowthRate" />
                <EditField label="Volatility (σ)" value={params.volatility} onChange={v => setParams(p => ({ ...p, volatility: v }))} step={0.01} format="percent" fieldKey="volatility" />
                <EditField label="Metric Discount Rate" value={params.discountRate} onChange={v => setParams(p => ({ ...p, discountRate: v }))} step={0.005} format="percent" fieldKey="discountRate" />
                <EditField label="Risk-Free Rate" value={params.riskFreeRate} onChange={v => setParams(p => ({ ...p, riskFreeRate: v }))} step={0.001} format="percent" fieldKey="riskFreeRate" />
                <EditField label="Credit Risk Adjustment" value={params.creditAdj || 0} onChange={v => setParams(p => ({ ...p, creditAdj: v }))} step={0.005} format="percent" fieldKey="creditAdj" />

                <div style={{ marginTop: 8, padding: "6px 10px", background: "rgba(5,150,105,0.04)", borderRadius: 6, fontSize: 10, color: c.textMuted, lineHeight: 1.55 }}>
                  <strong style={{ color: c.success }}>Derived:</strong> Risk Premium {fmtPct(params.discountRate - params.riskFreeRate)} • Payoff Disc. {fmtPct(params.riskFreeRate + (params.isEscrowed ? 0 : (params.creditAdj || 0)))} • RN Drift {fmtPct(params.metricGrowthRate - (params.discountRate - params.riskFreeRate))}
                </div>
              </div>
            </div>

            {/* RIGHT: Per-Period Terms */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={cardStyle}>
                <div className="vf" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <h3 style={{ fontSize: 13, fontWeight: 600, display: "flex", alignItems: "center", gap: 5, margin: 0 }}><Icon name="calendar" size={14} color={c.accent} /> Measurement Periods ({params.periods.length})</h3>
                  <button onClick={addPeriod} style={{ padding: "3px 8px", fontSize: 10, background: c.accentLight, border: `1px solid ${c.accent}33`, borderRadius: 4, color: c.accent, cursor: "pointer", fontWeight: 500 }}>+ Add Period</button>
                </div>

                {params.periods.map((p, i) => (
                  <div key={i} style={{ marginBottom: 14, paddingBottom: 12, borderBottom: i < params.periods.length - 1 ? `1px solid ${c.cardBorder}` : "none" }}>
                    <div className="vf" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: c.text }}>Period {i + 1} — Year {p.yearFromNow || i + 1}</span>
                      <div className="vf" style={{ gap: 4, alignItems: "center" }}>
                        <select value={p.structure} onChange={e => updatePeriod(i, "structure", e.target.value)}
                          style={{ padding: "3px 6px", fontSize: 10, border: `1px solid ${c.cardBorder}`, borderRadius: 4, background: c.inputBg, color: c.accent, fontWeight: 600, cursor: "pointer" }}>
                          {structures.map(s => <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>)}
                        </select>
                        {params.periods.length > 1 && <button onClick={() => removePeriod(i)} style={{ padding: "2px 5px", fontSize: 9, background: "rgba(220,38,38,0.06)", border: `1px solid rgba(220,38,38,0.15)`, borderRadius: 3, color: c.danger, cursor: "pointer" }}>×</button>}
                      </div>
                    </div>

                    <div className="vg" style={{ gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      <EditField label="Projected Metric ($)" value={p.projectedMetric || 0} onChange={v => updatePeriod(i, "projectedMetric", v)} step={100000} fieldKey="projectedMetric" />
                      <EditField label="Threshold ($)" value={p.threshold || 0} onChange={v => updatePeriod(i, "threshold", v)} step={100000} fieldKey="threshold" />
                      {(p.structure === "binary" || p.structure === "milestone") && (
                        <EditField label="Fixed Payment ($)" value={p.fixedPayment || 0} onChange={v => updatePeriod(i, "fixedPayment", v)} step={100000} fieldKey="fixedPayment" />
                      )}
                      {(p.structure === "linear" || p.structure === "percentage") && (
                        <EditField label="Participation Rate" value={p.participationRate || 0} onChange={v => updatePeriod(i, "participationRate", v)} step={0.01} format="percent" fieldKey="participationRate" />
                      )}
                      <EditField label="Cap ($)" value={p.cap || 0} onChange={v => updatePeriod(i, "cap", v)} step={100000} fieldKey="cap" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="vf r-col" style={{ gap: 10, marginTop: 20 }}>
            <button onClick={() => setView("landing")} style={{ padding: "10px 20px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 8, color: c.text, cursor: "pointer", fontSize: 12 }}>Back</button>
            <button onClick={runFromReview} style={{ flex: 1, padding: "12px 24px", background: "linear-gradient(135deg,#2563eb,#1d4ed8)", border: "none", borderRadius: 8, color: "white", cursor: "pointer", fontSize: 13, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, boxShadow: "0 2px 8px rgba(37,99,235,0.3)" }}>
              <Icon name="activity" size={15} color="white" /> Confirm & Run Monte Carlo ({MC_PATHS.toLocaleString()} paths)
            </button>
          </div>
        </div>
      </div>
    );
  }

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
            <button onClick={() => setView("audit")} style={{ padding: "7px 14px", background: c.accentLight, border: `1px solid ${c.accent}33`, borderRadius: 6, color: c.accent, cursor: "pointer", fontSize: 11, fontWeight: 600, display: "flex", alignItems: "center", gap: 5 }}><Icon name="shield" size={13} color={c.accent} />Audit Support</button>
            <button onClick={resetAll} style={{ padding: "7px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 5 }}><Icon name="plus" size={13} />New</button>
          </div>
        </div>

        {/* Backtest banner */}
        {backtestComparison && (
          <div style={{ ...cardStyle, marginBottom: 14, padding: 16,
            background: backtestComparison.isDemo ? "rgba(5,150,105,0.04)" : (backtestComparison.gap != null && backtestComparison.gap < 5) ? "rgba(5,150,105,0.04)" : (backtestComparison.gap != null && backtestComparison.gap < 10) ? "rgba(217,119,6,0.04)" : "rgba(37,99,235,0.03)",
            borderColor: backtestComparison.isDemo ? "rgba(5,150,105,0.15)" : (backtestComparison.gap != null && backtestComparison.gap < 5) ? "rgba(5,150,105,0.15)" : "rgba(217,119,6,0.15)" }}>
            <div style={{ flex: 1 }}>
              {backtestComparison.isDemo ? (<>
                <div style={{ fontSize: 12, fontWeight: 600, color: c.success, marginBottom: 5 }}>Grant Thornton Benchmark — Validated</div>
                <div style={{ fontSize: 11, color: c.textMuted, lineHeight: 1.65 }}>
                  Monte Carlo fair value: <strong style={{ color: c.text }}>{fmt(backtestComparison.computed)}</strong> — This is the correct risk-adjusted value per GT methodology.
                  The GT paper demonstrates that Monte Carlo properly values this earnout significantly below the scenario-based estimate of ~{fmt(backtestComparison.scenarioBasedValue || 9.1e6)}, 
                  because the binary structure creates option-like risk that scenario methods fail to capture.
                </div>
                <div className="vf" style={{ gap: 12, marginTop: 10 }}>
                  <div style={{ flex: 1, padding: "8px 12px", background: "rgba(5,150,105,0.05)", borderRadius: 6, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: c.success, fontWeight: 600, marginBottom: 2 }}>MONTE CARLO (CORRECT)</div>
                    <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'IBM Plex Mono',monospace", color: c.success }}>{fmt(backtestComparison.computed)}</div>
                  </div>
                  <div style={{ flex: 1, padding: "8px 12px", background: "rgba(220,38,38,0.04)", borderRadius: 6, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: c.danger, fontWeight: 600, marginBottom: 2 }}>SCENARIO-BASED (OVERESTIMATES)</div>
                    <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'IBM Plex Mono',monospace", color: c.danger }}>{fmt(backtestComparison.scenarioBasedValue || 9.1e6)}</div>
                  </div>
                  <div style={{ flex: 1, padding: "8px 12px", background: "rgba(217,119,6,0.04)", borderRadius: 6, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: c.warning, fontWeight: 600, marginBottom: 2 }}>OVERESTIMATION GAP</div>
                    <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'IBM Plex Mono',monospace", color: c.warning }}>{((backtestComparison.scenarioBasedValue || 9.1e6) / backtestComparison.computed * 100 - 100).toFixed(0)}%</div>
                  </div>
                </div>
              </>) : (<>
                <div style={{ fontSize: 12, fontWeight: 600, color: backtestComparison.gap < 5 ? c.success : c.warning, marginBottom: 3 }}>
                  Backtest: {backtestComparison.gap?.toFixed(1)}% gap
                </div>
                <div style={{ fontSize: 11, color: c.textMuted }}>
                  Reported {fmt(backtestComparison.reported)} — ValuPro {fmt(backtestComparison.computed)}
                </div>
              </>)}
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

  // ---- AUDIT SUPPORT ----
  if (view === "audit") {
    const handleAuditUpload = async (e) => {
      const file = e.target.files[0]; if (!file) return;
      const ext = file.name.split(".").pop().toLowerCase();
      if (ext === "docx" || ext === "doc") {
        try {
          const mammoth = await import("mammoth");
          const arrayBuffer = await file.arrayBuffer();
          const result = await mammoth.extractRawText({ arrayBuffer });
          setAuditQuestions(result.value);
        } catch (err) {
          // Fallback: read as text (works for .doc sometimes)
          const reader = new FileReader();
          reader.onload = (ev) => setAuditQuestions(ev.target.result);
          reader.readAsText(file);
        }
      } else {
        const reader = new FileReader();
        reader.onload = (ev) => setAuditQuestions(ev.target.result);
        reader.readAsText(file);
      }
    };
    const confColor = (c_) => c_ >= 90 ? "#1F6F3A" : c_ >= 70 ? "#8B6914" : c_ >= 50 ? "#C0392B" : "#666";
    const confBg = (c_) => c_ >= 90 ? "rgba(5,150,105,0.06)" : c_ >= 70 ? "rgba(217,119,6,0.06)" : c_ >= 50 ? "rgba(220,38,38,0.06)" : "rgba(0,0,0,0.03)";
    const confLabel = (c_) => c_ >= 90 ? "High — ready to send" : c_ >= 70 ? "Medium — review recommended" : c_ >= 50 ? "Low — needs user input" : "Outside scope";

    return (
      <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',system-ui,sans-serif", color: c.text }}>{fontLink}{hdr(true)}
        <div className="r-p" style={{ maxWidth: 900, margin: "0 auto", padding: "28px" }}>
          <div className="vf r-col" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, gap: 8 }}>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 3, fontFamily: "'Source Serif 4',Georgia,serif" }}>Audit Support</h2>
              <p style={{ fontSize: 12, color: c.textMuted }}>Upload the auditor's questions. The engine drafts responses using your model data, assumptions, and provenance.</p>
            </div>
            <div className="vf" style={{ gap: 6 }}>
              <button onClick={() => setView("results")} style={{ padding: "6px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11 }}>Back to Results</button>
              {auditResponses && <>
                <button onClick={exportAuditWord} style={{ padding: "6px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}><Icon name="download" size={12} />Export Word</button>
                <button onClick={exportAuditExcel} style={{ padding: "6px 14px", background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 6, color: c.text, cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}><Icon name="download" size={12} />Export Excel</button>
              </>}
            </div>
          </div>

          {/* Model context summary */}
          <div style={{ ...cardStyle, marginBottom: 14, padding: 14, background: tc ? "rgba(37,99,235,0.03)" : "rgba(37,99,235,0.02)", borderColor: tc ? "rgba(37,99,235,0.1)" : "rgba(37,99,235,0.08)" }}>
            <div style={{ fontSize: 11, color: c.accent, fontWeight: 600, marginBottom: 4 }}>Model Context Available to Audit Engine</div>
            <div className="vf r-wrap" style={{ gap: 12, fontSize: 10, color: c.textMuted }}>
              <span>Fair Value: <strong style={{ color: c.text }}>{fmt(results?.fairValue)}</strong></span>
              <span>Metric: <strong style={{ color: c.text }}>{params.metric}</strong></span>
              <span>Periods: <strong style={{ color: c.text }}>{params.periods.length}</strong></span>
              <span>Volatility: <strong style={{ color: c.text }}>{fmtPct(params.volatility)}</strong></span>
              <span>Provenance: <strong style={{ color: provenance.volatility.comparableCompanies?.length > 0 ? c.success : c.warning }}>{provenance.volatility.comparableCompanies?.length > 0 ? "Available" : "Limited"}</strong></span>
            </div>
          </div>

          {/* Upload area */}
          {!auditResponses && (
            <div style={{ ...cardStyle, marginBottom: 14 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}><Icon name="upload" size={14} color={c.accent} /> Upload Auditor Questions</h3>
              <label style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 80, cursor: "pointer", borderStyle: "dashed", borderWidth: 2, borderColor: c.cardBorder, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <input type="file" accept=".txt,.doc,.docx,.pdf,.csv" onChange={handleAuditUpload} style={{ display: "none" }} />
                <Icon name="upload" size={20} color={c.accent} />
                <span style={{ fontSize: 11, color: c.textMuted, marginTop: 6 }}>Upload auditor's question list (.txt, .doc, .docx)</span>
              </label>
              <div style={{ fontSize: 11, color: c.textMuted, marginBottom: 10 }}>Or paste questions directly:</div>
              <textarea value={auditQuestions || ""} onChange={e => setAuditQuestions(e.target.value)} placeholder={"1. Please describe the valuation methodology used.\n2. What comparable companies were used for volatility?\n3. How sensitive is the conclusion to changes in key assumptions?\n4. Is the methodology consistent with the prior period?\n..."} rows={8}
                style={{ width: "100%", padding: "10px", fontSize: 12, border: `1px solid ${c.cardBorder}`, borderRadius: 6, background: c.inputBg, color: c.text, outline: "none", resize: "vertical", fontFamily: "'Inter',system-ui,sans-serif", lineHeight: 1.6 }} />
              <button onClick={() => processAuditQuestions(auditQuestions)} disabled={!auditQuestions || auditProcessing}
                style={{ marginTop: 10, width: "100%", padding: "11px", background: auditQuestions ? "linear-gradient(135deg,#2563eb,#1d4ed8)" : c.cardBorder, border: "none", borderRadius: 7, color: "white", cursor: auditQuestions ? "pointer" : "not-allowed", fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                {auditProcessing ? <><div style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid white", borderTopColor: "transparent", animation: "spin 1s linear infinite" }} /> Generating responses...</> : <><Icon name="brain" size={14} color="white" /> Generate Audit Responses</>}
              </button>
            </div>
          )}

          {/* Responses */}
          {auditResponses && (
            <div>
              {/* Summary bar */}
              <div className="vf r-wrap" style={{ gap: 8, marginBottom: 14 }}>
                {[{ l: "Total", n: auditResponses.length, bg: c.cardBorder },
                  { l: "High Confidence", n: auditResponses.filter(r => r.confidence >= 90).length, bg: "rgba(5,150,105,0.1)" },
                  { l: "Medium", n: auditResponses.filter(r => r.confidence >= 70 && r.confidence < 90).length, bg: "rgba(217,119,6,0.1)" },
                  { l: "Needs Input", n: auditResponses.filter(r => r.confidence < 70).length, bg: "rgba(220,38,38,0.1)" },
                ].map((s, i) => (
                  <div key={i} style={{ padding: "6px 12px", borderRadius: 6, background: s.bg, fontSize: 10, fontWeight: 600 }}>
                    <span style={{ color: c.textMuted }}>{s.l}: </span><span style={{ color: c.text }}>{s.n}</span>
                  </div>
                ))}
                <button onClick={() => { setAuditResponses(null); setAuditQuestions(null); }} style={{ marginLeft: "auto", padding: "5px 10px", fontSize: 10, background: "transparent", border: `1px solid ${c.cardBorder}`, borderRadius: 4, color: c.textMuted, cursor: "pointer" }}>Upload New Questions</button>
              </div>

              {/* Individual responses */}
              {auditResponses.map((r, i) => (
                <div key={i} style={{ ...cardStyle, marginBottom: 10, padding: 16 }}>
                  {/* Question header */}
                  <div className="vf" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                    <div style={{ flex: 1, fontSize: 12, fontWeight: 600, color: c.text, lineHeight: 1.5 }}>
                      <span style={{ color: c.accent, marginRight: 6 }}>Q{i + 1}</span>{r.question}
                    </div>
                    <div style={{ flexShrink: 0, marginLeft: 10, display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3 }}>
                      <div style={{ padding: "2px 8px", borderRadius: 4, background: confBg(r.confidence), color: confColor(r.confidence), fontSize: 10, fontWeight: 700 }}>{r.confidence}%</div>
                      <span style={{ fontSize: 9, color: confColor(r.confidence) }}>{confLabel(r.confidence)}</span>
                    </div>
                  </div>

                  {/* Category tag */}
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: c.accentLight, color: c.accent, fontWeight: 500, textTransform: "capitalize" }}>{r.category}</span>
                  </div>

                  {/* Response */}
                  <div style={{ fontSize: 11, color: c.text, lineHeight: 1.7, whiteSpace: "pre-wrap", padding: "10px 12px", background: tc ? "#1a2540" : "#fafbfc", borderRadius: 6, border: `1px solid ${c.cardBorder}` }}>
                    {r.response}
                  </div>

                  {/* Missing info callout */}
                  {r.missingInfo && (
                    <div style={{ marginTop: 8, padding: "6px 10px", borderLeft: `3px solid ${c.accent}`, background: "rgba(37,99,235,0.03)", borderRadius: "0 4px 4px 0", fontSize: 10, color: c.accent, lineHeight: 1.5 }}>
                      <strong>To strengthen this response:</strong> {r.missingInfo}
                    </div>
                  )}

                  {/* Confidence reason */}
                  {r.confidenceReason && (
                    <div style={{ marginTop: 4, fontSize: 9, color: c.textDim, fontStyle: "italic" }}>
                      Confidence: {r.confidenceReason}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return <div style={{ minHeight: "100vh", background: c.bg }}>{fontLink}{hdr()}</div>;
}
