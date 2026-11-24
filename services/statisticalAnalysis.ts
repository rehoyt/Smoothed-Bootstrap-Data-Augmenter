
import { CsvData, Report, DescriptiveStatReport, StatisticalTestResult, Stats, ReportSummary, ColumnMetadata } from '../types';
import Papa from 'papaparse';
import { jStat } from 'jstat';

const SIGNIFICANCE_LEVEL = 0.05;
const TVD_THRESHOLD = 0.1;
const CARDINALITY_THRESHOLD = 10;

const calculateCramersV = (chiSq: number, n: number, categoriesCount: number): number => {
    if (n === 0 || categoriesCount <= 1) return 0;
    const v = Math.sqrt(chiSq / (n * (categoriesCount - 1)));
    return isNaN(v) ? 0 : v;
};

const calculateTVDFromCounts = (
    origCounts: Record<string, number>,
    nOrig: number,
    augCounts: Record<string, number>,
    nAug: number
): number => {
    const invOrig = 1 / (nOrig || 1);
    const invAug = 1 / (nAug || 1);
    let sumDiff = 0;

    for (const cat in origCounts) {
        const p = origCounts[cat] * invOrig;
        const q = (augCounts[cat] || 0) * invAug;
        sumDiff += Math.abs(p - q);
    }
    for (const cat in augCounts) {
        if (!(cat in origCounts)) {
            sumDiff += augCounts[cat] * invAug;
        }
    }

    return sumDiff / 2;
};

// Zero-allocation O(n1 + n2) Mann-Whitney U on pre-sorted ascending arrays
const calculateMannWhitneyUSorted = (s1: number[], s2: number[]): { U: number; p: number } => {
    const n1 = s1.length;
    const n2 = s2.length;
    if (n1 === 0 || n2 === 0) return { U: 0, p: 1 };

    let i = 0;
    let j = 0;
    let rank = 1;
    let R1 = 0;

    while (i < n1 || j < n2) {
        const val = (i < n1 && j < n2)
            ? (s1[i] <= s2[j] ? s1[i] : s2[j])
            : (i < n1 ? s1[i] : s2[j]);

        let c1 = 0;
        while (i < n1 && s1[i] === val) {
            i++;
            c1++;
        }
        let c2 = 0;
        while (j < n2 && s2[j] === val) {
            j++;
            c2++;
        }

        const totalTies = c1 + c2;
        const avgRank = rank + (totalTies - 1) / 2;
        if (c1 > 0) {
            R1 += c1 * avgRank;
        }
        rank += totalTies;
    }

    const U1 = R1 - (n1 * (n1 + 1)) / 2;
    const mu_U = (n1 * n2) / 2;
    const sigma_U = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);

    if (sigma_U === 0) return { U: U1, p: 1 };
    const z = (U1 - mu_U) / sigma_U;
    const p = 2 * (1 - jStat.normal.cdf(Math.abs(z), 0, 1));
    return { U: U1, p: isNaN(p) ? 1 : p };
};

// O(n1 + n2) two-pointer Kolmogorov-Smirnov test on pre-sorted ascending arrays
const calculateKSTestSorted = (s1: number[], s2: number[]): { D: number; p: number } => {
    const n1 = s1.length;
    const n2 = s2.length;
    if (n1 === 0 || n2 === 0) return { D: 0, p: 1 };

    const inv1 = 1 / n1;
    const inv2 = 1 / n2;
    let i = 0;
    let j = 0;
    let maxDiff = 0;

    while (i < n1 || j < n2) {
        const val = (i < n1 && j < n2)
            ? (s1[i] <= s2[j] ? s1[i] : s2[j])
            : (i < n1 ? s1[i] : s2[j]);

        while (i < n1 && s1[i] <= val) i++;
        while (j < n2 && s2[j] <= val) j++;

        const diff = Math.abs(i * inv1 - j * inv2);
        if (diff > maxDiff) maxDiff = diff;
    }

    const en = (n1 * n2) / (n1 + n2);
    const p = 2 * Math.exp(-2 * en * maxDiff * maxDiff);
    return { D: maxDiff, p: Math.min(isNaN(p) ? 1 : p, 1) };
};

const extractSortedNumbers = (data: CsvData, col: string): number[] => {
    const out: number[] = [];
    for (let i = 0; i < data.length; i++) {
        const raw = data[i][col];
        if (raw === null || raw === undefined || raw === '') continue;
        const v = typeof raw === 'number' ? raw : Number(raw);
        if (!isNaN(v)) out.push(v);
    }
    out.sort((a, b) => a - b);
    return out;
};

const computeMeanAndSampleStd = (values: number[]): { mean: number; std: number } => {
    const n = values.length;
    if (n === 0) return { mean: 0, std: 0 };
    let sum = 0;
    for (let i = 0; i < n; i++) sum += values[i];
    const mean = sum / n;
    if (n < 2) return { mean, std: 0 };
    let sqDiffSum = 0;
    for (let i = 0; i < n; i++) {
        const d = values[i] - mean;
        sqDiffSum += d * d;
    }
    return { mean, std: Math.sqrt(sqDiffSum / (n - 1)) };
};

export const getDetailedColumnMetadata = (
    data: CsvData,
    customColumnTypes?: Record<string, 'continuous' | 'numerical' | 'categorical' | 'ordinal'>
): Record<string, ColumnMetadata> => {
  if (data.length === 0) return {};
  const metadata: Record<string, ColumnMetadata> = {};
  const headers = Object.keys(data[0]);
  const rowCount = data.length;

  for (const header of headers) {
    const uniqueValues = new Set<string | number>();
    let validCount = 0;
    let isNumericType = true;
    let numericChecked = 0;
    let maxPrecision = 0;
    let minVal = Infinity;
    let maxVal = -Infinity;
    let hasNumeric = false;

    for (let i = 0; i < rowCount; i++) {
        const val = data[i][header];
        if (val === null || val === undefined) continue;
        if (typeof val === 'string' && val.trim() === '') continue;

        validCount++;
        uniqueValues.add(val);

        if (isNumericType && numericChecked < 100) {
            numericChecked++;
            const num = typeof val === 'number' ? val : Number(String(val).trim());
            if (isNaN(num)) {
                isNumericType = false;
            }
        }
    }

    if (validCount === 0) isNumericType = false;

    if (isNumericType) {
        for (let i = 0; i < rowCount; i++) {
            const val = data[i][header];
            if (val === null || val === undefined) continue;
            if (typeof val === 'string' && val.trim() === '') continue;
            const num = typeof val === 'number' ? val : Number(val);
            if (!isNaN(num)) {
                hasNumeric = true;
                if (num < minVal) minVal = num;
                if (num > maxVal) maxVal = num;
                const str = String(val);
                const dotIdx = str.indexOf('.');
                if (dotIdx !== -1) {
                    const prec = str.length - dotIdx - 1;
                    if (prec > maxPrecision) maxPrecision = prec;
                }
            }
        }
    }

    const uniqueCount = uniqueValues.size;
    const precision = isNumericType ? maxPrecision : 0;

    // Heuristic default detection
    const detectedType: 'continuous' | 'categorical' | 'ordinal' = !isNumericType
        ? 'categorical'
        : uniqueCount <= CARDINALITY_THRESHOLD
        ? 'categorical'
        : 'continuous';

    let type: 'continuous' | 'numerical' | 'categorical' | 'ordinal';
    if (customColumnTypes && customColumnTypes[header]) {
        const userT = customColumnTypes[header];
        type = userT === 'numerical' ? 'continuous' : userT;
    } else {
        type = detectedType;
    }

    // Distinct sample values for UI preview
    const sampleValues: (string | number)[] = [];
    for (const v of uniqueValues) {
        sampleValues.push(v);
        if (sampleValues.length >= 5) break;
    }

    metadata[header] = {
        type,
        detectedType,
        uniqueCount,
        isHeuristicCategorical: isNumericType && (type === 'categorical' || type === 'ordinal'),
        precision,
        sampleValues,
        isNumeric: isNumericType,
        min: hasNumeric ? minVal : undefined,
        max: hasNumeric ? maxVal : undefined
    };
  }
  return metadata;
};

export const generateReport = (
    originalData: CsvData, 
    augmentedData: CsvData,
    customColumnTypes?: Record<string, 'continuous' | 'numerical' | 'categorical' | 'ordinal'>
): Report => {
  const columnMetadata = getDetailedColumnMetadata(originalData, customColumnTypes);
  const headers = Object.keys(columnMetadata);
  const numericalColumns = headers.filter(k => columnMetadata[k].type === 'continuous' || columnMetadata[k].type === 'numerical');
  const categoricalColumns = headers.filter(k => columnMetadata[k].type === 'categorical' || columnMetadata[k].type === 'ordinal');

  const descriptiveStats: DescriptiveStatReport[] = [];
  const tTest: StatisticalTestResult[] = [];
  const mannWhitney: StatisticalTestResult[] = [];
  const ksTest: StatisticalTestResult[] = [];

  for (const col of numericalColumns) {
      const orig = extractSortedNumbers(originalData, col);
      const aug = extractSortedNumbers(augmentedData, col);
      const uniqueCount = columnMetadata[col].uniqueCount;

      const origBasic = computeMeanAndSampleStd(orig);
      const augBasic = computeMeanAndSampleStd(aug);

      // Descriptive stats & 95% CI
      if (orig.length >= 2 && aug.length >= 2) {
          const originalStats: Stats = {
              mean: origBasic.mean,
              std: origBasic.std,
              min: orig[0],
              max: orig[orig.length - 1],
              count: orig.length,
          };
          const augmentedStats: Stats = {
              mean: augBasic.mean,
              std: augBasic.std,
              min: aug[0],
              max: aug[aug.length - 1],
              count: aug.length,
          };

          const n1 = originalStats.count;
          const n2 = augmentedStats.count;
          const s1 = originalStats.std;
          const s2 = augmentedStats.std;
          const diff = augmentedStats.mean - originalStats.mean;
          const v1 = (s1 * s1) / n1;
          const v2 = (s2 * s2) / n2;
          const se = Math.sqrt(v1 + v2);
          const df_num = Math.pow(v1 + v2, 2);
          const df_den = (v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1);
          const df = df_den > 0 ? df_num / df_den : 0;
          const t_crit = df > 0 ? jStat.studentt.inv(1 - 0.025, df) : 1.96;
          const marginOfError = t_crit * se;

          descriptiveStats.push({
              column: col,
              original: originalStats,
              augmented: augmentedStats,
              meanDifferenceCI: { diff, lower: diff - marginOfError, upper: diff + marginOfError },
          });
      }

      // One-sample t-test (matching original jStat.ttest(augMean, orig) / jStat.tscore(augMean, orig))
      if (orig.length < 2 || aug.length < 1) {
          tTest.push({ column: col, statistic: 0, pValue: 1, isSimilar: true, uniqueCount });
      } else {
          const seOrig = origBasic.std / Math.sqrt(orig.length);
          const tStat = seOrig > 0 ? (augBasic.mean - origBasic.mean) / seOrig : 0;
          let p = seOrig > 0 ? jStat.ttest(tStat, orig.length, 2) : 1;
          if (isNaN(p)) p = 1;
          const safeStat = isNaN(tStat) || !isFinite(tStat) ? 0 : tStat;

          tTest.push({
              column: col,
              statistic: safeStat,
              pValue: p,
              isSimilar: p > SIGNIFICANCE_LEVEL,
              uniqueCount,
          });
      }

      // Mann-Whitney U test (reusing sorted arrays)
      const mwRes = calculateMannWhitneyUSorted(orig, aug);
      mannWhitney.push({
          column: col,
          statistic: mwRes.U,
          pValue: mwRes.p,
          isSimilar: mwRes.p > SIGNIFICANCE_LEVEL,
          uniqueCount,
      });

      // Kolmogorov-Smirnov test (reusing sorted arrays)
      const ksRes = calculateKSTestSorted(orig, aug);
      ksTest.push({
          column: col,
          statistic: ksRes.D,
          pValue: ksRes.p,
          isSimilar: ksRes.p > SIGNIFICANCE_LEVEL,
          uniqueCount,
      });
  }

  const chiSquare: StatisticalTestResult[] = [];
  const tvdResults: { column: string; value: number; isSimilar: boolean }[] = [];
  const origLen = originalData.length;
  const augLen = augmentedData.length;

  for (const col of categoricalColumns) {
      const origCounts: Record<string, number> = {};
      for (let i = 0; i < origLen; i++) {
          const raw = originalData[i][col];
          const key = raw === null || raw === undefined ? 'null' : String(raw);
          origCounts[key] = (origCounts[key] || 0) + 1;
      }

      const augCounts: Record<string, number> = {};
      for (let i = 0; i < augLen; i++) {
          const raw = augmentedData[i][col];
          const key = raw === null || raw === undefined ? 'null' : String(raw);
          augCounts[key] = (augCounts[key] || 0) + 1;
      }

      const tvdVal = calculateTVDFromCounts(origCounts, origLen, augCounts, augLen);
      tvdResults.push({ column: col, value: tvdVal, isSimilar: tvdVal < TVD_THRESHOLD });

      const cats = Object.keys(origCounts);
      if (cats.length >= 2 && origLen > 0 && augLen > 0) {
          let stat = 0;
          const scale = augLen / origLen;
          for (let c = 0; c < cats.length; c++) {
              const cat = cats[c];
              const obs = augCounts[cat] || 0;
              const exp = origCounts[cat] * scale;
              if (exp > 0) {
                  const diff = obs - exp;
                  stat += (diff * diff) / exp;
              }
          }
          let p = 1 - jStat.chisquare.cdf(stat, cats.length - 1);
          if (isNaN(p)) p = 1;
          const v = calculateCramersV(stat, augLen, cats.length);
          chiSquare.push({
              column: col,
              statistic: stat,
              pValue: p,
              cramersV: v,
              isSimilar: p > SIGNIFICANCE_LEVEL,
              uniqueCount: columnMetadata[col].uniqueCount,
          });
      }
  }

  const colTypes: Record<string, 'continuous' | 'numerical' | 'categorical' | 'ordinal'> = {};
  headers.forEach(h => colTypes[h] = columnMetadata[h].type);

  return {
    descriptiveStats, tTest, mannWhitney, ksTest, chiSquare, 
    totalVariationDistance: tvdResults,
    columnTypes: colTypes,
    columnMetadata,
    summary: {
        tTestSimilarCount: tTest.filter(r => r.isSimilar).length,
        mannWhitneySimilarCount: mannWhitney.filter(r => r.isSimilar).length,
        ksTestSimilarCount: ksTest.filter(r => r.isSimilar).length,
        chiSquareSimilarCount: chiSquare.filter(r => r.isSimilar).length,
        avgTVD: tvdResults.length > 0 ? tvdResults.reduce((s, r) => s + r.value, 0) / tvdResults.length : 0,
        avgCramersV: chiSquare.length > 0 ? chiSquare.reduce((s, r) => s + (r.cramersV || 0), 0) / chiSquare.length : 0,
        totalNumerical: numericalColumns.length,
        totalContinuous: numericalColumns.length,
        totalCategorical: categoricalColumns.length,
        totalOrdinal: headers.filter(k => columnMetadata[k].type === 'ordinal').length,
    }
  };
};

export const generateStatsSummaryCsv = (report: Report): string => {
    const rows: any[] = [];
    const allCols = Object.keys(report.columnTypes);

    allCols.forEach(col => {
        const meta = report.columnMetadata[col];
        const type = meta.type;
        const row: any = { 
            Variable: col, 
            Type: type, 
            UniqueValues: meta.uniqueCount, 
            Heuristic: meta.isHeuristicCategorical ? 'Discrete' : 'Standard' 
        };

        const desc = report.descriptiveStats.find(s => s.column === col);
        const tt = report.tTest.find(s => s.column === col);
        const mw = report.mannWhitney.find(s => s.column === col);
        const ks = report.ksTest.find(s => s.column === col);
        const cs = report.chiSquare?.find(s => s.column === col);
        const tvd = report.totalVariationDistance?.find(s => s.column === col);

        // Populate numerical statistics if present
        row['Mean (Orig)'] = desc ? desc.original.mean.toFixed(meta.precision) : 'N/A';
        row['Mean (Aug)'] = desc ? desc.augmented.mean.toFixed(meta.precision) : 'N/A';
        row['Mean Difference'] = desc ? desc.meanDifferenceCI.diff.toFixed(meta.precision) : 'N/A';
        row['CI Lower'] = desc ? desc.meanDifferenceCI.lower.toFixed(meta.precision) : 'N/A';
        row['CI Upper'] = desc ? desc.meanDifferenceCI.upper.toFixed(meta.precision) : 'N/A';
        row['T-Test P'] = tt ? tt.pValue.toFixed(5) : 'N/A';
        row['Mann-Whitney P'] = mw ? mw.pValue.toFixed(5) : 'N/A';
        row['KS-Test P'] = ks ? ks.pValue.toFixed(5) : 'N/A';

        // Populate categorical statistics if present
        row['Chi-Square P'] = cs ? cs.pValue.toFixed(5) : 'N/A';
        row['Cramers V'] = cs && cs.cramersV !== undefined ? cs.cramersV.toFixed(4) : 'N/A';
        row['TVD'] = tvd ? tvd.value.toFixed(5) : 'N/A';

        // Determine Similarity
        const isNumericSimilar = tt && mw && ks ? (tt.isSimilar && mw.isSimilar && ks.isSimilar) : true;
        const isCategoricalSimilar = cs && tvd ? (cs.isSimilar || cs.cramersV < 0.1 || tvd.isSimilar) : true;
        
        row['Is Similar'] = (isNumericSimilar && isCategoricalSimilar) ? 'YES' : 'NO';

        rows.push(row);
    });

    return Papa.unparse(rows);
};

export const formatReportAsText = (report: Report): string => {
    let text = 'BootsGN Statistical Comparison Report\n======================================\n\n';
    if (report.descriptiveStats.length > 0) {
        text += '1. Descriptive Statistics\n-------------------------\n';
        report.descriptiveStats.forEach(s => {
            const meta = report.columnMetadata[s.column];
            text += `\n${s.column}:\n  Mean: ${s.original.mean.toFixed(meta.precision)} -> ${s.augmented.mean.toFixed(meta.precision)}\n`;
            text += `  95% CI Diff: [${s.meanDifferenceCI.lower.toFixed(meta.precision)}, ${s.meanDifferenceCI.upper.toFixed(meta.precision)}]\n`;
        });
    }
    return text + '\n... (detailed p-values, TVD, and Cramers V available in CSV Summary)';
};
