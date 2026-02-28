
import { CsvData, Report, DescriptiveStatReport, StatisticalTestResult, Stats, ReportSummary, ColumnMetadata } from '../types';
import Papa from 'papaparse';
import { jStat } from 'jstat';

const SIGNIFICANCE_LEVEL = 0.05;
const TVD_THRESHOLD = 0.1;
const CARDINALITY_THRESHOLD = 10;

const getPrecision = (values: (string | number)[]): number => {
    let maxPrecision = 0;
    for (const val of values) {
        if (val === null || val === undefined) continue;
        const str = val.toString();
        if (str.includes('.')) {
            const precision = str.split('.')[1].length;
            if (precision > maxPrecision) maxPrecision = precision;
        }
    }
    return maxPrecision;
};

const calculateCramersV = (chiSq: number, n: number, categoriesCount: number): number => {
    if (n === 0 || categoriesCount <= 1) return 0;
    const v = Math.sqrt(chiSq / (n * (categoriesCount - 1)));
    return isNaN(v) ? 0 : v;
};

const calculateTVD = (origValues: any[], augValues: any[]): number => {
    const origCounts: Record<string, number> = {};
    origValues.forEach(v => {
        const key = v === null || v === undefined ? 'null' : String(v);
        origCounts[key] = (origCounts[key] || 0) + 1;
    });
    const augCounts: Record<string, number> = {};
    augValues.forEach(v => {
        const key = v === null || v === undefined ? 'null' : String(v);
        augCounts[key] = (augCounts[key] || 0) + 1;
    });

    const allCats = new Set([...Object.keys(origCounts), ...Object.keys(augCounts)]);
    let sumDiff = 0;
    const nOrig = origValues.length || 1;
    const nAug = augValues.length || 1;

    allCats.forEach(cat => {
        const p = (origCounts[cat] || 0) / nOrig;
        const q = (augCounts[cat] || 0) / nAug;
        sumDiff += Math.abs(p - q);
    });

    return sumDiff / 2;
};

const calculateMannWhitneyU = (arr1: number[], arr2: number[]): { U: number; p: number } => {
    const n1 = arr1.length;
    const n2 = arr2.length;
    if (n1 === 0 || n2 === 0) return { U: 0, p: 1 };

    const combined = [
        ...arr1.map(value => ({ value, group: 1 })),
        ...arr2.map(value => ({ value, group: 2 }))
    ].sort((a, b) => a.value - b.value);

    let rank = 1;
    const ranked = [];
    let i = 0;
    while (i < combined.length) {
        let j = i;
        while (j < combined.length - 1 && combined[j].value === combined[j + 1].value) j++;
        const avgRank = (rank + (rank + (j - i))) / 2;
        for (let k = i; k <= j; k++) ranked.push({ ...combined[k], rank: avgRank });
        rank += (j - i) + 1;
        i = j + 1;
    }

    const R1 = ranked.filter(d => d.group === 1).reduce((sum, d) => sum + d.rank, 0);
    const U1 = R1 - (n1 * (n1 + 1)) / 2;
    const mu_U = (n1 * n2) / 2;
    const sigma_U = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
    
    if (sigma_U === 0) return { U: U1, p: 1 };
    const z = (U1 - mu_U) / sigma_U;
    const p = 2 * (1 - jStat.normal.cdf(Math.abs(z), 0, 1));
    return { U: U1, p: isNaN(p) ? 1 : p };
};

const calculateKSTest = (arr1: number[], arr2: number[]): { D: number; p: number } => {
    const n1 = arr1.length;
    const n2 = arr2.length;
    if (n1 === 0 || n2 === 0) return { D: 0, p: 1 };

    const allValues = Array.from(new Set([...arr1, ...arr2])).sort((a, b) => a - b);
    let maxDiff = 0;
    for (const val of allValues) {
        const cdf1 = arr1.filter(v => v <= val).length / n1;
        const cdf2 = arr2.filter(v => v <= val).length / n2;
        maxDiff = Math.max(maxDiff, Math.abs(cdf1 - cdf2));
    }
    const en = (n1 * n2) / (n1 + n2);
    const p = 2 * Math.exp(-2 * en * maxDiff * maxDiff);
    return { D: maxDiff, p: Math.min(isNaN(p) ? 1 : p, 1) };
};

const getDetailedColumnMetadata = (data: CsvData): Record<string, ColumnMetadata> => {
  if (data.length === 0) return {};
  const metadata: Record<string, ColumnMetadata> = {};
  const headers = Object.keys(data[0]);
  
  for (const header of headers) {
    const values = data.map(r => r[header]).filter(v => v !== null && v !== undefined);
    const uniqueValues = new Set(values);
    const uniqueCount = uniqueValues.size;
    
    let isNumericType = values.length > 0;
    for (let i = 0; i < Math.min(values.length, 50); i++) {
        if (typeof values[i] !== 'number') {
            isNumericType = false;
            break;
        }
    }

    const precision = isNumericType ? getPrecision(values) : 0;
    const isHeuristicCategorical = isNumericType && uniqueCount <= CARDINALITY_THRESHOLD;
    const type = (isNumericType && !isHeuristicCategorical) ? 'numerical' : 'categorical';

    metadata[header] = {
        type,
        uniqueCount,
        isHeuristicCategorical,
        precision
    };
  }
  return metadata;
};

const getDescriptiveStats = (originalData: CsvData, augmentedData: CsvData, columns: string[]): DescriptiveStatReport[] => {
    const reports: DescriptiveStatReport[] = [];
    for (const col of columns) {
        const originalValues = originalData.map(row => row[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
        const augmentedValues = augmentedData.map(row => row[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
        if (originalValues.length < 2 || augmentedValues.length < 2) continue;

        const originalStats: Stats = {
            mean: jStat.mean(originalValues),
            std: jStat.stdev(originalValues, true),
            min: jStat.min(originalValues),
            max: jStat.max(originalValues),
            count: originalValues.length,
        };
        const augmentedStats: Stats = {
            mean: jStat.mean(augmentedValues),
            std: jStat.stdev(augmentedValues, true),
            min: jStat.min(augmentedValues),
            max: jStat.max(augmentedValues),
            count: augmentedValues.length,
        };
        
        const n1 = originalStats.count;
        const n2 = augmentedStats.count;
        const s1 = originalStats.std;
        const s2 = augmentedStats.std;
        const diff = augmentedStats.mean - originalStats.mean;
        const se = Math.sqrt((s1 * s1 / n1) + (s2 * s2 / n2));
        const df_num = Math.pow((s1 * s1 / n1) + (s2 * s2 / n2), 2);
        const df_den = (Math.pow(s1 * s1 / n1, 2) / (n1 - 1)) + (Math.pow(s2 * s2 / n2, 2) / (n2 - 1));
        const df = df_den > 0 ? df_num / df_den : 0;
        const t_crit = df > 0 ? jStat.studentt.inv(1 - 0.025, df) : 1.96;
        const marginOfError = t_crit * se;

        reports.push({
            column: col,
            original: originalStats,
            augmented: augmentedStats,
            meanDifferenceCI: { diff, lower: diff - marginOfError, upper: diff + marginOfError },
        });
    }
    return reports;
};

export const generateReport = (originalData: CsvData, augmentedData: CsvData): Report => {
  const columnMetadata = getDetailedColumnMetadata(originalData);
  const headers = Object.keys(columnMetadata);
  const numericalColumns = headers.filter(k => columnMetadata[k].type === 'numerical');
  const categoricalColumns = headers.filter(k => columnMetadata[k].type === 'categorical');

  const descriptiveStats = getDescriptiveStats(originalData, augmentedData, numericalColumns);
  
  const tTest = numericalColumns.map(col => {
      const orig = originalData.map(r => r[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
      const aug = augmentedData.map(r => r[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
      if (orig.length < 2 || aug.length < 1) return { column: col, statistic: 0, pValue: 1, isSimilar: true };
      
      const augMean = jStat.mean(aug);
      let p = jStat.ttest(augMean, orig);
      if (isNaN(p)) p = 1;

      return { 
          column: col, 
          statistic: jStat.tscore(augMean, orig), 
          pValue: p, 
          isSimilar: p > SIGNIFICANCE_LEVEL,
          uniqueCount: columnMetadata[col].uniqueCount
      };
  });

  const mannWhitney = numericalColumns.map(col => {
      const orig = originalData.map(r => r[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
      const aug = augmentedData.map(r => r[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
      const res = calculateMannWhitneyU(orig, aug);
      return { 
          column: col, 
          statistic: res.U, 
          pValue: res.p, 
          isSimilar: res.p > SIGNIFICANCE_LEVEL,
          uniqueCount: columnMetadata[col].uniqueCount
      };
  });

  const ksTest = numericalColumns.map(col => {
      const orig = originalData.map(r => r[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
      const aug = augmentedData.map(r => r[col] as number).filter(v => typeof v === 'number' && !isNaN(v));
      const res = calculateKSTest(orig, aug);
      return { 
          column: col, 
          statistic: res.D, 
          pValue: res.p, 
          isSimilar: res.p > SIGNIFICANCE_LEVEL,
          uniqueCount: columnMetadata[col].uniqueCount
      };
  });

  const chiSquare = categoricalColumns.map(col => {
      const origCounts: Record<string, number> = {};
      originalData.forEach(r => {
          const key = r[col] === null || r[col] === undefined ? 'null' : String(r[col]);
          origCounts[key] = (origCounts[key] || 0) + 1;
      });
      const augCounts: Record<string, number> = {};
      augmentedData.forEach(r => {
          const key = r[col] === null || r[col] === undefined ? 'null' : String(r[col]);
          augCounts[key] = (augCounts[key] || 0) + 1;
      });
      
      const cats = Object.keys(origCounts);
      if (cats.length < 2) return null;
      const observed = cats.map(c => augCounts[c] || 0);
      const expected = cats.map(c => (origCounts[c] / originalData.length) * augmentedData.length);
      const stat = observed.reduce((sum, obs, i) => sum + (expected[i] > 0 ? Math.pow(obs - expected[i], 2) / expected[i] : 0), 0);
      let p = 1 - jStat.chisquare.cdf(stat, cats.length - 1);
      if (isNaN(p)) p = 1;
      const v = calculateCramersV(stat, augmentedData.length, cats.length);
      return { 
          column: col, 
          statistic: stat, 
          pValue: p, 
          cramersV: v, 
          isSimilar: p > SIGNIFICANCE_LEVEL,
          uniqueCount: columnMetadata[col].uniqueCount
      };
  }).filter(r => r !== null) as StatisticalTestResult[];

  const tvdResults = categoricalColumns.map(col => {
    const val = calculateTVD(originalData.map(r => r[col]), augmentedData.map(r => r[col]));
    return { column: col, value: val, isSimilar: val < TVD_THRESHOLD };
  });

  const colTypes: Record<string, 'numerical' | 'categorical'> = {};
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
        totalCategorical: categoricalColumns.length,
    }
  };
};

export const generateStatsSummaryCsv = (report: Report): string => {
    const rows: any[] = [];
    const allCols = Object.keys(report.columnTypes);

    allCols.forEach(col => {
        const meta = report.columnMetadata[col];
        const type = meta.type;
        const row: any = { Variable: col, Type: type, UniqueValues: meta.uniqueCount, Heuristic: meta.isHeuristicCategorical ? 'Categorical' : 'Standard' };

        if (type === 'numerical') {
            const desc = report.descriptiveStats.find(s => s.column === col);
            const tt = report.tTest.find(s => s.column === col);
            const mw = report.mannWhitney.find(s => s.column === col);
            const ks = report.ksTest.find(s => s.column === col);

            row['T-Test P'] = tt?.pValue.toFixed(5) || 'N/A';
            row['Mann-Whitney P'] = mw?.pValue.toFixed(5) || 'N/A';
            row['KS-Test P'] = ks?.pValue.toFixed(5) || 'N/A';
            row['Mean Difference'] = desc?.meanDifferenceCI.diff.toFixed(meta.precision) || 'N/A';
            row['CI Lower'] = desc?.meanDifferenceCI.lower.toFixed(meta.precision) || 'N/A';
            row['CI Upper'] = desc?.meanDifferenceCI.upper.toFixed(meta.precision) || 'N/A';
            row['Is Similar'] = (tt?.isSimilar && mw?.isSimilar && ks?.isSimilar) ? 'YES' : 'NO';
        } else {
            const cs = report.chiSquare.find(s => s.column === col);
            const tvd = report.totalVariationDistance.find(s => s.column === col);
            row['Chi-Square P'] = cs?.pValue.toFixed(5) || 'N/A';
            row['Cramers V'] = cs?.cramersV?.toFixed(4) || 'N/A';
            row['TVD'] = tvd?.value.toFixed(5) || 'N/A';
            row['Is Similar'] = (cs?.isSimilar || (cs?.cramersV !== undefined && cs.cramersV < 0.1) || tvd?.isSimilar) ? 'YES' : 'NO';
        }
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
