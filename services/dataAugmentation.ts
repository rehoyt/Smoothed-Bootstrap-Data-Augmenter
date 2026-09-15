
import { CsvData } from '../types';
import { jStat } from 'jstat';

const NOISE_FACTOR = 0.05; 
const K_NEIGHBORS = 5; 
const CATEGORICAL_MISMATCH_PENALTY = 1.0; 

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

const getColumnTypes = (
  data: CsvData,
  customColumnTypes?: Record<string, 'continuous' | 'numerical' | 'categorical' | 'ordinal'>
): Record<string, { type: 'continuous' | 'numerical' | 'categorical' | 'ordinal', precision: number }> => {
  if (data.length === 0) return {};
  const columnTypes: Record<string, { type: 'continuous' | 'numerical' | 'categorical' | 'ordinal', precision: number }> = {};
  const headers = Object.keys(data[0]);

  for (const header of headers) {
    const values = data.map(row => row[header]).filter(val => val !== null && val !== undefined);
    const uniqueValues = new Set(values);
    
    const isNumericType = values.length > 0 && values.slice(0, 50).every(val => 
      val !== null && val !== undefined && String(val).trim() !== '' && !isNaN(Number(String(val).trim()))
    );

    const precision = isNumericType ? getPrecision(values) : 0;

    let type: 'continuous' | 'numerical' | 'categorical' | 'ordinal';
    if (customColumnTypes && customColumnTypes[header]) {
      type = customColumnTypes[header];
    } else {
      type = (isNumericType && uniqueValues.size > 10) ? 'continuous' : 'categorical';
    }

    columnTypes[header] = { type, precision };
  }
  return columnTypes;
};

const getColumnStats = (data: CsvData, numericalColumns: string[]) => {
    const stats: Record<string, { std: number; mean: number; min: number; max: number; isInteger: boolean }> = {};
    for (const col of numericalColumns) {
        const values = data
            .map(row => Number(row[col]))
            .filter(val => typeof val === 'number' && !isNaN(val));
        
        const isInteger = values.length > 0 && values.every(v => Number.isInteger(v));
        const min = values.length > 0 ? values.reduce((m, v) => v < m ? v : m, values[0]) : 0;
        const max = values.length > 0 ? values.reduce((m, v) => v > m ? v : m, values[0]) : 0;
        
        stats[col] = {
            mean: values.length > 0 ? jStat.mean(values) : 0,
            std: values.length > 1 ? jStat.stdev(values) : 0,
            min,
            max,
            isInteger
        };
    }
    return stats;
};

const getNormalizedFeatureMatrix = (data: CsvData, numericalColumns: string[], stats: Record<string, { std: number; mean: number }>) => {
    return data.map(row => {
        const normalized: Record<string, number> = {};
        numericalColumns.forEach(col => {
            const val = row[col] as number;
            const { mean, std } = stats[col];
            normalized[col] = std > 0 && typeof val === 'number' ? (val - mean) / std : 0;
        });
        return normalized;
    });
};

const weightedRandomChoice = (options: Map<any, number>): any => {
    if (options.size === 0) return null;
    const entries = Array.from(options.entries());
    const totalWeight = entries.reduce((sum, [_, weight]) => sum + weight, 0);
    if (totalWeight === 0) return entries[0][0];

    let random = Math.random() * totalWeight;
    for (const [value, weight] of entries) {
        random -= weight;
        if (random <= 0) return value;
    }
    return entries[0][0];
};

const predictCategoricalValue = (
    syntheticNumericalFeatures: Record<string, number>,
    originalNumericalMatrix: Record<string, number>[],
    originalData: CsvData,
    targetCol: string,
    numericalColumns: string[],
    categoricalColumns: string[],
    stats: Record<string, { std: number; mean: number }>,
    seedRow: any
): string | number | null => {
    const normalizedSynthetic: Record<string, number> = {};
    numericalColumns.forEach(col => {
        const { mean, std } = stats[col];
        normalizedSynthetic[col] = std > 0 ? (syntheticNumericalFeatures[col] - mean) / std : 0;
    });

    const distances = originalNumericalMatrix.map((originalRow, idx) => {
        let distSq = 0;
        numericalColumns.forEach(col => {
            distSq += Math.pow((normalizedSynthetic[col] || 0) - (originalRow[col] || 0), 2);
        });

        categoricalColumns.forEach(col => {
            if (col !== targetCol && originalData[idx][col] !== seedRow[col]) {
                distSq += CATEGORICAL_MISMATCH_PENALTY;
            }
        });

        return { index: idx, dist: Math.sqrt(distSq) };
    });

    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, K_NEIGHBORS);

    const classCounts = new Map<any, number>();
    neighbors.forEach(n => {
        const val = originalData[n.index][targetCol];
        classCounts.set(val, (classCounts.get(val) || 0) + 1);
    });

    return weightedRandomChoice(classCounts);
};

export const augmentData = (
  originalData: CsvData, 
  targetSize: number,
  customColumnTypes?: Record<string, 'continuous' | 'numerical' | 'categorical' | 'ordinal'>
): CsvData => {
  if (originalData.length === 0) return [];

  const colInfo = getColumnTypes(originalData, customColumnTypes);
  const headers = Object.keys(originalData[0]);
  const numericalColumns = Object.keys(colInfo).filter(key => colInfo[key].type === 'continuous' || colInfo[key].type === 'numerical');
  const categoricalColumns = Object.keys(colInfo).filter(key => colInfo[key].type === 'categorical' || colInfo[key].type === 'ordinal');
  
  const stats = getColumnStats(originalData, numericalColumns);
  const normalizedOriginalMatrix = getNormalizedFeatureMatrix(originalData, numericalColumns, stats);
  
  const augmentedData: CsvData = [];
  const originalDataSize = originalData.length;

  for (let i = 0; i < targetSize; i++) {
    const baseSampleIndex = Math.floor(Math.random() * originalDataSize);
    const baseSample = originalData[baseSampleIndex];
    
    const syntheticRow: Record<string, any> = {};
    const syntheticNumerical: Record<string, number> = {};

    // First pass: Numerical columns
    for (const col of numericalColumns) {
      const { std, mean } = stats[col];
      let rawBase = baseSample[col];
      let baseValue: number = typeof rawBase === 'number' ? rawBase : Number(rawBase);
      const precision = colInfo[col].precision;
      
      // Impute missing numerical values with the column mean
      if (baseValue === null || baseValue === undefined || isNaN(baseValue)) {
          baseValue = mean;
      }
      
      if (std > 0) {
        const noise = jStat.normal.sample(0, std * NOISE_FACTOR);
        let syntheticVal = (baseValue as number) + noise;
        
        // Clamp to original range to maintain biological plausibility
        syntheticVal = Math.max(stats[col].min, Math.min(stats[col].max, syntheticVal));

        if (isNaN(syntheticVal)) {
            syntheticNumerical[col] = baseValue as number;
        } else {
            if (stats[col].isInteger) {
                // For purely discrete columns, maintain discreteness
                syntheticNumerical[col] = Math.round(syntheticVal);
            } else {
                // For continuous columns, maintain original precision
                const factor = Math.pow(10, precision);
                syntheticNumerical[col] = Math.round(syntheticVal * factor) / factor;
            }
        }
      } else {
        syntheticNumerical[col] = baseValue as number;
      }
      syntheticRow[col] = syntheticNumerical[col];
    }

    // Second pass: Categorical columns
    for (const col of categoricalColumns) {
        if (numericalColumns.length > 0 || categoricalColumns.length > 1) {
            syntheticRow[col] = predictCategoricalValue(
                syntheticNumerical,
                normalizedOriginalMatrix,
                originalData,
                col,
                numericalColumns,
                categoricalColumns,
                stats,
                baseSample
            );
        } else {
            syntheticRow[col] = baseSample[col];
        }
    }

    // Final pass: Construct row in original header order
    const orderedRow: Record<string, any> = {};
    for (const header of headers) {
        if (header in syntheticRow) {
            orderedRow[header] = syntheticRow[header];
        } else {
            orderedRow[header] = baseSample[header];
        }
    }

    augmentedData.push(orderedRow);
  }

  return augmentedData;
};
