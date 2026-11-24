
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

// Fast Box-Muller standard normal sampler N(0, 1)
const sampleStandardNormal = (): number => {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
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
  const originalDataSize = originalData.length;
  const numNum = numericalColumns.length;
  const numCat = categoricalColumns.length;

  // Precompute numerical column parameters for fast indexed access
  const numMeans = new Float64Array(numNum);
  const numStds = new Float64Array(numNum);
  const numInvStds = new Float64Array(numNum);
  const numMins = new Float64Array(numNum);
  const numMaxs = new Float64Array(numNum);
  const numFactors = new Float64Array(numNum);
  const numIsInt = new Uint8Array(numNum);

  for (let j = 0; j < numNum; j++) {
      const col = numericalColumns[j];
      const s = stats[col];
      numMeans[j] = s.mean;
      numStds[j] = s.std;
      numInvStds[j] = s.std > 0 ? 1 / s.std : 0;
      numMins[j] = s.min;
      numMaxs[j] = s.max;
      numFactors[j] = Math.pow(10, colInfo[col].precision);
      numIsInt[j] = s.isInteger ? 1 : 0;
  }

  // Flat typed matrix of normalized numerical features: row * numNum + col
  const normOrigMatrix = new Float64Array(originalDataSize * numNum);
  const rawOrigNumMatrix = new Float64Array(originalDataSize * numNum);

  for (let idx = 0; idx < originalDataSize; idx++) {
      const row = originalData[idx];
      const rowOffset = idx * numNum;
      for (let j = 0; j < numNum; j++) {
          const raw = row[numericalColumns[j]];
          const val = (raw === null || raw === undefined || raw === '')
              ? NaN
              : (typeof raw === 'number' ? raw : Number(raw));
          if (!isNaN(val)) {
              rawOrigNumMatrix[rowOffset + j] = val;
              normOrigMatrix[rowOffset + j] = numStds[j] > 0 ? (val - numMeans[j]) * numInvStds[j] : 0;
          } else {
              rawOrigNumMatrix[rowOffset + j] = numMeans[j];
              normOrigMatrix[rowOffset + j] = 0;
          }
      }
  }

  // Integer-encode categorical columns into a flat Int32Array: row * numCat + col
  const catMatrix = new Int32Array(originalDataSize * numCat);
  const catDecoders: any[][] = new Array(numCat);

  for (let c = 0; c < numCat; c++) {
      const col = categoricalColumns[c];
      const encoder = new Map<any, number>();
      const decoder: any[] = [];
      for (let idx = 0; idx < originalDataSize; idx++) {
          const val = originalData[idx][col];
          let code = encoder.get(val);
          if (code === undefined) {
              code = decoder.length;
              encoder.set(val, code);
              decoder.push(val);
          }
          catMatrix[idx * numCat + c] = code;
      }
      catDecoders[c] = decoder;
  }

  const useKNN = numCat > 0 && (numNum > 0 || numCat > 1);
  const k = Math.min(K_NEIGHBORS, originalDataSize);

  // Reusable scratch buffers to eliminate per-row heap allocations
  const synthNorm = new Float64Array(numNum);
  const baseDistSq = useKNN ? new Float64Array(originalDataSize) : null;
  const topKDist = new Float64Array(k);
  const topKIdx = new Int32Array(k);

  const augmentedData: CsvData = new Array(targetSize);

  for (let i = 0; i < targetSize; i++) {
    const baseSampleIndex = Math.floor(Math.random() * originalDataSize);
    const baseSample = originalData[baseSampleIndex];
    const baseNumOffset = baseSampleIndex * numNum;
    const baseCatOffset = baseSampleIndex * numCat;

    const syntheticRow: Record<string, any> = {};

    // First pass: Numerical columns
    for (let j = 0; j < numNum; j++) {
      const col = numericalColumns[j];
      const std = numStds[j];
      const baseValue = rawOrigNumMatrix[baseNumOffset + j];
      let finalVal = baseValue;

      if (std > 0) {
        const noise = sampleStandardNormal() * std * NOISE_FACTOR;
        let syntheticVal = baseValue + noise;

        // Clamp to original range to maintain biological plausibility
        if (syntheticVal < numMins[j]) syntheticVal = numMins[j];
        else if (syntheticVal > numMaxs[j]) syntheticVal = numMaxs[j];

        if (!isNaN(syntheticVal)) {
            if (numIsInt[j]) {
                finalVal = Math.round(syntheticVal);
            } else {
                const factor = numFactors[j];
                finalVal = Math.round(syntheticVal * factor) / factor;
            }
        }
      }

      syntheticRow[col] = finalVal;
      synthNorm[j] = std > 0 ? (finalVal - numMeans[j]) * numInvStds[j] : 0;
    }

    // Second pass: Categorical & Ordinal columns via shared-distance O(N) top-K k-NN
    if (useKNN && baseDistSq) {
        // Compute total distance squared (all numerical dims + all categorical dims) ONCE per synthetic row
        for (let idx = 0; idx < originalDataSize; idx++) {
            let dSq = 0;
            const nOff = idx * numNum;
            for (let j = 0; j < numNum; j++) {
                const diff = synthNorm[j] - normOrigMatrix[nOff + j];
                dSq += diff * diff;
            }
            const cOff = idx * numCat;
            for (let c = 0; c < numCat; c++) {
                if (catMatrix[cOff + c] !== catMatrix[baseCatOffset + c]) {
                    dSq += CATEGORICAL_MISMATCH_PENALTY;
                }
            }
            baseDistSq[idx] = dSq;
        }

        // For each target categorical column, exclude its own mismatch penalty and pick top-k neighbors
        for (let c = 0; c < numCat; c++) {
            const seedCode = catMatrix[baseCatOffset + c];
            topKDist.fill(Infinity);
            let worstDist = Infinity;

            for (let idx = 0; idx < originalDataSize; idx++) {
                let dSq = baseDistSq[idx];
                if (catMatrix[idx * numCat + c] !== seedCode) {
                    dSq -= CATEGORICAL_MISMATCH_PENALTY;
                }

                if (dSq < worstDist) {
                    // Insertion into sorted top-k buffer (preserves lower index on ties)
                    let pos = k - 1;
                    while (pos > 0 && dSq < topKDist[pos - 1]) {
                        topKDist[pos] = topKDist[pos - 1];
                        topKIdx[pos] = topKIdx[pos - 1];
                        pos--;
                    }
                    topKDist[pos] = dSq;
                    topKIdx[pos] = idx;
                    worstDist = topKDist[k - 1];
                }
            }

            // Weighted random choice over the k nearest neighbors is equivalent to uniform pick among the k neighbors!
            const pickedNeighborIdx = topKIdx[Math.floor(Math.random() * k)];
            const pickedCode = catMatrix[pickedNeighborIdx * numCat + c];
            syntheticRow[categoricalColumns[c]] = catDecoders[c][pickedCode];
        }
    } else {
        for (let c = 0; c < numCat; c++) {
            const col = categoricalColumns[c];
            syntheticRow[col] = baseSample[col];
        }
    }

    // Final pass: Construct row in original header order
    const orderedRow: Record<string, any> = {};
    for (let h = 0; h < headers.length; h++) {
        const header = headers[h];
        orderedRow[header] = header in syntheticRow ? syntheticRow[header] : baseSample[header];
    }

    augmentedData[i] = orderedRow;
  }

  return augmentedData;
};
