
export type CsvData = Record<string, number | string>[];

export type DataType = 'continuous' | 'categorical' | 'ordinal' | 'numerical';

export interface ColumnMetadata {
  type: 'continuous' | 'categorical' | 'ordinal' | 'numerical';
  uniqueCount: number;
  isHeuristicCategorical: boolean; // True if numeric but treated as categorical due to low cardinality
  precision: number; // Number of decimal places to preserve
  detectedType?: 'continuous' | 'categorical' | 'ordinal';
  sampleValues?: (string | number)[];
  isNumeric?: boolean;
  min?: number;
  max?: number;
}

export interface ReportSummary {
  tTestSimilarCount: number;
  mannWhitneySimilarCount: number;
  ksTestSimilarCount: number;
  chiSquareSimilarCount: number;
  avgTVD: number;
  avgCramersV: number;
  totalNumerical: number;
  totalContinuous?: number;
  totalCategorical: number;
  totalOrdinal?: number;
}

export interface Report {
  descriptiveStats: DescriptiveStatReport[];
  tTest: StatisticalTestResult[];
  mannWhitney: StatisticalTestResult[];
  ksTest: StatisticalTestResult[];
  chiSquare: StatisticalTestResult[];
  totalVariationDistance: { column: string; value: number; isSimilar: boolean }[];
  columnTypes: Record<string, 'continuous' | 'categorical' | 'ordinal' | 'numerical'>;
  columnMetadata: Record<string, ColumnMetadata>;
  summary: ReportSummary;
}

export interface DescriptiveStatReport {
  column: string;
  original: Stats;
  augmented: Stats;
  meanDifferenceCI: {
    diff: number;
    lower: number;
    upper: number;
  };
}

export interface Stats {
  mean: number;
  std: number;
  min: number;
  max: number;
  count: number;
}

export interface StatisticalTestResult {
  column: string;
  pValue: number;
  statistic: number;
  cramersV?: number;
  isSimilar: boolean;
  uniqueCount?: number;
}

export enum ReportTab {
    SUMMARY = 'Summary',
    DATA_TYPES = 'Data Types',
    HISTOGRAMS = 'Histograms',
    T_TEST = 'T-Test',
    MANN_WHITNEY = 'Mann-Whitney',
    KS_TEST = 'KS Test',
    CATEGORICAL = 'Categorical Tests',
}
