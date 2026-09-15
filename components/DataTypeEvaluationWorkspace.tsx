import React, { useState, useMemo } from 'react';
import { CsvData } from '../types';
import { getDetailedColumnMetadata } from '../services/statisticalAnalysis';
import { 
    CheckCircle2, 
    AlertCircle, 
    SlidersHorizontal, 
    RotateCcw, 
    Play, 
    Table, 
    HelpCircle, 
    Search,
    Tag,
    Layers,
    Sparkles,
    Eye
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface DataTypeEvaluationWorkspaceProps {
    originalData: CsvData;
    fileName?: string;
    customColumnTypes: Record<string, 'continuous' | 'categorical' | 'ordinal'>;
    onColumnTypeChange: (col: string, newType: 'continuous' | 'categorical' | 'ordinal') => void;
    onResetColumnTypes: () => void;
    onAugment: () => void;
    targetSize: number;
    isLoading?: boolean;
    isPostSynthesis?: boolean;
}

export const DataTypeEvaluationWorkspace: React.FC<DataTypeEvaluationWorkspaceProps> = ({
    originalData,
    fileName,
    customColumnTypes,
    onColumnTypeChange,
    onResetColumnTypes,
    onAugment,
    targetSize,
    isLoading = false,
    isPostSynthesis = false,
}) => {
    const [viewMode, setViewMode] = useState<'types' | 'raw'>('types');
    const [searchTerm, setSearchTerm] = useState('');
    const [filterType, setFilterType] = useState<'all' | 'continuous' | 'categorical' | 'ordinal'>('all');

    // Calculate metadata for each column
    const columnMetadata = useMemo(() => {
        return getDetailedColumnMetadata(originalData, customColumnTypes);
    }, [originalData, customColumnTypes]);

    const headers = useMemo(() => Object.keys(columnMetadata), [columnMetadata]);

    const counts = useMemo(() => {
        let continuous = 0;
        let categorical = 0;
        let ordinal = 0;
        let overridden = 0;

        headers.forEach(h => {
            const meta = columnMetadata[h];
            const current = meta.type === 'numerical' ? 'continuous' : meta.type;
            if (current === 'continuous') continuous++;
            else if (current === 'categorical') categorical++;
            else if (current === 'ordinal') ordinal++;

            const detected = meta.detectedType === 'numerical' ? 'continuous' : meta.detectedType;
            if (current !== detected) {
                overridden++;
            }
        });

        return { continuous, categorical, ordinal, overridden, total: headers.length };
    }, [headers, columnMetadata]);

    const filteredHeaders = useMemo(() => {
        return headers.filter(h => {
            const matchesSearch = h.toLowerCase().includes(searchTerm.toLowerCase());
            if (!matchesSearch) return false;
            if (filterType === 'all') return true;
            const current = columnMetadata[h]?.type === 'numerical' ? 'continuous' : columnMetadata[h]?.type;
            return current === filterType;
        });
    }, [headers, searchTerm, filterType, columnMetadata]);

    const rawPreviewRows = useMemo(() => {
        return originalData.slice(0, 10);
    }, [originalData]);

    return (
        <div className="flex flex-col h-full space-y-6">
            {/* Header section */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                <div>
                    <div className="flex items-center gap-2.5 mb-1.5">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-blue-50 text-blue-700 border border-blue-200/70 text-[10px] font-black uppercase tracking-wider">
                            <SlidersHorizontal className="w-3 h-3 text-blue-600" />
                            {isPostSynthesis ? 'Variable Configuration' : 'Early Evaluation Step'}
                        </span>
                        {fileName && (
                            <span className="text-xs font-semibold text-slate-400">
                                {fileName} ({originalData.length} records)
                            </span>
                        )}
                    </div>
                    <h2 className="text-xl font-black text-slate-900 tracking-tight">
                        Evaluate & Validate Data Types
                    </h2>
                    <p className="text-xs text-slate-500 font-medium max-w-3xl mt-1 leading-relaxed">
                        Automatic heuristic detection may mislabel variables in small datasets. 
                        Inspect auto-detected classifications below and use the dropdown menu to correct any misclassifications (e.g. changing <strong>Continuous</strong> to <strong>Categorical</strong> or <strong>Ordinal</strong>) before statistical synthesis.
                    </p>
                </div>

                {/* Mode toggle */}
                <div className="flex items-center bg-slate-100/80 p-1 rounded-xl shrink-0 self-start sm:self-auto border border-slate-200/50">
                    <button
                        onClick={() => setViewMode('types')}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                            viewMode === 'types'
                                ? 'bg-white text-blue-600 shadow-xs'
                                : 'text-slate-500 hover:text-slate-800'
                        }`}
                    >
                        <Tag className="w-3.5 h-3.5" />
                        Variables ({headers.length})
                    </button>
                    <button
                        onClick={() => setViewMode('raw')}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                            viewMode === 'raw'
                                ? 'bg-white text-blue-600 shadow-xs'
                                : 'text-slate-500 hover:text-slate-800'
                        }`}
                    >
                        <Eye className="w-3.5 h-3.5" />
                        Raw Data (First 10)
                    </button>
                </div>
            </div>

            {/* Quick Summary Pill Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                <div className="p-3.5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs">
                    <span className="text-[10px] font-black uppercase tracking-wider text-slate-400 block mb-1">Total Variables</span>
                    <span className="text-xl font-black text-slate-800">{counts.total}</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-blue-50/50 border border-blue-100 shadow-2xs">
                    <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-black uppercase tracking-wider text-blue-600">Continuous</span>
                        <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    </div>
                    <span className="text-xl font-black text-blue-700">{counts.continuous}</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-purple-50/50 border border-purple-100 shadow-2xs">
                    <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-black uppercase tracking-wider text-purple-600">Categorical</span>
                        <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                    </div>
                    <span className="text-xl font-black text-purple-700">{counts.categorical}</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-amber-50/50 border border-amber-100 shadow-2xs">
                    <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-black uppercase tracking-wider text-amber-600">Ordinal</span>
                        <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                    </div>
                    <span className="text-xl font-black text-amber-700">{counts.ordinal}</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 shadow-2xs col-span-2 sm:col-span-1">
                    <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-black uppercase tracking-wider text-slate-500">Overrides</span>
                        {counts.overridden > 0 && <span className="w-2 h-2 rounded-full bg-emerald-500"></span>}
                    </div>
                    <span className={`text-xl font-black ${counts.overridden > 0 ? 'text-emerald-600' : 'text-slate-400'}`}>
                        {counts.overridden} {counts.overridden === 1 ? 'change' : 'changes'}
                    </span>
                </div>
            </div>

            {/* Main Content Area */}
            {viewMode === 'types' ? (
                <div className="flex-1 flex flex-col space-y-4 min-h-0">
                    {/* Controls row: Search & Filter */}
                    <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
                        <div className="relative flex-1 max-w-md">
                            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                            <input
                                type="text"
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                placeholder="Search variables by name..."
                                className="w-full pl-10 pr-4 py-2 text-xs bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all font-medium text-slate-800 placeholder:text-slate-400"
                            />
                        </div>

                        <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0">
                            <span className="text-[11px] font-bold text-slate-400 shrink-0">Filter:</span>
                            {(['all', 'continuous', 'categorical', 'ordinal'] as const).map((type) => (
                                <button
                                    key={type}
                                    onClick={() => setFilterType(type)}
                                    className={`px-3 py-1.5 rounded-lg text-xs font-bold capitalize transition-all shrink-0 ${
                                        filterType === type
                                            ? 'bg-slate-900 text-white shadow-xs'
                                            : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
                                    }`}
                                >
                                    {type}
                                </button>
                            ))}
                            {counts.overridden > 0 && (
                                <button
                                    onClick={onResetColumnTypes}
                                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-bold text-slate-600 hover:text-red-600 hover:bg-red-50 border border-slate-200 transition-all shrink-0 ml-auto"
                                    title="Reset all variables to heuristic defaults"
                                >
                                    <RotateCcw className="w-3 h-3" />
                                    Reset Defaults
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Table */}
                    <div className="flex-1 overflow-auto rounded-2xl border border-slate-200 bg-white shadow-xs custom-scrollbar">
                        <table className="w-full text-left text-xs border-collapse">
                            <thead>
                                <tr className="bg-slate-50/80 border-b border-slate-200 text-[10px] text-slate-500 uppercase font-black tracking-wider sticky top-0 z-10 backdrop-blur-xs">
                                    <th className="px-5 py-3.5">Variable Name</th>
                                    <th className="px-5 py-3.5">Auto-Detected</th>
                                    <th className="px-5 py-3.5">Distinct Samples</th>
                                    <th className="px-5 py-3.5 w-52">Validated Data Type (Dropdown)</th>
                                    <th className="px-5 py-3.5">Synthesis Method</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {filteredHeaders.length === 0 ? (
                                    <tr>
                                        <td colSpan={5} className="px-6 py-12 text-center text-slate-400">
                                            No variables match your search or filter.
                                        </td>
                                    </tr>
                                ) : (
                                    filteredHeaders.map((col) => {
                                        const meta = columnMetadata[col];
                                        const currentType = (meta?.type === 'numerical' ? 'continuous' : meta?.type) || 'continuous';
                                        const detectedType = (meta?.detectedType === 'numerical' ? 'continuous' : meta?.detectedType) || 'continuous';
                                        const isOverridden = currentType !== detectedType;

                                        return (
                                            <tr 
                                                key={col} 
                                                className={`transition-colors hover:bg-slate-50/60 ${
                                                    isOverridden ? 'bg-amber-50/20' : ''
                                                }`}
                                            >
                                                {/* Variable name */}
                                                <td className="px-5 py-4 font-bold text-slate-800">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-slate-900 font-extrabold text-sm">{col}</span>
                                                        {isOverridden && (
                                                            <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-[9px] font-black uppercase tracking-wider">
                                                                Overridden
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="text-[11px] text-slate-400 font-medium mt-0.5">
                                                        {meta.uniqueCount} distinct {meta.uniqueCount === 1 ? 'value' : 'values'}
                                                        {meta.min !== undefined && meta.max !== undefined && (
                                                            <span> • Range: [{meta.min}, {meta.max}]</span>
                                                        )}
                                                    </div>
                                                </td>

                                                {/* Detected type info */}
                                                <td className="px-5 py-4">
                                                    <div className="flex items-center gap-1.5">
                                                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider border ${
                                                            detectedType === 'continuous'
                                                                ? 'bg-blue-50 text-blue-700 border-blue-200'
                                                                : 'bg-purple-50 text-purple-700 border-purple-200'
                                                        }`}>
                                                            {detectedType}
                                                        </span>
                                                    </div>
                                                    <span className="text-[10px] text-slate-400 font-medium block mt-1">
                                                        {meta.isNumeric ? 'Numeric values' : 'Non-numeric strings'}
                                                    </span>
                                                </td>

                                                {/* Sample preview chips */}
                                                <td className="px-5 py-4">
                                                    <div className="flex flex-wrap gap-1 max-w-xs">
                                                        {meta.sampleValues && meta.sampleValues.length > 0 ? (
                                                            meta.sampleValues.map((val, idx) => (
                                                                <span 
                                                                    key={idx} 
                                                                    className="px-1.5 py-0.5 bg-slate-100 text-slate-700 rounded font-mono text-[10px] max-w-[90px] truncate"
                                                                    title={String(val)}
                                                                >
                                                                    {String(val)}
                                                                </span>
                                                            ))
                                                        ) : (
                                                            <span className="text-slate-400 italic text-[10px]">No samples</span>
                                                        )}
                                                    </div>
                                                </td>

                                                {/* Dropdown menu */}
                                                <td className="px-5 py-4">
                                                    <div className="flex flex-col gap-1">
                                                        <select
                                                            value={currentType}
                                                            onChange={(e) => onColumnTypeChange(col, e.target.value as 'continuous' | 'categorical' | 'ordinal')}
                                                            className={`text-xs font-black rounded-xl px-3 py-2 border outline-none cursor-pointer transition-all shadow-2xs ${
                                                                currentType === 'continuous'
                                                                    ? 'bg-blue-50 border-blue-300 text-blue-800 focus:ring-2 focus:ring-blue-500/20'
                                                                    : currentType === 'ordinal'
                                                                    ? 'bg-amber-50 border-amber-300 text-amber-800 focus:ring-2 focus:ring-amber-500/20'
                                                                    : 'bg-purple-50 border-purple-300 text-purple-800 focus:ring-2 focus:ring-purple-500/20'
                                                            }`}
                                                        >
                                                            <option value="continuous">Continuous</option>
                                                            <option value="categorical">Categorical</option>
                                                            <option value="ordinal">Ordinal</option>
                                                        </select>
                                                        {isOverridden && (
                                                            <button
                                                                onClick={() => onColumnTypeChange(col, detectedType as any)}
                                                                className="text-[10px] text-slate-400 hover:text-slate-600 text-left underline pl-0.5"
                                                            >
                                                                Reset to {detectedType}
                                                            </button>
                                                        )}
                                                    </div>
                                                </td>

                                                {/* Synthesis explanation */}
                                                <td className="px-5 py-4 text-[11px] text-slate-500 font-medium">
                                                    {currentType === 'continuous' ? (
                                                        <div className="flex items-start gap-1.5 text-blue-900/80">
                                                            <Sparkles className="w-3.5 h-3.5 text-blue-500 shrink-0 mt-0.5" />
                                                            <span>Gaussian KDE bootstrap with adaptive noise injection</span>
                                                        </div>
                                                    ) : currentType === 'ordinal' ? (
                                                        <div className="flex items-start gap-1.5 text-amber-900/80">
                                                            <Layers className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                                                            <span>Rank-preserved ordered categorical sampling</span>
                                                        </div>
                                                    ) : (
                                                        <div className="flex items-start gap-1.5 text-purple-900/80">
                                                            <CheckCircle2 className="w-3.5 h-3.5 text-purple-500 shrink-0 mt-0.5" />
                                                            <span>Empirical probability mass discrete resampling</span>
                                                        </div>
                                                    )}
                                                </td>
                                            </tr>
                                        );
                                    })
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            ) : (
                /* Raw Data Table View */
                <div className="flex-1 overflow-auto rounded-2xl border border-slate-200 bg-white shadow-xs custom-scrollbar">
                    <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-600">
                            Showing first {Math.min(10, originalData.length)} rows of {originalData.length} records in uploaded CSV
                        </span>
                        <span className="text-xs text-slate-400 font-medium">
                            Verify data values match your expected variable classifications
                        </span>
                    </div>
                    <table className="w-full text-left text-xs border-collapse">
                        <thead>
                            <tr className="bg-slate-50/90 border-b border-slate-200 text-[10px] text-slate-500 uppercase font-black tracking-wider">
                                <th className="px-4 py-3 bg-slate-100 text-slate-400 w-12 text-center">#</th>
                                {headers.map((h) => (
                                    <th key={h} className="px-4 py-3 whitespace-nowrap">
                                        <div className="font-extrabold text-slate-800">{h}</div>
                                        <div className="text-[9px] font-bold text-blue-600 capitalize mt-0.5">
                                            {customColumnTypes[h] || columnMetadata[h]?.type || 'auto'}
                                        </div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                            {rawPreviewRows.map((row, rowIdx) => (
                                <tr key={rowIdx} className="hover:bg-slate-50/60">
                                    <td className="px-4 py-2.5 text-center text-slate-400 bg-slate-50/50 font-bold text-[10px]">
                                        {rowIdx + 1}
                                    </td>
                                    {headers.map((h) => (
                                        <td key={h} className="px-4 py-2.5 text-slate-700 whitespace-nowrap">
                                            {row[h] !== null && row[h] !== undefined ? String(row[h]) : (
                                                <span className="text-slate-300 italic font-sans text-[10px]">null</span>
                                            )}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Bottom Action Footer */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-100 bg-white/50 backdrop-blur-xs">
                <div className="flex items-center gap-2 text-xs text-slate-500 font-medium">
                    <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                    <span>
                        Validation complete: <strong>{counts.continuous}</strong> continuous, <strong>{counts.categorical}</strong> categorical, <strong>{counts.ordinal}</strong> ordinal variables configured.
                    </span>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                    {counts.overridden > 0 && (
                        <button
                            onClick={onResetColumnTypes}
                            className="btn-secondary text-xs py-2.5 px-4"
                            disabled={isLoading}
                        >
                            <RotateCcw className="w-3.5 h-3.5" />
                            Reset Defaults
                        </button>
                    )}
                    <button
                        onClick={onAugment}
                        disabled={isLoading}
                        className="btn-primary text-xs py-2.5 px-6 flex-1 sm:flex-initial shadow-md shadow-blue-500/20"
                    >
                        <Play className="w-3.5 h-3.5 fill-current" />
                        {isLoading 
                            ? 'Synthesizing Data...' 
                            : isPostSynthesis 
                            ? `Re-synthesize (${targetSize.toLocaleString()} records)`
                            : `Confirm Types & Generate Synthetic Data (${targetSize.toLocaleString()} records)`}
                    </button>
                </div>
            </div>
        </div>
    );
};
