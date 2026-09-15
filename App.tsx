
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import Papa from 'papaparse';
import { CsvData, Report, ReportTab, DescriptiveStatReport, StatisticalTestResult } from './types';
import { augmentData } from './services/dataAugmentation';
import { generateReport, formatReportAsText, generateStatsSummaryCsv, getDetailedColumnMetadata } from './services/statisticalAnalysis';
import { GoogleGenAI, Chat } from '@google/genai';
import { 
    Upload, 
    Play, 
    RotateCcw, 
    Download, 
    FileText, 
    BarChart3, 
    MessageSquare, 
    X, 
    Send,
    ChevronRight,
    Activity,
    Database,
    CheckCircle2,
    AlertCircle,
    Info,
    SlidersHorizontal
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { DataTypeEvaluationWorkspace } from './components/DataTypeEvaluationWorkspace';

const formatPValue = (pValue: number): string => {
    if (pValue < 0.001) return '< 0.001';
    return pValue.toFixed(3);
};

const getCramersVColor = (v: number): string => {
    if (v < 0.1) return 'text-green-600';
    if (v < 0.3) return 'text-yellow-600';
    return 'text-red-600';
};

interface ChatMessage {
    role: 'user' | 'model';
    text: string;
}

const App: React.FC = () => {
    const [originalData, setOriginalData] = useState<CsvData | null>(null);
    const [augmentedData, setAugmentedData] = useState<CsvData | null>(null);
    const [customColumnTypes, setCustomColumnTypes] = useState<Record<string, 'continuous' | 'categorical' | 'ordinal'>>({});
    const [fileName, setFileName] = useState<string>('');
    const [targetSize, setTargetSize] = useState<number>(1000);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');
    const [report, setReport] = useState<Report | null>(null);
    const [isChatOpen, setIsChatOpen] = useState<boolean>(false);
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
    const chatInstance = useRef<Chat | null>(null);

    useEffect(() => {
        if(isChatOpen && !chatInstance.current) {
            try {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY! });
                chatInstance.current = ai.chats.create({
                  model: 'gemini-3-flash-preview',
                   config: {
                    systemInstruction: `You are an expert statistical assistant for BootsGN. Help users interpret augmentation results (t-test, Mann-Whitney, KS test, chi-square, Total Variation Distance, Cramer's V). Be concise.`,
                  },
                });
            } catch (e) {
                 setIsChatOpen(false);
            }
        }
    }, [isChatOpen]);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setError(''); setOriginalData(null); setReport(null); setAugmentedData(null); setFileName(file.name);
            Papa.parse(file, {
                header: true, dynamicTyping: true, skipEmptyLines: true,
                complete: (results: any) => {
                    if (results.errors.length > 0) { setError(`Error: ${results.errors[0].message}`); return; }
                    
                    // Clean data: treat empty / standard null strings as null
                    const cleanedData = results.data.map((row: any) => {
                        const newRow = { ...row };
                        for (const key in newRow) {
                            const val = newRow[key];
                            if (typeof val === 'string') {
                                const lower = val.toLowerCase().trim();
                                if (lower === 'null' || lower === 'nan' || lower === 'n/a' || lower === 'na' || lower === 'undefined' || lower === '') {
                                    newRow[key] = null;
                                }
                            }
                        }
                        return newRow;
                    });
                    
                    if (cleanedData.length > 0) {
                        const meta = getDetailedColumnMetadata(cleanedData);
                        const initialTypes: Record<string, 'continuous' | 'categorical' | 'ordinal'> = {};
                        Object.keys(meta).forEach(col => {
                            const t = meta[col].type;
                            initialTypes[col] = (t === 'numerical' ? 'continuous' : t) as 'continuous' | 'categorical' | 'ordinal';
                        });
                        setCustomColumnTypes(initialTypes);
                    } else {
                        setCustomColumnTypes({});
                    }

                    setOriginalData(cleanedData);
                },
            });
        }
    };

    const handleReset = useCallback(() => {
        setOriginalData(null);
        setAugmentedData(null);
        setCustomColumnTypes({});
        setFileName('');
        setReport(null);
        setError('');
        setChatMessages([]);
    }, []);

    const handleColumnTypeChange = useCallback((columnName: string, newType: 'continuous' | 'categorical' | 'ordinal') => {
        setCustomColumnTypes(prev => {
            const next = { ...prev, [columnName]: newType };
            if (originalData) {
                try {
                    if (augmentedData) {
                        const newAug = augmentData(originalData, targetSize, next);
                        setAugmentedData(newAug);
                        setReport(generateReport(originalData, newAug, next));
                    }
                } catch (e) {
                    console.error("Error updating augmented data with new column types:", e);
                }
            }
            return next;
        });
    }, [originalData, augmentedData, targetSize]);

    const handleResetColumnTypes = useCallback(() => {
        if (!originalData) return;
        const meta = getDetailedColumnMetadata(originalData);
        const initialTypes: Record<string, 'continuous' | 'categorical' | 'ordinal'> = {};
        Object.keys(meta).forEach(col => {
            const t = meta[col].type;
            initialTypes[col] = (t === 'numerical' ? 'continuous' : t) as 'continuous' | 'categorical' | 'ordinal';
        });
        setCustomColumnTypes(initialTypes);
        if (augmentedData) {
            const newAug = augmentData(originalData, targetSize, initialTypes);
            setAugmentedData(newAug);
            setReport(generateReport(originalData, newAug, initialTypes));
        }
    }, [originalData, augmentedData, targetSize]);

    const handleAugmentData = useCallback(() => {
        if (!originalData) return;
        setIsLoading(true); setError(''); setReport(null); setAugmentedData(null);
        setTimeout(() => {
            try {
                const augmented = augmentData(originalData, targetSize, customColumnTypes);
                setAugmentedData(augmented);
                setReport(generateReport(originalData, augmented, customColumnTypes));
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Error during augmentation.');
            } finally { setIsLoading(false); }
        }, 50);
    }, [originalData, targetSize, customColumnTypes]);

    const handleDownload = () => {
        if (!augmentedData) return;
        const csv = Papa.unparse(augmentedData);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url; link.download = `augmented_${fileName}`; link.click();
    };

    const handleDownloadReport = () => {
        if (!report) return;
        const text = formatReportAsText(report);
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url; link.download = `report_${fileName.replace('.csv','')}.txt`; link.click();
    };

    const handleDownloadStatsSummary = () => {
        if (!report) return;
        const csv = generateStatsSummaryCsv(report);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url; link.download = `stats_summary_${fileName.replace('.csv','')}.csv`; link.click();
    };

    const handleSendMessage = async (message: string) => {
        if (!message.trim() || !chatInstance.current) return;
        setChatMessages(prev => [...prev, { role: 'user', text: message }]);
        setIsChatLoading(true);
        try {
            const res = await chatInstance.current.sendMessage({ message });
            setChatMessages(prev => [...prev, { role: 'model', text: res.text }]);
        } catch (e) {
            setChatMessages(prev => [...prev, { role: 'model', text: 'Error communicating with AI.' }]);
        } finally { setIsChatLoading(false); }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-4 md:p-8 selection:bg-blue-100">
            <div className="w-full max-w-7xl">
                <motion.header 
                    initial={{ opacity: 0, y: -30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="text-center mb-12 relative"
                >
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-blue-400/10 blur-[100px] rounded-full -z-10" />
                    <div className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-[2rem] mb-6 shadow-2xl shadow-blue-200 animate-float">
                        <Activity className="text-white h-10 w-10" />
                    </div>
                    <h1 className="text-6xl font-[900] text-slate-900 tracking-tight leading-none">
                        Bootstrap<span className="text-blue-600">MD</span>
                    </h1>
                    <p className="text-slate-500 mt-4 font-semibold text-lg max-w-2xl mx-auto">
                        High-fidelity medical data synthesis and rigorous statistical validation for clinical research.
                    </p>
                </motion.header>

                <main className="glass-card overflow-hidden relative">
                    <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
                        <Database className="h-32 w-32" />
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-slate-100/50">
                        <div className="lg:col-span-4 p-8 lg:p-10 bg-slate-50/30">
                            <ControlsSection
                                fileName={fileName} onFileChange={handleFileChange} 
                                targetSize={targetSize} onTargetSizeChange={setTargetSize}
                                onAugment={handleAugmentData} isDataLoaded={!!originalData} isLoading={isLoading}
                                onReset={handleReset}
                                originalData={originalData}
                                customColumnTypes={customColumnTypes}
                                onColumnTypeChange={handleColumnTypeChange}
                                onResetColumnTypes={handleResetColumnTypes}
                            />
                        </div>
                        
                        <div className="lg:col-span-8 p-8 lg:p-10 flex flex-col min-h-[600px] bg-white/40">
                            <div className="flex justify-between items-center mb-6">
                                <div className="flex items-center gap-2">
                                    <div className="p-1.5 bg-slate-100 rounded-lg">
                                        <BarChart3 className="h-4 w-4 text-slate-600" />
                                    </div>
                                    <h3 className="text-lg font-bold text-slate-800 tracking-tight">Analysis Workspace</h3>
                                </div>
                                {fileName && (
                                    <button 
                                        onClick={handleReset} 
                                        className="text-xs font-bold text-slate-400 hover:text-red-500 transition-colors uppercase tracking-widest flex items-center gap-1"
                                    >
                                        <RotateCcw className="h-3 w-3" />
                                        Reset
                                    </button>
                                )}
                            </div>

                            <div className="flex-1 flex flex-col relative">
                                <AnimatePresence mode="wait">
                                    {isLoading ? (
                                        <motion.div 
                                            key="loading"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                            className="absolute inset-0 flex items-center justify-center"
                                        >
                                            <LoadingSpinner />
                                        </motion.div>
                                    ) : error ? (
                                        <motion.div 
                                            key="error"
                                            initial={{ opacity: 0, scale: 0.95 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            className="absolute inset-0 flex items-center justify-center"
                                        >
                                            <ErrorMessage message={error} />
                                        </motion.div>
                                    ) : report ? (
                                        <motion.div 
                                            key="report"
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className="h-full"
                                        >
                                            <ReportDisplay 
                                                report={report} 
                                                onDownload={handleDownload} 
                                                onDownloadReport={handleDownloadReport} 
                                                onDownloadStats={handleDownloadStatsSummary} 
                                                originalData={originalData!} 
                                                augmentedData={augmentedData!}
                                                customColumnTypes={customColumnTypes}
                                                onColumnTypeChange={handleColumnTypeChange}
                                                onResetColumnTypes={handleResetColumnTypes}
                                                onAugment={handleAugmentData}
                                                targetSize={targetSize}
                                                isLoading={isLoading}
                                            />
                                        </motion.div>
                                    ) : originalData ? (
                                        <motion.div 
                                            key="evaluation"
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className="h-full flex flex-col"
                                        >
                                            <DataTypeEvaluationWorkspace
                                                originalData={originalData}
                                                fileName={fileName}
                                                customColumnTypes={customColumnTypes}
                                                onColumnTypeChange={handleColumnTypeChange}
                                                onResetColumnTypes={handleResetColumnTypes}
                                                onAugment={handleAugmentData}
                                                targetSize={targetSize}
                                                isLoading={isLoading}
                                                isPostSynthesis={false}
                                            />
                                        </motion.div>
                                    ) : (
                                        <motion.div 
                                            key="placeholder"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            className="absolute inset-0 flex items-center justify-center"
                                        >
                                            <Placeholder />
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </div>
                    </div>
                </main>
                
                <footer className="mt-8 text-center text-slate-400 text-xs font-medium">
                    &copy; 2026 BootstrapMD Research Systems. All rights reserved.
                </footer>
            </div>

            <ChatButton onClick={() => setIsChatOpen(true)} />
            <ChatModal isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} messages={chatMessages} onSendMessage={handleSendMessage} isLoading={isChatLoading} />
        </div>
    );
};

const ControlsSection: React.FC<any> = ({ 
    fileName, 
    onFileChange, 
    targetSize, 
    onTargetSizeChange, 
    onAugment, 
    isDataLoaded, 
    isLoading, 
    onReset,
    originalData,
    customColumnTypes,
    onColumnTypeChange,
    onResetColumnTypes
}) => (
    <div className="space-y-8">
        <section>
            <div className="flex items-center gap-3 mb-6">
                <span className="flex items-center justify-center w-8 h-8 rounded-2xl bg-blue-600 text-white text-xs font-black shadow-lg shadow-blue-100">1</span>
                <h3 className="text-xs font-black text-slate-400 uppercase tracking-[0.2em]">Source Dataset</h3>
            </div>
            
            {!fileName ? (
                <div className="group relative border-2 border-dashed border-slate-200 rounded-[2rem] p-8 text-center hover:border-blue-400 hover:bg-blue-50/50 transition-all duration-500 cursor-pointer overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <input type="file" onChange={onFileChange} accept=".csv" className="absolute inset-0 opacity-0 cursor-pointer z-10" id="f-up" />
                    <div className="flex flex-col items-center relative z-0">
                        <div className="p-4 bg-white rounded-2xl shadow-sm border border-slate-100 mb-4 group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                            <Upload className="h-8 w-8 text-blue-600" />
                        </div>
                        <p className="text-base font-extrabold text-slate-800">Import CSV Data</p>
                        <p className="text-xs text-slate-400 mt-2 font-medium">Drag & drop or click to browse</p>
                    </div>
                </div>
            ) : (
                <div className="p-4 bg-white rounded-2xl border border-slate-200 flex justify-between items-center shadow-sm group hover:border-blue-200 transition-all">
                    <div className="flex items-center gap-3 overflow-hidden">
                        <div className="p-2.5 bg-blue-50 rounded-xl group-hover:bg-blue-100 transition-colors shrink-0">
                            <Database className="h-5 w-5 text-blue-600" />
                        </div>
                        <div className="overflow-hidden">
                            <span className="block font-bold text-slate-800 truncate text-sm">{fileName}</span>
                            <span className="block text-[10px] text-slate-400 font-bold uppercase tracking-tighter">
                                {originalData?.length || 0} rows • {Object.keys(originalData?.[0] || {}).length} variables
                            </span>
                        </div>
                    </div>
                    <button onClick={onReset} className="p-2 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all shrink-0">
                        <X className="h-5 w-5" />
                    </button>
                </div>
            )}
        </section>

        <section className={isDataLoaded ? 'animate-in fade-in slide-in-from-top-4 duration-700' : 'opacity-30 pointer-events-none'}>
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center w-8 h-8 rounded-2xl bg-blue-600 text-white text-xs font-black shadow-lg shadow-blue-100">2</span>
                    <h3 className="text-xs font-black text-slate-400 uppercase tracking-[0.2em]">Step 2: Evaluate Data Types</h3>
                </div>
                {isDataLoaded && (
                    <button 
                        onClick={onResetColumnTypes} 
                        className="text-[10px] font-bold text-slate-400 hover:text-blue-600 transition-colors uppercase tracking-wider"
                    >
                        Reset Defaults
                    </button>
                )}
            </div>

            {isDataLoaded && originalData && originalData.length > 0 ? (
                <div className="space-y-3">
                    <p className="text-xs text-slate-500 font-medium leading-relaxed">
                        Verify auto-detected variable types. Select from the dropdown to correct any mislabeled variables:
                    </p>
                    
                    <div className="max-h-56 overflow-y-auto pr-1 space-y-2 rounded-2xl border border-slate-200/80 p-2.5 bg-slate-50/50">
                        {Object.keys(originalData[0] || {}).map((col) => {
                            const rawType = customColumnTypes[col] || 'continuous';
                            const currentType = (rawType === 'numerical' ? 'continuous' : rawType) as 'continuous' | 'categorical' | 'ordinal';
                            const values = originalData.map(r => r[col]).filter(v => v !== null && v !== undefined && String(v).trim() !== '');
                            const uniqCount = new Set(values).size;
                            
                            return (
                                <div key={col} className="p-2.5 bg-white rounded-xl border border-slate-100 shadow-2xs flex items-center justify-between gap-3 hover:border-blue-200 transition-all">
                                    <div className="min-w-0 flex-1">
                                        <span className="block font-black text-slate-800 text-xs truncate" title={col}>
                                            {col}
                                        </span>
                                        <span className="block text-[10px] text-slate-400 font-semibold">
                                            {uniqCount} unique value{uniqCount !== 1 ? 's' : ''}
                                        </span>
                                    </div>
                                    
                                    <select
                                        value={currentType}
                                        onChange={(e) => onColumnTypeChange(col, e.target.value as 'continuous' | 'categorical' | 'ordinal')}
                                        className={`text-xs font-black rounded-lg px-2.5 py-1.5 border outline-none transition-all cursor-pointer shrink-0 ${
                                            currentType === 'continuous'
                                                ? 'bg-blue-50/80 border-blue-200 text-blue-700 focus:ring-2 focus:ring-blue-500/20'
                                                : currentType === 'ordinal'
                                                ? 'bg-amber-50/80 border-amber-200 text-amber-700 focus:ring-2 focus:ring-amber-500/20'
                                                : 'bg-purple-50/80 border-purple-200 text-purple-700 focus:ring-2 focus:ring-purple-500/20'
                                        }`}
                                    >
                                        <option value="continuous">Continuous</option>
                                        <option value="categorical">Categorical</option>
                                        <option value="ordinal">Ordinal</option>
                                    </select>
                                </div>
                            );
                        })}
                    </div>
                    
                    <div className="flex items-center justify-between text-[10px] text-slate-400 font-bold uppercase tracking-wider pt-1 px-1">
                        <span>Total: {Object.keys(originalData[0] || {}).length} variables</span>
                        <span className="text-slate-600">
                            <span className="text-blue-600 font-black">{Object.values(customColumnTypes).filter(t => t === 'continuous' || (t as any) === 'numerical').length}</span> Continuous • <span className="text-amber-600 font-black">{Object.values(customColumnTypes).filter(t => t === 'ordinal').length}</span> Ordinal • <span className="text-purple-600 font-black">{Object.values(customColumnTypes).filter(t => t === 'categorical').length}</span> Categorical
                        </span>
                    </div>
                </div>
            ) : null}
        </section>

        <section className={isDataLoaded ? 'animate-in fade-in slide-in-from-top-4 duration-700' : 'opacity-30 pointer-events-none'}>
            <div className="flex items-center gap-3 mb-6">
                <span className="flex items-center justify-center w-8 h-8 rounded-2xl bg-blue-100 text-blue-600 text-xs font-black">3</span>
                <h3 className="text-xs font-black text-slate-400 uppercase tracking-[0.2em]">Synthesis Config</h3>
            </div>
            
            <div className="space-y-6">
                <div>
                    <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3 ml-1">Target Population Size</label>
                    <div className="relative group">
                        <select 
                            value={targetSize} 
                            onChange={e => onTargetSizeChange(Number(e.target.value))} 
                            className="input-field appearance-none pr-12 font-extrabold text-slate-800 cursor-pointer"
                        >
                            {[1000, 2000, 3000, 4000, 5000].map(size => (
                                <option key={size} value={size}>{size.toLocaleString()} Synthetic Patients</option>
                            ))}
                        </select>
                        <div className="absolute right-5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 group-hover:text-blue-500 transition-colors">
                            <ChevronRight className="h-5 w-5 rotate-90" />
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <div className="pt-2">
            <button 
                onClick={onAugment} 
                disabled={!isDataLoaded || isLoading} 
                className="btn-primary w-full py-5 text-lg"
            >
                {isLoading ? (
                    <>
                        <div className="h-5 w-5 border-3 border-white/30 border-t-white rounded-full animate-spin" />
                        <span className="tracking-tight">Synthesizing Population...</span>
                    </>
                ) : (
                    <>
                        <Play className="h-5 w-5 fill-current" />
                        <span className="tracking-tight">Generate Synthetic Data</span>
                    </>
                )}
            </button>
            
            {isDataLoaded && !isLoading && (
                <motion.p 
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mt-5 text-[10px] text-slate-400 font-bold flex items-center justify-center gap-1.5 uppercase tracking-widest"
                >
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    Dataset validated & ready
                </motion.p>
            )}
        </div>
    </div>
);

const ReportDisplay: React.FC<any> = ({ 
    report, 
    onDownload, 
    onDownloadReport, 
    onDownloadStats, 
    originalData, 
    augmentedData,
    customColumnTypes,
    onColumnTypeChange,
    onResetColumnTypes,
    onAugment,
    targetSize,
    isLoading
}) => {
    const [activeTab, setActiveTab] = useState<ReportTab>(ReportTab.SUMMARY);
    
    const tabs = [
        { id: ReportTab.SUMMARY, icon: Info, label: 'Overview' },
        { id: ReportTab.DATA_TYPES, icon: SlidersHorizontal, label: 'Variable Types' },
        { id: ReportTab.HISTOGRAMS, icon: BarChart3, label: 'Distributions' },
        { id: ReportTab.T_TEST, icon: Activity, label: 'T-Tests' },
        { id: ReportTab.MANN_WHITNEY, icon: Activity, label: 'U-Tests' },
        { id: ReportTab.KS_TEST, icon: Activity, label: 'KS-Tests' },
        { id: ReportTab.CATEGORICAL, icon: Database, label: 'Categorical' },
    ];

    return (
        <div className="flex flex-col h-full">
            <div className="flex flex-wrap gap-3 mb-8">
                <button onClick={onDownload} className="btn-primary text-xs py-2.5 bg-emerald-600 hover:bg-emerald-700 shadow-emerald-100">
                    <Download className="h-3.5 w-3.5" />
                    Export Synthetic CSV
                </button>
                <button onClick={onDownloadReport} className="btn-secondary text-xs py-2.5">
                    <FileText className="h-3.5 w-3.5" />
                    Technical Report
                </button>
                <button onClick={onDownloadStats} className="btn-secondary text-xs py-2.5 border-blue-100 text-blue-600 hover:bg-blue-50">
                    <Database className="h-3.5 w-3.5" />
                    Stats Summary
                </button>
            </div>

            <div className="flex border-b border-slate-100 mb-8 overflow-x-auto scrollbar-hide">
                {tabs.map(({ id, icon: Icon, label }) => (
                    <button 
                        key={id} 
                        onClick={() => setActiveTab(id)} 
                        className={`px-5 pb-4 whitespace-nowrap transition-all flex items-center gap-2.5 text-sm font-extrabold relative group ${
                            activeTab === id ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'
                        }`}
                    >
                        <Icon className={`h-4 w-4 transition-transform duration-300 ${activeTab === id ? 'scale-110' : 'group-hover:scale-110'}`} />
                        {label}
                        {activeTab === id && (
                            <motion.div 
                                layoutId="activeTab"
                                className="absolute bottom-0 left-0 right-0 h-1 bg-blue-600 rounded-t-full"
                            />
                        )}
                    </button>
                ))}
            </div>

            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        transition={{ duration: 0.2 }}
                    >
                        {activeTab === ReportTab.SUMMARY && (
                            <SummaryView 
                                report={report} 
                                onNavigateToDataTypes={() => setActiveTab(ReportTab.DATA_TYPES)}
                            />
                        )}
                        {activeTab === ReportTab.DATA_TYPES && (
                            <DataTypeEvaluationWorkspace
                                originalData={originalData}
                                customColumnTypes={customColumnTypes}
                                onColumnTypeChange={onColumnTypeChange}
                                onResetColumnTypes={onResetColumnTypes}
                                onAugment={onAugment}
                                targetSize={targetSize}
                                isLoading={isLoading}
                                isPostSynthesis={true}
                            />
                        )}
                        {activeTab === ReportTab.HISTOGRAMS && <HistogramDisplay report={report} originalData={originalData} augmentedData={augmentedData} />}
                        {activeTab === ReportTab.CATEGORICAL && <CategoricalResultsTable report={report} />}
                        {activeTab !== ReportTab.SUMMARY && activeTab !== ReportTab.DATA_TYPES && activeTab !== ReportTab.HISTOGRAMS && activeTab !== ReportTab.CATEGORICAL && (
                            <TestResultTable 
                                title={`${activeTab} Results`} 
                                data={(report as any)[activeTab === ReportTab.T_TEST ? 'tTest' : activeTab === ReportTab.MANN_WHITNEY ? 'mannWhitney' : 'ksTest']} 
                                report={report} 
                            />
                        )}
                    </motion.div>
                </AnimatePresence>
            </div>
        </div>
    );
};

const SummaryView: React.FC<any> = ({ report, onNavigateToDataTypes }) => (
    <div className="space-y-8 animate-in fade-in duration-700">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatCard 
                label="T-Test Similarity" 
                value={report.summary.tTestSimilarCount} 
                total={report.summary.totalNumerical} 
                color="blue"
                icon={Activity}
            />
            <StatCard 
                label="Distribution Fidelity" 
                value={report.summary.ksTestSimilarCount} 
                total={report.summary.totalNumerical} 
                color="indigo"
                icon={BarChart3}
            />
            <StatCard 
                label="Categorical Match" 
                value={report.summary.chiSquareSimilarCount} 
                total={report.summary.totalCategorical} 
                color="purple"
                icon={Database}
            />
        </div>

        {/* Evaluated Variable Overview Banner */}
        <div className="p-5 bg-white rounded-2xl border border-slate-100 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
                <div className="p-2.5 bg-blue-50 text-blue-600 rounded-xl">
                    <SlidersHorizontal className="h-5 w-5" />
                </div>
                <div>
                    <h5 className="text-xs font-black text-slate-800 uppercase tracking-wider">Evaluated Variable Data Types</h5>
                    <p className="text-xs text-slate-500 font-medium mt-0.5">
                        <span className="text-blue-600 font-black">{report.summary.totalContinuous ?? report.summary.totalNumerical} Continuous</span> • <span className="text-purple-600 font-black">{report.summary.totalCategorical} Categorical</span> {report.summary.totalOrdinal ? <span>• <span className="text-amber-600 font-black">{report.summary.totalOrdinal} Ordinal</span></span> : null}
                    </p>
                </div>
            </div>
            {onNavigateToDataTypes && (
                <button
                    onClick={onNavigateToDataTypes}
                    className="self-start sm:self-center text-xs font-black text-blue-600 hover:text-blue-700 bg-blue-50 hover:bg-blue-100/80 px-4 py-2 rounded-xl transition-all flex items-center gap-1.5 shrink-0"
                >
                    Review / Adjust Classifications
                    <ChevronRight className="w-3.5 h-3.5" />
                </button>
            )}
        </div>

        <div className="bg-slate-50/50 p-8 rounded-[2rem] border border-slate-100 flex flex-col md:flex-row justify-between gap-8 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500" />
            <div className="space-y-6 relative z-10">
                <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Synthesis Performance Index</h4>
                <div className="flex gap-12">
                    <div>
                        <div className="text-[10px] text-slate-400 uppercase font-black mb-2 flex items-center gap-1.5">
                            <Activity className="h-3 w-3" />
                            Avg. TVD
                        </div>
                        <div className={`text-4xl font-black tracking-tighter ${report.summary.avgTVD < 0.1 ? 'text-emerald-600' : 'text-orange-600'}`}>
                            {report.summary.avgTVD.toFixed(3)}
                        </div>
                    </div>
                    <div>
                        <div className="text-[10px] text-slate-400 uppercase font-black mb-2 flex items-center gap-1.5">
                            <Activity className="h-3 w-3" />
                            Avg. Cramer's V
                        </div>
                        <div className={`text-4xl font-black tracking-tighter ${getCramersVColor(report.summary.avgCramersV).replace('text-', 'text-')}`}>
                            {report.summary.avgCramersV.toFixed(3)}
                        </div>
                    </div>
                </div>
            </div>
            <div className="max-w-xs relative z-10">
                <div className="p-4 bg-white rounded-2xl border border-slate-100 shadow-sm">
                    <div className="flex items-start gap-3 text-slate-500 text-[11px] leading-relaxed font-medium italic">
                        <Info className="h-4 w-4 mt-0.5 shrink-0 text-blue-500" />
                        Cramer's V &lt; 0.1 indicates negligible difference in categorical distribution strength, suggesting high fidelity in synthetic generation.
                    </div>
                </div>
            </div>
        </div>

        <div className="bg-white rounded-[2rem] border border-slate-100 overflow-hidden shadow-sm">
            <div className="px-6 py-4 bg-slate-50/50 border-b border-slate-100 flex justify-between items-center">
                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Descriptive Statistics Matrix</h4>
            </div>
            <DescriptiveStatsTable data={report.descriptiveStats} report={report} />
        </div>
    </div>
);

const StatCard: React.FC<{label: string, value: number, total: number, color: string, icon: any}> = ({label, value, total, color, icon: Icon}) => {
    const percentage = total > 0 ? (value / total) * 100 : 0;
    const colorClasses: any = {
        blue: 'bg-blue-50/50 text-blue-600 border-blue-100',
        indigo: 'bg-indigo-50/50 text-indigo-600 border-indigo-100',
        purple: 'bg-purple-50/50 text-purple-600 border-purple-100',
    };
    
    return (
        <div className={`p-6 rounded-[2rem] border ${colorClasses[color]} flex flex-col justify-between relative group hover:scale-[1.02] transition-transform duration-300`}>
            <div className="flex justify-between items-start mb-4">
                <span className="text-[10px] font-black uppercase tracking-widest opacity-70">{label}</span>
                <Icon className="h-4 w-4 opacity-40 group-hover:opacity-100 transition-opacity" />
            </div>
            <div className="flex items-baseline gap-1.5 mt-2">
                <span className="text-4xl font-black tracking-tighter">{value}</span>
                <span className="text-sm font-bold opacity-40">/ {total}</span>
            </div>
            <div className="w-full h-2 bg-white rounded-full mt-5 overflow-hidden shadow-inner">
                <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className={`h-full rounded-full ${color === 'blue' ? 'bg-blue-500' : color === 'indigo' ? 'bg-indigo-500' : 'bg-purple-500'}`}
                />
            </div>
        </div>
    );
};

const VariableTypeBadge: React.FC<{meta: any}> = ({meta}) => {
    if (meta.type === 'ordinal') {
        return <span className="px-2.5 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-lg text-[9px] font-black uppercase tracking-widest">Ordinal</span>;
    }
    if (meta.type === 'continuous' || meta.type === 'numerical') {
        return <span className="px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-lg text-[9px] font-black uppercase tracking-widest">Continuous</span>;
    }
    return <span className="px-2.5 py-1 bg-purple-50 text-purple-700 border border-purple-200 rounded-lg text-[9px] font-black uppercase tracking-widest">Categorical</span>;
};

const TestResultTable: React.FC<any> = ({ title, data, report }) => (
    <div className="bg-white rounded-[2rem] border border-slate-100 overflow-hidden shadow-sm">
        <div className="px-6 py-4 bg-slate-50/50 border-b border-slate-100">
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">{title}</h4>
        </div>
        <table className="min-w-full text-xs text-left">
            <thead>
                <tr className="bg-slate-50/30 border-b border-slate-100 text-[10px] text-slate-400 uppercase font-black tracking-widest">
                    <th className="px-6 py-4">Variable</th>
                    <th className="px-6 py-4">Type</th>
                    <th className="px-6 py-4 text-center">Uniq</th>
                    <th className="px-6 py-4">P-Val</th>
                    <th className="px-6 py-4">Stat</th>
                    <th className="px-6 py-4 text-right">Fidelity</th>
                </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
                {data.map((item: any) => (
                    <tr key={item.column} className="hover:bg-slate-50/50 transition-colors group">
                        <td className="px-6 py-4 font-extrabold text-slate-800 group-hover:text-blue-600 transition-colors">{item.column}</td>
                        <td className="px-6 py-4"><VariableTypeBadge meta={report.columnMetadata[item.column]} /></td>
                        <td className="px-6 py-4 text-center font-mono text-slate-400 font-bold">{report.columnMetadata[item.column].uniqueCount}</td>
                        <td className="px-6 py-4 font-mono text-slate-600 font-bold">{formatPValue(item.pValue)}</td>
                        <td className="px-6 py-4 font-mono text-slate-600 font-bold">{item.statistic.toFixed(2)}</td>
                        <td className="px-6 py-4 text-right">
                            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-black ${
                                item.isSimilar ? 'bg-emerald-50 text-emerald-600 border border-emerald-100' : 'bg-red-50 text-red-600 border border-red-100'
                            }`}>
                                {item.isSimilar ? <CheckCircle2 className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}
                                {item.isSimilar ? 'HIGH' : 'LOW'}
                            </span>
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
);

const CategoricalResultsTable: React.FC<any> = ({ report }) => (
    <div className="bg-white rounded-[2rem] border border-slate-100 overflow-hidden shadow-sm">
        <div className="px-6 py-4 bg-slate-50/50 border-b border-slate-100">
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Categorical Fidelity Analysis</h4>
        </div>
        <table className="min-w-full text-xs text-left">
            <thead>
                <tr className="bg-slate-50/30 border-b border-slate-100 text-[10px] text-slate-400 uppercase font-black tracking-widest">
                    <th className="px-6 py-4">Variable</th>
                    <th className="px-6 py-4">Type</th>
                    <th className="px-6 py-4 text-center">Uniq</th>
                    <th className="px-6 py-4">χ² P-Val</th>
                    <th className="px-6 py-4">Cramer's V</th>
                    <th className="px-6 py-4">TVD</th>
                    <th className="px-6 py-4 text-right">Fidelity</th>
                </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
                {report.totalVariationDistance.map((item: any) => {
                    const chiResult = report.chiSquare.find((c: any) => c.column === item.column);
                    const meta = report.columnMetadata[item.column];
                    const v = chiResult?.cramersV || 0;
                    const isSimilar = (chiResult?.isSimilar || v < 0.1 || item.isSimilar);
                    return (
                        <tr key={item.column} className="hover:bg-slate-50/50 transition-colors group">
                            <td className="px-6 py-4 font-extrabold text-slate-800 group-hover:text-blue-600 transition-colors">{item.column}</td>
                            <td className="px-6 py-4"><VariableTypeBadge meta={meta} /></td>
                            <td className="px-6 py-4 text-center font-mono text-slate-400 font-bold">{meta.uniqueCount}</td>
                            <td className="px-6 py-4 font-mono text-slate-600 font-bold">{chiResult ? formatPValue(chiResult.pValue) : 'N/A'}</td>
                            <td className={`px-6 py-4 font-mono font-black ${getCramersVColor(v)}`}>{v.toFixed(3)}</td>
                            <td className="px-6 py-4 font-mono text-slate-600 font-bold">{item.value.toFixed(3)}</td>
                            <td className="px-6 py-4 text-right">
                                <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-black ${
                                    isSimilar ? 'bg-emerald-50 text-emerald-600 border border-emerald-100' : 'bg-red-50 text-red-600 border border-red-100'
                                }`}>
                                    {isSimilar ? <CheckCircle2 className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}
                                    {isSimilar ? 'HIGH' : 'LOW'}
                                </span>
                            </td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    </div>
);

const DescriptiveStatsTable: React.FC<any> = ({ data, report }) => (
    <table className="min-w-full text-[11px] text-left">
        <thead>
            <tr className="text-slate-400 uppercase font-black tracking-widest border-b border-slate-50 bg-slate-50/30">
                <th className="px-6 py-4">Variable</th>
                <th className="px-6 py-4">Mean (Orig)</th>
                <th className="px-6 py-4">Mean (Aug)</th>
                <th className="px-6 py-4 text-right">95% CI Difference</th>
            </tr>
        </thead>
        <tbody className="divide-y divide-slate-50">
            {data.map((item: any) => {
                const meta = report.columnMetadata[item.column];
                const p = meta.precision;
                return (
                    <tr key={item.column} className="hover:bg-slate-50/50 transition-colors group">
                        <td className="px-6 py-4 font-extrabold text-slate-800 group-hover:text-blue-600 transition-colors">{item.column}</td>
                        <td className="px-6 py-4 font-mono text-slate-600 font-bold">{item.original.mean.toFixed(p)}</td>
                        <td className="px-6 py-4 font-mono text-slate-600 font-bold">{item.augmented.mean.toFixed(p)}</td>
                        <td className="px-6 py-4 text-right font-mono text-slate-400 italic font-medium">
                            [{item.meanDifferenceCI.lower.toFixed(p)}, {item.meanDifferenceCI.upper.toFixed(p)}]
                        </td>
                    </tr>
                );
            })}
        </tbody>
    </table>
);

const HistogramDisplay: React.FC<any> = ({ report, originalData, augmentedData }) => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-8">
        {Object.keys(report.columnMetadata || {}).map((colName) => (
            <Histogram 
                key={colName} 
                title={colName} 
                isNumerical={report.columnMetadata[colName]?.type === 'numerical' && !report.columnMetadata[colName]?.isHeuristicCategorical}
                originalData={originalData.map((r: any) => r[colName])} 
                augmentedData={augmentedData.map((r: any) => r[colName])} 
            />
        ))}
    </div>
);

const Histogram: React.FC<any> = ({ originalData, augmentedData, title, isNumerical }) => {
    const cleanOrig = originalData.filter((v: any) => v !== null && v !== undefined);
    const cleanAug = augmentedData.filter((v: any) => v !== null && v !== undefined);
    
    if (cleanOrig.length === 0 && cleanAug.length === 0) return null;

    let counts: { label: string; o: number; a: number }[] = [];
    let minLabel = '';
    let maxLabel = '';

    if (isNumerical) {
        const bins = 15;
        const allNumeric = [...cleanOrig, ...cleanAug].map(Number).filter(v => typeof v === 'number' && !isNaN(v));
        if (allNumeric.length === 0) return null;

        const min = Math.min(...allNumeric);
        const max = Math.max(...allNumeric);
        const range = max - min;
        const step = range === 0 ? 1 : range / bins;

        counts = Array.from({ length: bins }, (_, i) => {
            const binMin = min + i * step;
            const binMax = i === bins - 1 ? max + 0.0001 : min + (i + 1) * step;
            const originalInBin = cleanOrig.filter((v: any) => {
                const num = Number(v);
                return !isNaN(num) && num >= binMin && num < binMax;
            }).length;
            const augmentedInBin = cleanAug.filter((v: any) => {
                const num = Number(v);
                return !isNaN(num) && num >= binMin && num < binMax;
            }).length;
            
            return {
                label: `${binMin.toFixed(1)}-${binMax.toFixed(1)}`,
                o: originalInBin / (cleanOrig.length || 1),
                a: augmentedInBin / (cleanAug.length || 1)
            };
        });
        minLabel = min.toFixed(1);
        maxLabel = max.toFixed(1);
    } else {
        // Categorical or discrete discrete variables: find unique values from original and augmented
        const uniqueVals = Array.from(new Set([...cleanOrig, ...cleanAug])).map(v => String(v).trim());
        
        // Sort unique values nicely
        uniqueVals.sort((a, b) => {
            const numA = Number(a);
            const numB = Number(b);
            if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
            return a.localeCompare(b);
        });

        // Limit to top 15 categories to avoid visual clutter
        const topVals = uniqueVals.slice(0, 15);

        counts = topVals.map(val => {
            const oCount = cleanOrig.filter((v: any) => String(v).trim() === val).length;
            const aCount = cleanAug.filter((v: any) => String(v).trim() === val).length;
            return {
                label: val,
                o: oCount / (cleanOrig.length || 1),
                a: aCount / (cleanAug.length || 1)
            };
        });
        minLabel = topVals[0] || '';
        maxLabel = topVals[topVals.length - 1] || '';
    }

    const maxH = Math.max(...counts.map(c => Math.max(c.o, c.a)), 0.001);

    return (
        <div className="bg-white p-6 rounded-[2rem] border border-slate-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all duration-300 flex flex-col justify-between min-h-[300px]">
            <div>
                <div className="flex justify-between items-center mb-6">
                    <div className="text-xs font-black text-slate-800 uppercase tracking-widest truncate max-w-[60%]">{title}</div>
                    <div className="flex gap-4 shrink-0">
                        <div className="flex items-center gap-2 text-[9px] font-black text-slate-400 uppercase tracking-tighter">
                            <div className="w-2.5 h-2.5 bg-blue-500 rounded-full shadow-sm shadow-blue-200"></div>
                            Original
                        </div>
                        <div className="flex items-center gap-2 text-[9px] font-black text-slate-400 uppercase tracking-tighter">
                            <div className="w-2.5 h-2.5 bg-rose-500 rounded-full shadow-sm shadow-rose-200"></div>
                            Synthetic
                        </div>
                    </div>
                </div>
                <div className="h-40 flex items-end gap-[3px] border-b border-slate-100 px-1 relative">
                    {counts.map((c, i) => (
                        <div key={i} className="flex-1 flex gap-[1.5px] items-end h-full group relative" style={{ width: '100%' }}>
                            <motion.div 
                                initial={{ height: 0 }}
                                animate={{ height: `${(c.o / maxH) * 100}%` }}
                                transition={{ duration: 0.8, delay: i * 0.02 }}
                                className="bg-blue-500 w-full rounded-t-sm opacity-80 group-hover:opacity-100 transition-opacity"
                            />
                            <motion.div 
                                initial={{ height: 0 }}
                                animate={{ height: `${(c.a / maxH) * 100}%` }}
                                transition={{ duration: 0.8, delay: i * 0.02 + 0.1 }}
                                className="bg-rose-500 w-full rounded-t-sm opacity-80 group-hover:opacity-100 transition-opacity"
                            />
                            {/* Elegant custom tooltip on hover */}
                            <div className="absolute hidden group-hover:flex bottom-full left-1/2 -translate-x-1/2 bg-slate-900 border border-slate-850 text-white text-[9px] font-bold p-2.5 rounded-xl shadow-2xl z-20 whitespace-nowrap mb-1">
                                {c.label}: Orig {(c.o * 100).toFixed(1)}% | Synth {(c.a * 100).toFixed(1)}%
                            </div>
                        </div>
                    ))}
                </div>
            </div>
            <div className="flex justify-between text-[10px] font-black text-slate-300 mt-4 uppercase tracking-wider overflow-hidden">
                <span className="truncate max-w-[45%] font-bold">{minLabel}</span>
                <span className="truncate max-w-[45%] font-bold">{maxLabel}</span>
            </div>
        </div>
    );
};

const LoadingSpinner = () => (
    <div className="flex flex-col items-center gap-6">
        <div className="relative">
            <div className="h-16 w-16 border-4 border-slate-100 rounded-full"></div>
            <div className="h-16 w-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin absolute top-0 left-0"></div>
        </div>
        <div className="text-center">
            <h4 className="text-blue-600 font-black text-sm uppercase tracking-widest animate-pulse">Analyzing Statistics</h4>
            <p className="text-slate-400 text-xs mt-1">Verifying synthetic population fidelity...</p>
        </div>
    </div>
);

const ErrorMessage = ({message}: any) => (
    <div className="flex flex-col items-center gap-4 p-8 bg-red-50 rounded-3xl border border-red-100 text-center max-w-sm">
        <div className="p-3 bg-white rounded-2xl shadow-sm">
            <AlertCircle className="h-8 w-8 text-red-500" />
        </div>
        <div>
            <h4 className="text-red-600 font-black text-sm uppercase tracking-widest">Synthesis Failed</h4>
            <p className="text-red-500/80 text-xs mt-1 leading-relaxed">{message}</p>
        </div>
    </div>
);

const Placeholder = () => (
    <div className="flex flex-col items-center gap-6 opacity-40">
        <div className="p-6 bg-slate-100 rounded-full">
            <BarChart3 className="h-12 w-12 text-slate-400" />
        </div>
        <div className="text-center">
            <p className="text-slate-500 font-bold">Workspace Empty</p>
            <p className="text-slate-400 text-xs mt-1">Upload a dataset to begin statistical synthesis.</p>
        </div>
    </div>
);

const ChatButton = ({onClick}: any) => (
    <motion.button 
        whileHover={{ scale: 1.05, y: -5 }}
        whileTap={{ scale: 0.95 }}
        onClick={onClick} 
        className="fixed bottom-8 right-8 bg-gradient-to-br from-slate-900 to-slate-800 text-white p-5 rounded-[2rem] shadow-2xl shadow-slate-400/50 hover:shadow-blue-500/20 transition-all z-40 flex items-center gap-4 border border-white/10"
    >
        <div className="relative">
            <MessageSquare className="h-6 w-6" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full border-2 border-slate-900 animate-pulse" />
        </div>
        <span className="text-xs font-black uppercase tracking-[0.2em] pr-2">AI Assistant</span>
    </motion.button>
);

const ChatModal = ({isOpen, onClose, messages, onSendMessage, isLoading}: any) => {
    const [input, setInput] = useState('');
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 bg-slate-900/60 backdrop-blur-md z-50 flex items-center justify-center p-4"
                >
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95, y: 40 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 40 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className="bg-white w-full max-w-xl rounded-[2.5rem] shadow-2xl flex flex-col h-[80vh] overflow-hidden border border-white/20"
                    >
                        <div className="p-8 bg-slate-900 text-white flex justify-between items-center relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
                                <Activity className="h-24 w-24" />
                            </div>
                            <div className="flex items-center gap-4 relative z-10">
                                <div className="p-3 bg-blue-600 rounded-2xl shadow-lg shadow-blue-500/20">
                                    <Activity className="h-6 w-6" />
                                </div>
                                <div>
                                    <h4 className="font-black text-base uppercase tracking-widest">Statistical Intelligence</h4>
                                    <p className="text-[10px] opacity-50 font-black uppercase tracking-[0.2em]">BootstrapMD Core Assistant</p>
                                </div>
                            </div>
                            <button onClick={onClose} className="p-3 hover:bg-white/10 rounded-2xl transition-colors relative z-10">
                                <X className="h-6 w-6" />
                            </button>
                        </div>
                        
                        <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-8 bg-slate-50/30 custom-scrollbar">
                            {messages.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-6 opacity-30">
                                    <div className="p-8 bg-slate-100 rounded-full">
                                        <MessageSquare className="h-16 w-16 text-slate-400" />
                                    </div>
                                    <div className="space-y-2">
                                        <p className="text-base font-black text-slate-500 uppercase tracking-widest">How can I assist today?</p>
                                        <p className="text-xs text-slate-400 font-medium italic">"Analyze the distribution fidelity of the age variable."</p>
                                    </div>
                                </div>
                            )}
                            
                            {messages.map((m: any, i: number) => (
                                <motion.div 
                                    initial={{ opacity: 0, x: m.role === 'user' ? 20 : -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    key={i} 
                                    className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div className={`max-w-[85%] p-5 rounded-3xl shadow-sm text-sm leading-relaxed ${
                                        m.role === 'user' 
                                            ? 'bg-blue-600 text-white rounded-tr-none font-bold' 
                                            : 'bg-white border border-slate-100 rounded-tl-none text-slate-700 font-bold'
                                    }`}>
                                        {m.text}
                                    </div>
                                </motion.div>
                            ))}
                            
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-white border border-slate-100 rounded-3xl rounded-tl-none p-5 shadow-sm flex gap-2">
                                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                                    </div>
                                </div>
                            )}
                        </div>
                        
                        <form className="p-8 bg-white border-t border-slate-100 flex gap-4" onSubmit={e => { e.preventDefault(); onSendMessage(input); setInput(''); }}>
                            <input 
                                value={input} 
                                onChange={e => setInput(e.target.value)} 
                                className="flex-1 bg-slate-50 border border-slate-200 rounded-[1.5rem] px-6 py-4 text-sm focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 outline-none transition-all font-bold text-slate-700" 
                                placeholder="Inquire about synthesis results..." 
                                autoFocus
                            />
                            <button type="submit" disabled={!input.trim() || isLoading} className="bg-blue-600 text-white p-4 rounded-2xl disabled:opacity-50 hover:bg-blue-700 shadow-xl shadow-blue-200 transition-all active:scale-95 flex items-center justify-center">
                                <Send className="h-6 w-6" />
                            </button>
                        </form>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default App;
