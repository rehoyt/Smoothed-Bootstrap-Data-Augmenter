
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import Papa from 'papaparse';
import { CsvData, Report, ReportTab, DescriptiveStatReport, StatisticalTestResult } from './types';
import { augmentData } from './services/dataAugmentation';
import { generateReport, formatReportAsText, generateStatsSummaryCsv } from './services/statisticalAnalysis';
import { GoogleGenAI, Chat } from '@google/genai';

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
                    
                    // Clean data: treat "null", "continuous", "NaN" strings as null
                    const cleanedData = results.data.map((row: any) => {
                        const newRow = { ...row };
                        for (const key in newRow) {
                            const val = newRow[key];
                            if (typeof val === 'string') {
                                const lower = val.toLowerCase().trim();
                                if (lower === 'null' || lower === 'continuous' || lower === 'nan' || lower === 'n/a' || lower === 'undefined') {
                                    newRow[key] = null;
                                }
                            }
                        }
                        return newRow;
                    });
                    
                    setOriginalData(cleanedData);
                },
            });
        }
    };

    const handleReset = useCallback(() => {
        setOriginalData(null);
        setAugmentedData(null);
        setFileName('');
        setReport(null);
        setError('');
        setChatMessages([]);
    }, []);

    const handleAugmentData = useCallback(() => {
        if (!originalData) return;
        setIsLoading(true); setError(''); setReport(null); setAugmentedData(null);
        setTimeout(() => {
            try {
                const augmented = augmentData(originalData, targetSize);
                setAugmentedData(augmented);
                setReport(generateReport(originalData, augmented));
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Error during augmentation.');
            } finally { setIsLoading(false); }
        }, 50);
    }, [originalData, targetSize]);

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
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
            <div className="w-full max-w-5xl">
                <header className="text-center mb-8">
                    <h1 className="text-4xl font-extrabold text-gray-800">Boots<span className="text-blue-600">GN</span></h1>
                    <p className="text-gray-600">Augment medical datasets with statistical integrity.</p>
                </header>
                <main className="bg-white rounded-xl shadow-2xl p-6 lg:p-10">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                        <ControlsSection
                            fileName={fileName} onFileChange={handleFileChange} 
                            targetSize={targetSize} onTargetSizeChange={setTargetSize}
                            onAugment={handleAugmentData} isDataLoaded={!!originalData} isLoading={isLoading}
                            onReset={handleReset}
                        />
                        <div className="flex flex-col min-h-[400px]">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-xl font-semibold">Results</h3>
                                {fileName && (
                                    <button 
                                        onClick={handleReset} 
                                        className="text-xs font-bold text-gray-400 hover:text-red-500 transition-colors uppercase tracking-wider"
                                    >
                                        Clear All
                                    </button>
                                )}
                            </div>
                            {isLoading && <LoadingSpinner />}
                            {error && <ErrorMessage message={error} />}
                            {report && !isLoading && (
                                <ReportDisplay 
                                    report={report} onDownload={handleDownload} onDownloadReport={handleDownloadReport} 
                                    onDownloadStats={handleDownloadStatsSummary} originalData={originalData!} augmentedData={augmentedData!}
                                />
                            )}
                            {!report && !isLoading && !error && <Placeholder />}
                        </div>
                    </div>
                </main>
            </div>
            <ChatButton onClick={() => setIsChatOpen(true)} />
            <ChatModal isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} messages={chatMessages} onSendMessage={handleSendMessage} isLoading={isChatLoading} />
        </div>
    );
};

const ControlsSection: React.FC<any> = ({ fileName, onFileChange, targetSize, onTargetSizeChange, onAugment, isDataLoaded, isLoading, onReset }) => (
    <div className="space-y-8">
        <div>
            <h3 className="text-xl font-semibold mb-4">1. Upload Data</h3>
            {!fileName ? (
                <div className="border-2 border-dashed rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                    <input type="file" onChange={onFileChange} accept=".csv" className="hidden" id="f-up" />
                    <label htmlFor="f-up" className="cursor-pointer text-blue-600 font-medium">Click to upload CSV</label>
                </div>
            ) : (
                <div className="p-3 bg-gray-50 rounded border flex justify-between items-center group">
                    <span className="font-medium text-gray-700 truncate mr-2">{fileName}</span>
                    <button onClick={onReset} className="text-gray-400 hover:text-red-500 text-xl leading-none transition-colors" title="Remove dataset">&times;</button>
                </div>
            )}
        </div>
        <div className={isDataLoaded ? '' : 'opacity-40 pointer-events-none'}>
            <h3 className="text-xl font-semibold mb-4">2. Configure</h3>
            <label className="block text-sm mb-1">Target Size</label>
            <select 
                value={targetSize} 
                onChange={e => onTargetSizeChange(Number(e.target.value))} 
                className="w-full border rounded p-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all bg-white"
            >
                <option value={1000}>1000</option>
                <option value={2000}>2000</option>
                <option value={3000}>3000</option>
                <option value={4000}>4000</option>
                <option value={5000}>5000</option>
            </select>
        </div>
        <div className="flex flex-col gap-2">
            <button onClick={onAugment} disabled={!isDataLoaded || isLoading} className="w-full py-3 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-700 disabled:bg-gray-400 shadow-md transition-all active:scale-[0.98]">
                {isLoading ? 'Processing...' : 'Run Augmentation'}
            </button>
            {isDataLoaded && (
                <button onClick={onReset} className="w-full py-2 text-gray-500 hover:text-red-600 text-sm font-medium transition-colors">
                    Reset & Upload New
                </button>
            )}
        </div>
    </div>
);

const ReportDisplay: React.FC<any> = ({ report, onDownload, onDownloadReport, onDownloadStats, originalData, augmentedData }) => {
    const [activeTab, setActiveTab] = useState<ReportTab>(ReportTab.SUMMARY);
    return (
        <div className="flex flex-col h-full">
            <div className="flex flex-wrap gap-2 mb-4">
                <button onClick={onDownload} className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 shadow-sm transition-colors">Data (CSV)</button>
                <button onClick={onDownloadReport} className="px-3 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-700 shadow-sm transition-colors">Report (TXT)</button>
                <button onClick={onDownloadStats} className="px-3 py-1 bg-indigo-600 text-white text-xs rounded hover:bg-indigo-700 font-bold shadow-sm transition-colors">Stats Summary (CSV)</button>
            </div>
            <div className="border-b flex space-x-4 mb-4 overflow-x-auto text-sm scrollbar-hide">
                {[ReportTab.SUMMARY, ReportTab.HISTOGRAMS, ReportTab.T_TEST, ReportTab.MANN_WHITNEY, ReportTab.KS_TEST, ReportTab.CATEGORICAL].map(t => (
                    <button key={t} onClick={() => setActiveTab(t)} className={`pb-2 whitespace-nowrap transition-all ${activeTab === t ? 'border-b-2 border-blue-600 text-blue-600 font-bold' : 'text-gray-500 hover:text-gray-700'}`}>{t}</button>
                ))}
            </div>
            <div className="flex-1 overflow-y-auto pr-2">
                {activeTab === ReportTab.SUMMARY && <SummaryView report={report} />}
                {activeTab === ReportTab.HISTOGRAMS && <HistogramDisplay report={report} originalData={originalData} augmentedData={augmentedData} />}
                {activeTab === ReportTab.CATEGORICAL && <CategoricalResultsTable report={report} />}
                {activeTab !== ReportTab.SUMMARY && activeTab !== ReportTab.HISTOGRAMS && activeTab !== ReportTab.CATEGORICAL && <TestResultTable title={`${activeTab} Results`} data={(report as any)[activeTab === ReportTab.T_TEST ? 'tTest' : activeTab === ReportTab.MANN_WHITNEY ? 'mannWhitney' : 'ksTest']} report={report} />}
            </div>
        </div>
    );
};

const SummaryView: React.FC<any> = ({ report }) => (
    <div className="space-y-4 animate-in fade-in duration-300">
        <div className="bg-blue-50 p-4 rounded border border-blue-100 text-sm">
            <h4 className="font-bold mb-2 text-blue-800">Performance Summary</h4>
            <div className="grid grid-cols-2 gap-4">
                <ul className="list-disc ml-5 space-y-1 text-blue-700">
                    <li>T-Test Similarity: {report.summary.tTestSimilarCount}/{report.summary.totalNumerical}</li>
                    <li>Distribution (KS) Similarity: {report.summary.ksTestSimilarCount}/{report.summary.totalNumerical}</li>
                    <li>Categorical (χ²) Similarity: {report.summary.chiSquareSimilarCount}/{report.summary.totalCategorical}</li>
                </ul>
                <div className="space-y-1 text-blue-700 border-l pl-4 border-blue-200">
                    <div>Avg. Categorical TVD: <span className={report.summary.avgTVD < 0.1 ? 'text-green-600 font-bold' : 'text-orange-600 font-bold'}>{report.summary.avgTVD.toFixed(3)}</span></div>
                    <div>Avg. Cramer's V: <span className={`${getCramersVColor(report.summary.avgCramersV)} font-bold`}>{report.summary.avgCramersV.toFixed(3)}</span></div>
                </div>
            </div>
            <p className="mt-3 text-[10px] text-blue-500 italic">* Cramer's V &lt; 0.1 indicates negligible difference in categorical distribution strength.</p>
        </div>
        <DescriptiveStatsTable data={report.descriptiveStats} report={report} />
    </div>
);

const VariableTypeBadge: React.FC<{meta: any}> = ({meta}) => {
    if (meta.isHeuristicCategorical) {
        return <span className="px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded text-[9px] font-bold uppercase" title="Treated as Categorical due to low unique value count (<=10)">Discrete</span>;
    }
    if (meta.type === 'numerical') {
        return <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-[9px] font-bold uppercase">Continuous</span>;
    }
    return <span className="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-[9px] font-bold uppercase">Categorical</span>;
};

const TestResultTable: React.FC<any> = ({ title, data, report }) => (
    <table className="min-w-full text-xs text-left animate-in slide-in-from-bottom-2 duration-300">
        <thead><tr className="border-b text-gray-400 uppercase tracking-tighter"><th>Variable</th><th>Type</th><th>Uniq</th><th>P-Val</th><th>Stat</th><th>Result</th></tr></thead>
        <tbody>{data.map((item: any) => (
            <tr key={item.column} className="border-b h-8 hover:bg-gray-50 transition-colors">
                <td className="font-medium">{item.column}</td>
                <td><VariableTypeBadge meta={report.columnMetadata[item.column]} /></td>
                <td className="font-mono text-gray-400">{report.columnMetadata[item.column].uniqueCount}</td>
                <td className="font-mono">{formatPValue(item.pValue)}</td>
                <td className="font-mono">{item.statistic.toFixed(2)}</td>
                <td><span className={`font-bold ${item.isSimilar ? 'text-green-600' : 'text-red-600'}`}>{item.isSimilar ? 'OK' : 'DIFF'}</span></td>
            </tr>
        ))}</tbody>
    </table>
);

const CategoricalResultsTable: React.FC<any> = ({ report }) => (
    <table className="min-w-full text-xs text-left animate-in slide-in-from-bottom-2 duration-300">
        <thead><tr className="border-b text-gray-400 uppercase tracking-tighter"><th>Variable</th><th>Type</th><th>Uniq</th><th>χ² P-Val</th><th>V</th><th>TVD</th><th>Stat</th></tr></thead>
        <tbody>{report.totalVariationDistance.map((item: any) => {
            const chiResult = report.chiSquare.find((c: any) => c.column === item.column);
            const meta = report.columnMetadata[item.column];
            const v = chiResult?.cramersV || 0;
            const isSimilar = (chiResult?.isSimilar || v < 0.1 || item.isSimilar);
            return (
                <tr key={item.column} className="border-b h-8 hover:bg-gray-50 transition-colors">
                    <td className="font-medium">{item.column}</td>
                    <td><VariableTypeBadge meta={meta} /></td>
                    <td className="font-mono text-gray-400">{meta.uniqueCount}</td>
                    <td className="font-mono">{chiResult ? formatPValue(chiResult.pValue) : 'N/A'}</td>
                    <td className={`font-mono font-bold ${getCramersVColor(v)}`}>{v.toFixed(3)}</td>
                    <td className="font-mono">{item.value.toFixed(3)}</td>
                    <td>
                        <span className={`font-bold ${isSimilar ? 'text-green-600' : 'text-red-600'}`}>
                            {isSimilar ? 'OK' : 'DIFF'}
                        </span>
                    </td>
                </tr>
            );
        })}</tbody>
    </table>
);

const DescriptiveStatsTable: React.FC<any> = ({ data, report }) => (
    <table className="min-w-full text-[10px] text-left">
        <thead><tr className="border-b text-gray-400 uppercase tracking-tighter"><th>Variable</th><th>Mean (Orig)</th><th>Mean (Aug)</th><th>95% CI Diff</th></tr></thead>
        <tbody>{data.map((item: any) => {
            const meta = report.columnMetadata[item.column];
            const p = meta.precision;
            return (
                <tr key={item.column} className="border-b h-8 hover:bg-gray-50 transition-colors">
                    <td className="font-medium">{item.column}</td>
                    <td className="font-mono">{item.original.mean.toFixed(p)}</td>
                    <td className="font-mono">{item.augmented.mean.toFixed(p)}</td>
                    <td className="font-mono text-gray-500 italic">
                        {`${item.meanDifferenceCI.lower.toFixed(p)}, ${item.meanDifferenceCI.upper.toFixed(p)}`}
                    </td>
                </tr>
            );
        })}</tbody>
    </table>
);

const HistogramDisplay: React.FC<any> = ({ report, originalData, augmentedData }) => (
    <div className="space-y-8 animate-in fade-in duration-500">
        {report.descriptiveStats.map((s: any) => (
            <Histogram key={s.column} title={s.column} originalData={originalData.map((r: any) => r[s.column])} augmentedData={augmentedData.map((r: any) => r[s.column])} />
        ))}
    </div>
);

const Histogram: React.FC<any> = ({ originalData, augmentedData, title }) => {
    const bins = 15;
    const all = [...originalData, ...augmentedData].filter(v => v !== null && typeof v === 'number');
    if (all.length === 0) return null;
    
    const min = Math.min(...all); const max = Math.max(...all);
    const range = max - min;
    const step = range === 0 ? 1 : range / bins;
    
    const counts = Array.from({length: bins}, (_, i) => {
        const binMin = min + i*step;
        const binMax = i === bins - 1 ? max + 0.0001 : min + (i+1)*step;
        return {
            o: originalData.filter((v: any) => v >= binMin && v < binMax).length / (originalData.length || 1),
            a: augmentedData.filter((v: any) => v >= binMin && v < binMax).length / (augmentedData.length || 1)
        };
    });
    const maxH = Math.max(...counts.map(c => Math.max(c.o, c.a)), 0.001);

    return (
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
            <div className="text-center font-bold text-xs mb-2 text-gray-700">{title}</div>
            <div className="h-24 flex items-end gap-[1px] border-b border-l border-gray-300 px-1">
                {counts.map((c, i) => (
                    <div key={i} className="flex-1 flex gap-[1px] items-end h-full" style={{ width: '100%' }}>
                        <div className="bg-blue-400 w-full rounded-t-sm" style={{height: `${(c.o/maxH)*100}%`}}></div>
                        <div className="bg-red-400 w-full rounded-t-sm opacity-80" style={{height: `${(c.a/maxH)*100}%`}}></div>
                    </div>
                ))}
            </div>
            <div className="flex justify-between text-[8px] text-gray-400 mt-1">
                <span>{min.toFixed(1)}</span>
                <div className="flex gap-2">
                    <span className="flex items-center gap-1"><span className="w-2 h-2 bg-blue-400 rounded-full"></span>Orig</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 bg-red-400 rounded-full"></span>Aug</span>
                </div>
                <span>{max.toFixed(1)}</span>
            </div>
        </div>
    );
};

const LoadingSpinner = () => (
    <div className="m-auto flex flex-col items-center gap-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="text-blue-600 font-medium text-sm animate-pulse">Analyzing Statistics...</span>
    </div>
);

const ErrorMessage = ({message}: any) => <div className="text-red-500 p-4 bg-red-50 rounded-lg border border-red-100 text-sm font-medium">{message}</div>;
const Placeholder = () => <div className="m-auto text-gray-400 text-center italic text-sm">Select a file and run augmentation to see statistical comparisons.</div>;
const ChatButton = ({onClick}: any) => (
    <button onClick={onClick} className="fixed bottom-6 right-6 bg-indigo-600 text-white p-4 rounded-full shadow-2xl hover:bg-indigo-700 hover:scale-110 transition-all active:scale-95 group">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
    </button>
);
const ChatModal = ({isOpen, onClose, messages, onSendMessage, isLoading}: any) => {
    const [input, setInput] = useState('');
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    if(!isOpen) return null;
    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white w-full max-w-lg rounded-2xl shadow-2xl flex flex-col h-[70vh] overflow-hidden animate-in zoom-in-95 duration-200">
                <div className="p-4 bg-indigo-600 text-white flex justify-between items-center">
                    <div>
                        <h4 className="font-bold">BootsGN Statistical Assistant</h4>
                        <p className="text-[10px] opacity-80">Powered by Gemini AI</p>
                    </div>
                    <button onClick={onClose} className="hover:bg-white/20 p-1 rounded-lg">&times;</button>
                </div>
                <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 text-sm bg-gray-50">
                    {messages.length === 0 && (
                        <div className="text-center text-gray-400 mt-10 space-y-2">
                            <p>Ask me questions like:</p>
                            <p className="text-xs italic">"Why is the KS-test result significant?"</p>
                            <p className="text-xs italic">"How should I interpret a low Mann-Whitney p-value?"</p>
                        </div>
                    )}
                    {messages.map((m: any, i: number) => (
                        <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] p-3 rounded-2xl shadow-sm ${m.role === 'user' ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-white border rounded-tl-none text-gray-800'}`}>
                                {m.text}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-white border rounded-2xl rounded-tl-none p-3 shadow-sm flex gap-1">
                                <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce"></span>
                                <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                                <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                            </div>
                        </div>
                    )}
                </div>
                <form className="p-4 bg-white border-t flex gap-2" onSubmit={e => { e.preventDefault(); onSendMessage(input); setInput(''); }}>
                    <input 
                        value={input} 
                        onChange={e => setInput(e.target.value)} 
                        className="flex-1 border border-gray-200 rounded-xl px-4 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all" 
                        placeholder="Type your question..." 
                        autoFocus
                    />
                    <button type="submit" disabled={!input.trim() || isLoading} className="bg-indigo-600 text-white p-2 rounded-xl disabled:opacity-50 hover:bg-indigo-700 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                    </button>
                </form>
            </div>
        </div>
    );
};

export default App;
