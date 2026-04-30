import { ArrowDownIcon, ArrowUpTrayIcon, CommandLineIcon, PlayIcon } from '@heroicons/react/16/solid';
import React, { useState, useEffect, type ChangeEvent } from 'react';
import { loadAndRun } from './pyodide-interface';

interface PlaygroundState {
  dashArray: string;
  scale: number;
  fillOpacity: number;
  strokeOpacity: number;
  strokeWidth: number;
  doorColor: number;
  tireColor: number;
  hoodColor: number;
  lightsColor: number;
  windowColor: number;
}

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('convert');
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [processedSvg, setProcessedSvg] = useState<string | undefined>(undefined);
  const [uploadedFile] = useState<string | null>(null);
  
  const [playgroundState, setPlaygroundState] = useState<PlaygroundState>({
    dashArray: '0',
    scale: 100,
    fillOpacity: 100,
    strokeOpacity: 100,
    strokeWidth: 2,
    doorColor: 50,
    tireColor: 50,
    hoodColor: 50,
    lightsColor: 50,
    windowColor: 50,
  });

  // Intersection Observer for Scroll-Spy
  useEffect(() => {
    const sections = ['convert', 'learn', 'animate', 'playground'];
    
    const observerCallback = (entries: IntersectionObserverEntry[]) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setActiveSection(entry.target.id);
        }
      });
    };

    const observerOptions: IntersectionObserverInit = {
      root: null,
      threshold: 0.6,
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);

    sections.forEach((id) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Mock result for visualization
      loadAndRun(file).then(svgString => {
        console.log(svgString);
        setProcessedSvg(svgString);
      }).catch(err => {
        console.error("Error processing PSD file:", err);
      });
      // setUploadedFile(URL.createObjectURL(file));
    }
  };

  const runAnimation = (): void => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 2000);
  };

  const updatePlayground = (field: keyof PlaygroundState, value: string | number): void => {
    setPlaygroundState(prev => ({ 
      ...prev, 
      [field]: typeof value === 'string' && !isNaN(Number(value)) && field !== 'dashArray' 
        ? Number(value) 
        : value 
    }));
  };

  const navItems = [
    { label: 'Convert', id: 'convert' },
    { label: 'Learn', id: 'learn' },
    { label: 'Animate', id: 'animate' },
    { label: 'Playground', id: 'playground' },
  ];

  return (
    <div className="min-h-screen font-sans text-slate-900 scroll-smooth">
      {/* Navigation Bar */}
      <nav className="fixed top-0 w-full bg-white/90 backdrop-blur-md shadow-sm z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="font-bold text-xl tracking-tight text-indigo-600">PSD to SVG</div>
          <div className="flex gap-8">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={`text-sm font-semibold transition-all duration-200 ${
                  activeSection === item.id 
                    ? 'text-indigo-600 border-b-2 border-indigo-600 pb-1' 
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                {item.label}
              </a>
            ))}
          </div>
        </div>
      </nav>

      {/* Convert Section */}
      <section id="convert" className="min-h-screen pt-24 pb-12 bg-slate-50 flex flex-col items-center justify-center">
        <div className="max-w-4xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-8 text-slate-800">Convert</h2>
          <div className="bg-white rounded-3xl shadow-xl p-10 flex flex-col items-center border border-slate-100">
            <div className="w-full max-w-sm aspect-square bg-slate-50 rounded-2xl border-2 border-dashed border-slate-200 flex items-center justify-center mb-8 overflow-hidden relative">
              {processedSvg ? (
                <div className="p-8 w-full h-full flex items-center justify-center">
                  <img src={`data:image/svg+xml;utf8,${encodeURIComponent(processedSvg)}`} />
                </div>
              ) : (
                <p className="text-slate-400 font-medium">SVG Preview Area</p>
              )}
            </div>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 bg-indigo-600 text-white px-8 py-4 rounded-xl hover:bg-indigo-700 transition-all cursor-pointer shadow-lg shadow-indigo-200">
                <ArrowUpTrayIcon className="size-6" />
                <span className="font-bold">Upload PSD</span>
                <input type="file" className="hidden" accept=".psd" onChange={handleFileUpload} />
              </label>
              <button
                disabled={!uploadedFile}
                className="flex items-center gap-2 bg-white border-2 border-slate-100 px-8 py-4 rounded-xl hover:border-slate-200 transition-all disabled:opacity-40 disabled:grayscale text-slate-700 font-bold"
              >
                <ArrowDownIcon className="size-6" />
                <span>Download</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Learn Section */}
      <section id="learn" className="min-h-screen pt-24 pb-12 bg-white flex flex-col items-center justify-center">
        <div className="max-w-5xl w-full px-6 text-center">
          <h2 className="text-4xl font-extrabold mb-8 text-slate-800">Learn</h2>
          <div className="aspect-video w-full rounded-3xl overflow-hidden shadow-2xl border-8 border-slate-50">
            <iframe
              className="w-full h-full"
              src="https://www.youtube.com/embed/5OeyH-UHewI"
              title="D3.js Tutorial"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </div>
        </div>
      </section>

      {/* Animate Section */}
      <section id="animate" className="min-h-screen pt-24 pb-12 bg-slate-50 flex flex-col items-center justify-center">
        <div className="max-w-5xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-8 text-slate-800">Animate</h2>
          <div className="grid md:grid-cols-2 gap-10">
            <div className="bg-white rounded-3xl shadow-lg p-8 border border-slate-100 flex flex-col items-center">
              <div className="w-full aspect-square bg-slate-50 rounded-2xl flex items-center justify-center mb-8">
                <svg width="160" height="160" viewBox="0 0 160 160">
                  <rect 
                    x="40" y="40" width="80" height="80" rx="8"
                    className={`fill-indigo-500 transition-all duration-1000 origin-center ${isAnimating ? 'rotate-180 scale-125' : ''}`}
                  />
                </svg>
              </div>
              <button
                onClick={runAnimation}
                disabled={isAnimating}
                className="w-full flex items-center justify-center gap-3 bg-indigo-600 text-white px-8 py-4 rounded-xl hover:bg-indigo-700 transition-all disabled:opacity-50 font-bold"
              >
                <PlayIcon className="size-6" />
                {isAnimating ? "Animating..." : "Run D3 Script"}
              </button>
            </div>
            
            <div className="bg-slate-900 rounded-3xl p-8 text-indigo-300 font-mono text-sm shadow-2xl relative overflow-hidden">
              <div className="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
                <div className="flex items-center gap-2 text-slate-400">
                  <CommandLineIcon className="size-6" />
                  <span className="text-xs uppercase tracking-widest font-bold">d3_transition.ts</span>
                </div>
              </div>
              <pre className="leading-relaxed">
{`d3.select("#box")
  .transition()
  .duration(1000)
  .style("fill", "#6366f1")
  .attr("transform", "rotate(180)")
  .on("end", () => {
    console.log("Sequence complete");
  });`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Playground Section */}
      <section id="playground" className="min-h-screen pt-24 pb-20 bg-white flex flex-col items-center justify-center">
        <div className="max-w-6xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-12 text-slate-800">Playground</h2>
          <div className="grid lg:grid-cols-12 gap-8">
            
            <div className="lg:col-span-7 bg-slate-50 rounded-[2.5rem] p-12 flex items-center justify-center shadow-inner border border-slate-100">
              <div style={{ transform: `scale(${playgroundState.scale / 100})` }} className="transition-transform duration-300 ease-out">
                <svg width="400" height="240" viewBox="0 0 400 240" className="drop-shadow-2xl">
                  {/* Procedural Car Body */}
                  <rect x="60" y="110" width="280" height="70" rx="15" 
                    fill={`hsl(${playgroundState.hoodColor * 3.6}, 65%, 55%)`} 
                    fillOpacity={playgroundState.fillOpacity / 100}
                    stroke="#1e293b"
                    strokeWidth={playgroundState.strokeWidth}
                    strokeDasharray={playgroundState.dashArray === '0' ? '0' : (Number(playgroundState.dashArray) * 12).toString()}
                    strokeOpacity={playgroundState.strokeOpacity / 100}
                  />
                  <path d="M100 110 L140 60 L260 60 L300 110 Z" 
                    fill={`hsl(${playgroundState.windowColor * 3.6}, 40%, 80%)`} 
                    stroke="#1e293b" 
                    strokeWidth={playgroundState.strokeWidth}
                  />
                  {/* Wheels */}
                  <circle cx="110" cy="180" r="30" fill={`hsl(${playgroundState.tireColor * 3.6}, 10%, 15%)`} stroke="#fff" strokeWidth="2" />
                  <circle cx="290" cy="180" r="30" fill={`hsl(${playgroundState.tireColor * 3.6}, 10%, 15%)`} stroke="#fff" strokeWidth="2" />
                </svg>
              </div>
            </div>

            <div className="lg:col-span-5 space-y-6">
              <div className="bg-slate-50 p-8 rounded-3xl border border-slate-100">
                <p className="font-bold mb-5 text-slate-700 uppercase tracking-tight">Dash Array Pattern</p>
                <div className="grid grid-cols-2 gap-3">
                  {['None', 'Small', 'Medium', 'Large'].map((label, idx) => (
                    <label key={label} className={`flex items-center justify-center gap-2 p-3 rounded-xl border-2 transition-all cursor-pointer ${playgroundState.dashArray === idx.toString() ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-100 text-slate-500 hover:border-indigo-200'}`}>
                      <input 
                        type="radio" 
                        className="hidden"
                        value={idx} 
                        checked={playgroundState.dashArray === idx.toString()}
                        onChange={(e) => updatePlayground('dashArray', e.target.value)}
                      />
                      <span className="text-sm font-bold">{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="bg-slate-50 p-8 rounded-3xl border border-slate-100 space-y-6">
                {[
                  { label: 'Global Scale', key: 'scale' as const, min: 100, max: 150 },
                  { label: 'Stroke Width', key: 'strokeWidth' as const, min: 0, max: 7 },
                  { label: 'Fill Opacity', key: 'fillOpacity' as const, min: 0, max: 100 },
                ].map((s) => (
                  <div key={s.key}>
                    <div className="flex justify-between text-xs font-black text-slate-400 uppercase mb-2">
                      <span>{s.label}</span>
                      <span className="text-indigo-600">{playgroundState[s.key]}</span>
                    </div>
                    <input 
                      type="range" min={s.min} max={s.max} value={playgroundState[s.key]}
                      onChange={(e) => updatePlayground(s.key, e.target.value)}
                      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                    />
                  </div>
                ))}
              </div>

              <div className="bg-slate-900 p-8 rounded-3xl space-y-5 shadow-xl">
                <p className="text-white font-bold text-xs uppercase tracking-widest mb-2 opacity-50">Color Mapping</p>
                {['Hood', 'Window', 'Tire'].map((part) => {
                  const key = `${part.toLowerCase()}Color` as keyof PlaygroundState;
                  return (
                    <div key={key}>
                      <input 
                        type="range" min="0" max="100" value={playgroundState[key] as number}
                        onChange={(e) => updatePlayground(key, e.target.value)}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-400"
                      />
                      <span className="text-[10px] text-slate-500 font-bold uppercase">{part} HUE</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default App;