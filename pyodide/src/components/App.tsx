import {
  ArrowDownIcon,
  ArrowUpTrayIcon,
  ChevronDownIcon,
  CommandLineIcon,
  PlayIcon
} from '@heroicons/react/16/solid';
import React, { useState, useEffect, type ChangeEvent } from 'react';
import { loadAndRun } from '../services/svg-generate.service.pyodide';
import { saveAs } from 'file-saver-es';
import { runAnimation as runSvgAnimation } from '../services/svg-animation.service';
import AnimateGraphic from './AnimateGraphic';
import AnimateCodeSnippet from './AnimateCodeSnippet';
import PlaygroundGraphic from './PlaygroundGraphic';
import {
  initialPlaygroundState,
  type PlaygroundState,
  updatePlaygroundState
} from '../services/svg-playground.service';
import YoutubeEmbed from './YoutubeEmbed';

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('convert');
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [processing, setProcessing] = useState<boolean>(false);
  const [processedSvg, setProcessedSvg] = useState<
    { fileName: string; svgString: string } | undefined
  >(undefined);

  const [playgroundState, setPlaygroundState] = useState<PlaygroundState>(
    initialPlaygroundState
  );

  const [expandedCards, setExpandedCards] = useState({
    dashArray: true,
    sliders: true,
    color: false
  });

  const toggleCard = (card: 'dashArray' | 'sliders' | 'color') => {
    setExpandedCards((prev) => ({
      ...prev,
      [card]: !prev[card]
    }));
  };

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
      threshold: 0.2
    };

    const observer = new IntersectionObserver(
      observerCallback,
      observerOptions
    );

    sections.forEach((id) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setProcessing(true);
      loadAndRun(file)
        .then((svgString) => {
          setProcessedSvg({
            fileName: file.name.replace(/\.psd$/i, '') + '.svg',
            svgString
          });
          setProcessing(false);
        })
        .catch((err) => {
          console.error('Error processing PSD file:', err);
          setProcessing(false);
        });
    }
  };

  const runAnimation = (): void => {
    runSvgAnimation(
      () => setIsAnimating(true),
      () => setIsAnimating(false)
    );
  };

  const updatePlayground = (
    field: keyof PlaygroundState,
    value: string | number
  ): void => {
    setPlaygroundState((prev) => updatePlaygroundState(prev, field, value));
  };

  const navItems = [
    { label: 'Convert', id: 'convert' },
    { label: 'Learn', id: 'learn' },
    { label: 'Animate', id: 'animate' },
    { label: 'Playground', id: 'playground' }
  ];

  return (
    <div className="min-h-screen font-sans text-slate-900 dark:text-white scroll-smooth">
      {/* Navigation Bar */}
      <nav className="sticky top-0 w-full bg-white/90 dark:bg-gray-800/90 backdrop-blur-md shadow-sm z-50">
        <div className="max-w-6xl mx-auto px-6 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="hidden sm:block font-bold text-xl tracking-tight text-indigo-600 dark:text-indigo-400">
            PSD to SVG
          </div>
          <div className="flex flex-wrap gap-4">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={`text-sm font-semibold transition-all duration-200 pb-1 border-b-2 ${
                  activeSection === item.id
                ? 'text-indigo-600 dark:text-indigo-400 border-indigo-600'
                    : 'text-slate-500 dark:text-gray-400 hover:text-slate-900 dark:hover:text-white border-transparent'
                }`}
              >
                {item.label}
              </a>
            ))}
          </div>
        </div>
      </nav>

      {/* Convert Section */}
      <section
        id="convert"
        className="-mt-14 pt-24 min-h-screen bg-slate-50 dark:bg-gray-900 flex flex-col items-center"
      >
        <div className="max-w-4xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-8 text-slate-800 dark:text-white">
            Convert
          </h2>
          <div className="max-content-height bg-white dark:bg-gray-800 rounded-3xl shadow-xl p-10 flex flex-col items-center border border-slate-100 dark:border-gray-700">
            <div className="w-full max-w-lg aspect-square bg-slate-50 dark:bg-gray-700 rounded-2xl border-2 border-dashed border-slate-200 dark:border-gray-600 flex items-center justify-center mb-8 overflow-hidden relative">
              {processing ? (
                <div className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-12 w-12 text-slate-400 dark:text-gray-500"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                </div>
              ) : processedSvg ? (
                <div className="p-8 flex w-full h-full items-center justify-center">
                  <img className="max-h-full max-w-full" alt="Processed SVG"
                    src={`data:image/svg+xml;utf8,${encodeURIComponent(processedSvg.svgString)}`}
                  />
                </div>
              ) : (
                <p className="text-slate-400 dark:text-gray-500 font-medium">SVG Preview Area</p>
              )}
            </div>
            <div className="flex flex-wrap gap-4 justify-center w-full">
              <label className="flex flex-wrap items-center gap-2 bg-indigo-600 text-white px-8 py-4 rounded-xl hover:bg-indigo-700 transition-all cursor-pointer shadow-lg shadow-indigo-200/80 dark:shadow-indigo-900/40">
                <ArrowUpTrayIcon className="size-6" />
                <span className="font-bold whitespace-normal">Upload PSD</span>
                <input
                  type="file"
                  className="hidden"
                  accept=".psd"
                  onChange={handleFileUpload}
                />
              </label>
              <button
                role="link"
                disabled={!processedSvg}
                onClick={() => {
                  if (processedSvg) {
                    const blob = new Blob([processedSvg.svgString], {
                      type: 'image/svg+xml'
                    });
                    saveAs(blob, processedSvg.fileName);
                  }
                }}
                className={`flex flex-wrap items-center gap-2 bg-white dark:bg-gray-800 border-2 border-slate-100 dark:border-gray-700 px-8 py-4 rounded-xl hover:border-slate-200 dark:hover:border-gray-500 transition-all disabled:opacity-40 disabled:grayscale text-slate-700 dark:text-white font-bold ${processedSvg ? 'cursor-pointer' : ''}`}
              >
                <ArrowDownIcon className="size-6" />
                <span className="whitespace-normal">Download</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Learn Section */}
      <section
        id="learn"
        className="min-h-screen pt-24 pb-12 bg-white dark:bg-gray-800 flex flex-col items-center"
      >
        <div className="max-w-5xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-8 text-slate-800 dark:text-white">Learn</h2>
          <div className="max-content-height aspect-video w-full shadow-2xl">
                <YoutubeEmbed />
          </div>
        </div>
      </section>

      {/* Animate Section */}
      <section
        id="animate"
        className="min-h-screen pt-24 pb-12 bg-slate-50 dark:bg-gray-900 flex flex-col items-center"
      >
        <div className="max-w-6xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-8 text-slate-800 dark:text-white">
            Animate
          </h2>
          <div className="grid lg:grid-cols-2 gap-10">
            <div className="bg-white dark:bg-gray-800 rounded-3xl shadow-lg p-8 border border-slate-100 dark:border-gray-700 flex flex-col items-center w-full flex flex-column max-content-height">
              <div className="w-full grow aspect-square bg-slate-50 dark:bg-gray-700 rounded-2xl flex items-center justify-center mb-8 min-h-0">
                <AnimateGraphic />
              </div>
              <button
                onClick={runAnimation}
                disabled={isAnimating}
                className="w-full flex-none flex items-center justify-center gap-3 bg-indigo-600 text-white px-8 py-4 rounded-xl hover:bg-indigo-700 transition-all disabled:opacity-50 font-bold"
              >
                <PlayIcon className="size-6" />
                {isAnimating ? 'Animating...' : 'Run D3 Script'}
              </button>
            </div>

            <div className="bg-black rounded-3xl pt-8 pb-8 text-indigo-300 font-mono text-sm shadow-2xl flex flex-col min-h-120">
              <div className="flex flex-none items-center justify-between border-b border-slate-800 dark:border-gray-700 pb-4">
                <div className="ml-8 mr-8 flex items-center gap-2 text-slate-400 dark:text-gray-500">
                  <CommandLineIcon className="size-6" />
                  <span className="text-xs uppercase tracking-widest font-bold">
                    D3 Transition
                  </span>
                </div>
              </div>
              <div className="flex-[1_1_0] pl-8 pr-8 pb-4 border-b border-slate-800 leading-relaxed overflow-auto">
                <AnimateCodeSnippet />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Playground Section */}
      <section
        id="playground"
        className="min-h-screen pt-24 pb-20 bg-white dark:bg-gray-800 flex flex-col items-center"
      >
        <div className="max-w-6xl w-full px-6">
          <h2 className="text-4xl font-extrabold mb-12 text-slate-800 dark:text-white">
            Playground
          </h2>
          <div className="grid lg:grid-cols-12 gap-8">
            <div className="lg:col-span-7 bg-slate-50 dark:bg-gray-700 rounded-[2.5rem] max-content-height flex items-center justify-center shadow-inner border border-slate-100 dark:border-gray-700 p-8">
              <div className="flex grow min-h-0 h-full items-center justify-center transition-transform duration-300 ease-out grow">
                <PlaygroundGraphic playgroundState={playgroundState} />
              </div>
            </div>

            <div className="lg:col-span-5 space-y-4">
              <div className="bg-slate-50 dark:bg-gray-700 p-4 rounded-3xl border border-slate-100 dark:border-gray-700">
                <button
                  type="button"
                  onClick={() => toggleCard('dashArray')}
                  className="w-full flex items-center justify-between gap-3 mb-5 text-left"
                >
                  <p className="font-bold text-slate-700 dark:text-white uppercase tracking-tight">
                    Dash Array Pattern
                  </p>
                  <ChevronDownIcon
                    className={`w-5 h-5 text-slate-500 transition-transform ${
                      expandedCards.dashArray ? 'rotate-180' : 'rotate-0'
                    }`}
                  />
                </button>
                <div
                  className={`overflow-hidden transition-all duration-300 ${
                    expandedCards.dashArray ? 'max-h-[1000px]' : 'max-h-0'
                  }`}
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {['None', 'Small', 'Medium', 'Large'].map((label, idx) => (
                      <label
                        key={label}
                        className={`flex flex-wrap items-center justify-center gap-2 p-3 rounded-xl border-2 transition-all cursor-pointer ${playgroundState.dashArray === label.toLocaleLowerCase() ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white dark:bg-gray-800 border-slate-100 dark:border-gray-700 text-slate-500 dark:text-gray-400 hover:border-indigo-200 dark:hover:border-indigo-500'}`}
                      >
                        <input
                          type="radio"
                          className="hidden"
                          value={idx}
                          checked={
                            playgroundState.dashArray ===
                            label.toLocaleLowerCase()
                          }
                          onChange={() =>
                            updatePlayground(
                              'dashArray',
                              label.toLocaleLowerCase()
                            )
                          }
                        />
                        <span className="text-sm font-bold">{label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-gray-700 p-4 rounded-3xl border border-slate-100 dark:border-gray-700">
                <button
                  type="button"
                  onClick={() => toggleCard('sliders')}
                  className="w-full flex items-center justify-between gap-3 mb-5 text-left"
                >
                  <p className="font-bold text-slate-700 dark:text-white uppercase tracking-tight">
                    Controls
                  </p>
                  <ChevronDownIcon
                    className={`w-5 h-5 text-slate-500 transition-transform ${
                      expandedCards.sliders ? 'rotate-180' : 'rotate-0'
                    }`}
                  />
                </button>
                <div
                  className={`overflow-hidden transition-all duration-300 ${
                    expandedCards.sliders ? 'max-h-[2000px]' : 'max-h-0'
                  }`}
                >
                  <div className="space-y-2">
                    {[
                      {
                        label: 'Global Scale',
                        key: 'scale' as const,
                        min: 0,
                        max: 1,
                        step: 0.1
                      },
                      {
                        label: 'Stroke Width',
                        key: 'strokeWidth' as const,
                        min: 0,
                        max: 10
                      },
                      {
                        label: 'Fill Opacity',
                        key: 'fillOpacity' as const,
                        min: 0,
                        max: 100
                      },
                      {
                        label: 'Stroke Opacity',
                        key: 'strokeOpacity' as const,
                        min: 0,
                        max: 100
                      }
                    ].map((s) => (
                      <div key={s.key}>
                        <div className="flex justify-between text-xs font-black text-slate-400 dark:text-gray-300 uppercase mb-2">
                          <span>{s.label}</span>
                          <span className="text-indigo-700 dark:text-indigo-300">
                            {playgroundState[s.key]}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={s.min}
                          max={s.max}
                          step={s.step ?? 1}
                          value={playgroundState[s.key]}
                          onChange={(e) =>
                            updatePlayground(s.key, e.target.value)
                          }
                          className="w-full h-2 bg-slate-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer accent-indigo-600 dark:accent-indigo-400"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 dark:bg-gray-900 p-4 rounded-3xl shadow-xl">
                <button
                  type="button"
                  onClick={() => toggleCard('color')}
                  className="w-full flex items-center justify-between gap-3 mb-5 text-left"
                >
                  <p className="text-white font-bold text-xs uppercase tracking-widest">
                    Color Mapping
                  </p>
                  <ChevronDownIcon
                    className={`w-5 h-5 text-slate-300 transition-transform ${
                      expandedCards.color ? 'rotate-180' : 'rotate-0'
                    }`}
                  />
                </button>
                <div
                  className={`overflow-hidden transition-all duration-300 ${
                    expandedCards.color ? 'max-h-[1400px]' : 'max-h-0'
                  }`}
                >
                  <div className="space-y-2">
                    {['Door', 'Tire', 'Hood', 'Lights', 'Window'].map(
                      (part) => {
                        const key =
                          `${part.toLowerCase()}Color` as keyof PlaygroundState;
                        return (
                          <div key={key}>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.01"
                              value={playgroundState[key] as number}
                              onChange={(e) =>
                                updatePlayground(key, e.target.value)
                              }
                              className="w-full h-1.5 bg-slate-700 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-400 dark:accent-indigo-400"
                            />
                            <span className="text-[10px] text-slate-300 dark:text-gray-300 font-bold uppercase">
                              {part} HUE
                            </span>
                          </div>
                        );
                      }
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default App;
