import { useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, User, MapPin, Stethoscope, HeartPulse } from 'lucide-react';

export default function App() {
  const [formData, setFormData] = useState({
    age: '',
    previous_pregnancies: '',
    anc_visits: '',
    pregnancy_complications: '',
    education_level: '',
    settlement_type: '',
    place_delivered: ''
  });

  const [result, setResult] = useState(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['age', 'previous_pregnancies', 'anc_visits'].includes(name) ? Number(value) : value
    }));
    setErrorMsg(''); 
  };

  const calculateRisk = async () => {
    if (Object.values(formData).some(val => val === '')) {
      setErrorMsg('Please complete all patient clinical profile fields before running the assessment.');
      return;
    }

    setIsCalculating(true);
    
    try {
      // PRACTICAL FIX: Automatically detect if we are live on Vercel or testing locally
      // This completely bypasses Vercel's environment variable bugs.
      const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
      const FALLBACK_URL = isLocal 
        ? 'http://127.0.0.1:8000/predict' 
        : 'https://triage-dashboard-9xap.onrender.com/predict';
        
      const API_URL = import.meta.env.VITE_API_URL || FALLBACK_URL;
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
      setIsCalculating(false);
      
    } catch (error) {
      console.error("Connection error:", error);
      setErrorMsg("Failed to connect to the AI Inference Engine. Please ensure the backend server is running.");
      setIsCalculating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-4 md:p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="flex items-center gap-3 pb-6 border-b border-slate-200">
          <div className="p-3 bg-blue-600 text-white rounded-lg shadow-sm">
            <Activity size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">ACEPHAP Maternal Health Triage</h1>
            <p className="text-sm text-slate-500 font-medium">Pregnancy Complications in Previous Pregnancy</p>
          </div>
        </header>

        {errorMsg && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg border border-red-200 flex items-center gap-2 font-medium">
            <AlertTriangle size={20} />
            {errorMsg}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Form Column */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
              <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                <User size={20} className="text-blue-500"/> Patient Clinical Profile
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600">Maternal Age (Years)</label>
                  <input type="number" name="age" value={formData.age} onChange={handleInputChange} min="10" max="60" placeholder="e.g. 25"
                    className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" />
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600">Previous Pregnancies (Parity)</label>
                  <input type="number" name="previous_pregnancies" value={formData.previous_pregnancies} onChange={handleInputChange} min="0" max="20" placeholder="e.g. 2"
                    className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" />
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600">ANC Visits</label>
                  <input type="number" name="anc_visits" value={formData.anc_visits} onChange={handleInputChange} min="0" max="15" placeholder="e.g. 4"
                    className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" />
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600 flex items-center gap-1">
                    <HeartPulse size={16} className="text-red-500"/> Pregnancy Complications?
                  </label>
                  <select name="pregnancy_complications" value={formData.pregnancy_complications} onChange={handleInputChange} 
                    className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none bg-white">
                    <option value="" disabled>Select option...</option>
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
              <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                <MapPin size={20} className="text-blue-500"/> Socio-Demographic & Delivery
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600">Education Level</label>
                  <select name="education_level" value={formData.education_level} onChange={handleInputChange} className="w-full p-2.5 border border-slate-300 rounded-lg outline-none bg-white">
                    <option value="" disabled>Select option...</option>
                    <option value="none_or_non_formal">None / Non-Formal</option>
                    <option value="arabic_ismiyya">Arabic / Ismiyya</option>
                    <option value="primary">Primary</option>
                    <option value="secondary">Secondary</option>
                    <option value="tertiary">Tertiary</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600">Settlement Type</label>
                  <select name="settlement_type" value={formData.settlement_type} onChange={handleInputChange} className="w-full p-2.5 border border-slate-300 rounded-lg outline-none bg-white">
                    <option value="" disabled>Select option...</option>
                    <option value="rural">Rural</option>
                    <option value="semi-urban">Semi-Urban</option>
                    <option value="urban">Urban</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-semibold text-slate-600 flex items-center gap-1">
                    <Stethoscope size={16} className="text-slate-500"/> Place of Delivery in Previous Pregnancy
                  </label>
                  <select name="place_delivered" value={formData.place_delivered} onChange={handleInputChange} className="w-full p-2.5 border border-slate-300 rounded-lg outline-none bg-white">
                    <option value="" disabled>Select option...</option>
                    <option value="hf">Health Facility (HF)</option>
                    <option value="home">Home</option>
                    <option value="enroute">Enroute to Facility</option>
                  </select>
                </div>
              </div>
            </div>

            <button 
              onClick={calculateRisk}
              disabled={isCalculating}
              className="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-md transition-all flex items-center justify-center gap-2 disabled:opacity-70">
              {isCalculating ? (
                <span className="animate-pulse flex items-center gap-2">Analyzing Patient Matrix...</span>
              ) : (
                <>Run Risk Assessment <Activity size={20}/></>
              )}
            </button>
          </div>

          {/* Results Column */}
          <div className="lg:col-span-1">
            <div className={`p-6 rounded-xl shadow-sm border transition-all duration-500 h-full ${
              !result ? 'bg-slate-100 border-slate-200 border-dashed flex flex-col items-center justify-center text-center min-h-[400px]' : 
              result.probability >= 50 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
            }`}>
              
              {!result ? (
                <div className="text-slate-400 space-y-3">
                  <Activity size={48} className="mx-auto opacity-50" />
                  <p className="font-medium">Awaiting patient data...</p>
                  <p className="text-sm">Submit the form to generate a predictive risk score via the LightGBM engine.</p>
                </div>
              ) : (
                <div className="space-y-6 animate-in fade-in zoom-in duration-300">
                  <div className="text-center pb-6 border-b border-white/20">
                    <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-2">Mortality Probability</h3>
                    <div className={`text-6xl font-black ${result.probability >= 50 ? 'text-red-600' : 'text-green-600'}`}>
                      {result.probability}%
                    </div>
                    <div className={`mt-3 inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-bold ${
                      result.probability >= 50 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                    }`}>
                      {result.probability >= 50 ? <AlertTriangle size={16}/> : <CheckCircle size={16}/>}
                      {result.classification}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-bold text-slate-800 mb-3 text-sm uppercase">Key Clinical Drivers</h4>
                    <ul className="space-y-3">
                      {result.drivers.map((driver, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm bg-white/60 p-3 rounded-lg border border-white/40">
                          {driver.impact.includes('Increase') || driver.impact.includes('Acute') ? (
                            <span className="text-red-500 mt-0.5">↑</span>
                          ) : (
                            <span className="text-green-500 mt-0.5">↓</span>
                          )}
                          <div>
                            <span className="font-bold text-slate-700 block">{driver.factor}</span>
                            <span className="text-slate-500 text-xs">{driver.impact}</span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {result.probability >= 50 && (
                    <div className="mt-4 p-4 bg-red-600 text-white rounded-lg shadow-inner">
                      <p className="font-bold text-sm mb-1 flex items-center gap-2"><AlertTriangle size={16}/> ACTION REQUIRED:</p>
                      <p className="text-xs opacity-90">Patient requires immediate escalation to a senior obstetrician and continuous monitoring.</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}