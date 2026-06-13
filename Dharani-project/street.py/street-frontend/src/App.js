import { useState } from "react";
import Navbar from "./components/Navbar";
import PredictTab from "./components/PredictTab";
import StatsTab from "./components/StatsTab";
import ReportTab from "./components/ReportTab";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [lastPrediction, setLastPrediction] = useState(null);

  return (
    <div className="app">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {activeTab === "predict" && (
          <PredictTab onPrediction={setLastPrediction} />
        )}
        {activeTab === "stats"   && <StatsTab />}
        {activeTab === "report"  && (
          <ReportTab lastPrediction={lastPrediction} />
        )}
      </main>
    </div>
  );
}