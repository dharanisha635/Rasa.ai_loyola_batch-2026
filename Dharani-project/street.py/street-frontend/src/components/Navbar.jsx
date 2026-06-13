export default function Navbar({ activeTab, setActiveTab }) {
  const tabs = [
    { id: "predict", label: "Predict",    icon: "🔍" },
    { id: "stats",   label: "Statistics", icon: "📊" },
    { id: "report",  label: "Report",     icon: "📄" },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="brand-icon">🏙️</span>
        <span className="brand-name">Street Cleanliness Detector</span>
      </div>
      <div className="navbar-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>
    </nav>
  );
}