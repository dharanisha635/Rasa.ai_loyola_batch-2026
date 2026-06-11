const LOCAL_API  = "http://localhost:5000";
const HOSTED_API = "https://your-app.onrender.com"; // update when deployed

// Tries local first, falls back to hosted
const BASE = process.env.REACT_APP_API_URL || LOCAL_API;

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${BASE}/stats`);
  if (!res.ok) throw new Error(`Stats fetch failed: ${res.status}`);
  return res.json();
}

export async function generateReport(predictionResult) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.REACT_APP_GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: "llama-3.1-8b-instant",
      messages: [{
        role: "user",
        content: `You are an AI assistant for a municipal street cleanliness monitoring system.

A CNN model with MC Dropout uncertainty estimation analysed a street image and produced:
- Classification: ${predictionResult.label}
- Confidence: ${(predictionResult.confidence * 100).toFixed(1)}%
- MC Dropout uncertainty (std): ${predictionResult.uncertainty}
- Predictive entropy: ${predictionResult.entropy}
- Ambiguous prediction: ${predictionResult.is_ambiguous}

Write exactly 4 sentences as a professional street-level cleanliness report for a municipal officer. 
Sentence 1: State the classification result and confidence score clearly.
Sentence 2: Describe what this implies about the street's physical condition.
Sentence 3: Give one specific recommended action (cleaning crew dispatch / routine monitoring / re-inspection).
Sentence 4: Comment on prediction reliability using the uncertainty value — low uncertainty means high reliability.

Rules: No bullet points. No headers. No markdown bold. Plain paragraph only. Maximum 120 words.`
      }],
      max_tokens: 300
    })
  });
  if (!res.ok) {
  const errText = await res.text();
  console.error("Groq Error:", errText);
  throw new Error(`Report failed: ${res.status}`);
}
  const data = await res.json();
  return data.choices[0].message.content;
}