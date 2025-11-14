export interface ExoplanetData {
  id: string;
  name: string;
  orbitalPeriod: number;
  transitDepth: number;
  radius: number;
  distance: number;
  starTemp: number;
  tabularPrediction: "Confirmed" | "Candidate" | "False Positive";
  tabularConfidence: number;
  lightCurvePrediction: "Confirmed" | "Candidate" | "False Positive";
  lightCurveConfidence: number;
  votes: { agree: number; disagree: number };
}

export const mockExoplanets: ExoplanetData[] = [
  {
    id: "KOI-1",
    name: "Kepler-1649c",
    orbitalPeriod: 19.5,
    transitDepth: 0.0012,
    radius: 1.06,
    distance: 300,
    starTemp: 3800,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.95,
    lightCurvePrediction: "Confirmed",
    lightCurveConfidence: 0.92,
    votes: { agree: 847, disagree: 23 },
  },
  {
    id: "KOI-2",
    name: "K2-18b",
    orbitalPeriod: 33.0,
    transitDepth: 0.0015,
    radius: 2.61,
    distance: 124,
    starTemp: 3457,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.98,
    lightCurvePrediction: "Confirmed",
    lightCurveConfidence: 0.96,
    votes: { agree: 1243, disagree: 15 },
  },
  {
    id: "KOI-3",
    name: "TRAPPIST-1e",
    orbitalPeriod: 6.1,
    transitDepth: 0.0008,
    radius: 0.92,
    distance: 39,
    starTemp: 2559,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.99,
    lightCurvePrediction: "Confirmed",
    lightCurveConfidence: 0.97,
    votes: { agree: 2156, disagree: 8 },
  },
  {
    id: "KOI-4",
    name: "Proxima b",
    orbitalPeriod: 11.2,
    transitDepth: 0.0006,
    radius: 1.17,
    distance: 4.2,
    starTemp: 3042,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.91,
    lightCurvePrediction: "Candidate",
    lightCurveConfidence: 0.78,
    votes: { agree: 567, disagree: 89 },
  },
  {
    id: "KOI-5",
    name: "TOI-700d",
    orbitalPeriod: 37.4,
    transitDepth: 0.0011,
    radius: 1.19,
    distance: 101.4,
    starTemp: 3480,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.94,
    lightCurvePrediction: "Confirmed",
    lightCurveConfidence: 0.89,
    votes: { agree: 723, disagree: 34 },
  },
  {
    id: "KOI-6",
    name: "LHS-1140b",
    orbitalPeriod: 24.7,
    transitDepth: 0.0019,
    radius: 1.73,
    distance: 40.7,
    starTemp: 3131,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.97,
    lightCurvePrediction: "Confirmed",
    lightCurveConfidence: 0.95,
    votes: { agree: 989, disagree: 21 },
  },
  {
    id: "KOI-7",
    name: "GJ-367b",
    orbitalPeriod: 0.32,
    transitDepth: 0.0005,
    radius: 0.72,
    distance: 30.9,
    starTemp: 3430,
    tabularPrediction: "Confirmed",
    tabularConfidence: 0.88,
    lightCurvePrediction: "Confirmed",
    lightCurveConfidence: 0.85,
    votes: { agree: 445, disagree: 67 },
  },
  {
    id: "KOI-8",
    name: "HD-219134b",
    orbitalPeriod: 3.09,
    transitDepth: 0.0007,
    radius: 1.60,
    distance: 21.3,
    starTemp: 4730,
    tabularPrediction: "Candidate",
    tabularConfidence: 0.73,
    lightCurvePrediction: "Candidate",
    lightCurveConfidence: 0.68,
    votes: { agree: 234, disagree: 156 },
  },
];

// Generate light curve data points
export const generateLightCurve = (transitDepth: number, period: number) => {
  const points = [];
  const numPoints = 150;
  
  for (let i = 0; i < numPoints; i++) {
    const phase = (i / numPoints) * 2;
    let flux = 1.0;
    
    // Add transit dip
    if (phase > 0.4 && phase < 0.6) {
      const transitProgress = (phase - 0.4) / 0.2;
      flux = 1.0 - transitDepth * Math.sin(transitProgress * Math.PI);
    }
    
    // Add noise
    flux += (Math.random() - 0.5) * 0.0002;
    
    points.push({
      phase,
      flux,
      time: (period * phase) / 2,
    });
  }
  
  return points;
};
