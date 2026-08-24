/**
 * Typed client for the ExoDiscover API.
 *
 * Every screen in this app reads from these functions. There is no mock data:
 * if the API is down, the UI says so rather than rendering invented numbers.
 */

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface Contribution {
  feature: string;
  value: number;
  shap: number;
}

export interface Prediction {
  label: string;
  probability: number;
  band: string;
  contributions: Contribution[];
}

export interface KOIInput {
  koi_period: number;
  koi_depth: number;
  koi_duration: number;
  koi_prad: number;
  koi_srad: number;
  koi_slogg: number;
  koi_steff: number;
  koi_impact: number;
  koi_model_snr?: number | null;
  n_kois_on_star: number;
}

export interface AblationRow {
  setup?: string;
  framing?: string;
  target?: string;
  note?: string;
  roc_auc: number;
  pr_auc: number;
  brier: number;
  precision_at_50?: number;
}

export interface LadderRow {
  name: string;
  pr_auc: number;
  roc_auc: number;
  brier: number;
  /** Fold-to-fold spread. Without it, ranking rows this close is meaningless. */
  pr_auc_std: number;
  roc_auc_std: number;
  brier_std: number;
}

export interface Calibration {
  chosen: string;
  brier_by_method: Record<string, number>;
  distinct_probabilities: Record<string, number>;
  n_calibration: number;
  n_validation: number;
}

export interface Metrics {
  model: {
    name: string;
    version: string;
    trained_at: string;
    framing: string;
    n_train_rows: number;
    n_train_stars: number;
  };
  features: string[];
  test: {
    roc_auc: number;
    pr_auc: number;
    brier: number;
    precision_at_50: number;
    confusion_matrix: number[][];
    n_test: number;
    /** 95% CI from a bootstrap that resamples host stars, not rows. */
    ci95: { roc_auc: [number, number]; pr_auc: [number, number]; n_resamples: number };
  };
  calibration: Calibration;
  ladder: LadderRow[];
  ablation: { leakage: AblationRow[]; framing: AblationRow[] };
  transfer: {
    in_domain: { n: number; roc_auc: number; pr_auc: number; base_rate: number };
    zero_shot: { n: number; roc_auc: number; pr_auc: number; base_rate: number };
    roc_auc_drop: number;
  };
  importance: { feature: string; mean_abs_shap: number }[];
  reliability: { bin_centers: number[]; observed: number[]; counts: number[] };
}

export interface Candidate {
  kepoi_name: string;
  kepler_name: string | null;
  kepid: number;
  probability: number;
  /** Uncalibrated model score. Breaks ties where isotonic calibration saturates. */
  score: number;
  rank: number;
  top_reasons: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${BASE}. Is it running? (\`make serve\`)`,
      0,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>("/health"),

  metrics: () => request<Metrics>("/metrics"),

  predict: (payload: KOIInput) =>
    request<Prediction>("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  discoveries: (limit = 25) =>
    request<{ n: number; candidates: Candidate[] }>(`/discoveries?limit=${limit}`),

  predictBatch: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ n_rows: number; results: (Prediction & { row: number })[] }>(
      "/predict/batch",
      { method: "POST", body: form },
    );
  },
};

/** Kepler-10 b, as a worked example the user can start from. */
export const KEPLER10B: KOIInput = {
  koi_period: 0.837495,
  koi_depth: 152.0,
  koi_duration: 1.811,
  koi_prad: 1.47,
  koi_srad: 1.065,
  koi_slogg: 4.35,
  koi_steff: 5627,
  koi_impact: 0.3,
  koi_model_snr: 25,
  n_kois_on_star: 2,
};
