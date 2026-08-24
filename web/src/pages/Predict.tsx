import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api, KEPLER10B, type KOIInput } from "../lib/api";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Badge } from "../components/ui/badge";
import { Failed } from "../components/ApiState";

/** Field metadata: label, unit, and step, so the form is self-documenting. */
const FIELDS: { key: keyof KOIInput; label: string; unit: string; step: number }[] = [
  { key: "koi_period", label: "Orbital period", unit: "days", step: 0.001 },
  { key: "koi_depth", label: "Transit depth", unit: "ppm", step: 1 },
  { key: "koi_duration", label: "Transit duration", unit: "hours", step: 0.01 },
  { key: "koi_prad", label: "Planet radius", unit: "R⊕", step: 0.01 },
  { key: "koi_srad", label: "Stellar radius", unit: "R☉", step: 0.01 },
  { key: "koi_slogg", label: "Stellar log g", unit: "log₁₀ cgs", step: 0.01 },
  { key: "koi_steff", label: "Stellar temperature", unit: "K", step: 1 },
  { key: "koi_impact", label: "Impact parameter", unit: "0 = central", step: 0.01 },
  { key: "koi_model_snr", label: "Transit SNR", unit: "ratio", step: 0.1 },
  { key: "n_kois_on_star", label: "KOIs on this star", unit: "count", step: 1 },
];

const BAND_STYLE: Record<string, string> = {
  confirmed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  "needs vetting": "bg-amber-500/15 text-amber-300 border-amber-500/30",
  "false positive": "bg-rose-500/15 text-rose-300 border-rose-500/30",
};

const Predict = () => {
  const [form, setForm] = useState<KOIInput>(KEPLER10B);
  const mutation = useMutation({ mutationFn: api.predict });

  const waterfall = (mutation.data?.contributions ?? [])
    .map((c) => ({ name: c.feature, shap: c.shap, value: c.value }))
    .reverse();

  return (
    <div className="container px-4 py-12">
      <div className="mx-auto max-w-6xl">
        <h1 className="text-3xl font-bold">Classify a transit signal</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Enter the measured parameters of a transit. The model returns a calibrated
          probability — one that means what it says, so a 0.7 is right about 70% of the
          time — together with the features that drove it.
        </p>

        <div className="mt-8 grid gap-6 lg:grid-cols-5">
          <Card className="glass-card lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-lg">Parameters</CardTitle>
              <CardDescription>Pre-filled with Kepler-10 b, a confirmed planet.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {FIELDS.map((field) => (
                <div key={field.key} className="grid grid-cols-2 items-center gap-3">
                  <Label htmlFor={field.key} className="text-sm">
                    {field.label}
                    <span className="ml-1 text-xs text-muted-foreground">({field.unit})</span>
                  </Label>
                  <Input
                    id={field.key}
                    type="number"
                    step={field.step}
                    value={String(form[field.key] ?? "")}
                    onChange={(e) =>
                      setForm({ ...form, [field.key]: Number(e.target.value) })
                    }
                  />
                </div>
              ))}
              <div className="flex gap-2 pt-2">
                <Button
                  className="flex-1"
                  onClick={() => mutation.mutate(form)}
                  disabled={mutation.isPending}
                >
                  {mutation.isPending ? "Classifying…" : "Classify"}
                </Button>
                <Button variant="secondary" onClick={() => setForm(KEPLER10B)}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="lg:col-span-3">
            {mutation.isError && <Failed error={mutation.error as Error} />}

            {mutation.data && (
              <div className="space-y-6">
                <Card className="glass-card">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Result</CardTitle>
                      <Badge className={BAND_STYLE[mutation.data.band]}>
                        {mutation.data.band}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-baseline gap-3">
                      <span className="text-5xl font-bold text-cosmic-purple">
                        {(mutation.data.probability * 100).toFixed(1)}%
                      </span>
                      <span className="text-muted-foreground">
                        calibrated probability this is a planet
                      </span>
                    </div>
                    <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cosmic-blue to-cosmic-purple"
                        style={{ width: `${mutation.data.probability * 100}%` }}
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-lg">Why</CardTitle>
                    <CardDescription>
                      SHAP contributions. Bars to the right push toward "planet"; bars to
                      the left push toward "false positive".
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={waterfall} layout="vertical" margin={{ left: 40 }}>
                        <XAxis type="number" tick={{ fontSize: 11 }} />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={130}
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "hsl(240 18% 10%)",
                            border: "1px solid hsl(240 15% 20%)",
                            borderRadius: 8,
                          }}
                          formatter={(v: number, _n, item) => [
                            `${v >= 0 ? "+" : ""}${v.toFixed(3)} (value ${item.payload.value.toPrecision(3)})`,
                            "SHAP",
                          ]}
                        />
                        <ReferenceLine x={0} stroke="hsl(215 20% 45%)" />
                        <Bar dataKey="shap" radius={3}>
                          {waterfall.map((d, i) => (
                            <Cell
                              key={i}
                              fill={d.shap >= 0 ? "hsl(158 64% 52%)" : "hsl(350 75% 60%)"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>
            )}

            {!mutation.data && !mutation.isError && (
              <Card className="glass-card flex h-full items-center justify-center">
                <CardContent className="py-20 text-center text-muted-foreground">
                  Adjust the parameters and press Classify.
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
