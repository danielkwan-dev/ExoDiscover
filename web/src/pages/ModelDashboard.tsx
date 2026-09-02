import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api, type AblationRow } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { Failed, Loading } from "../components/ApiState";

const pct = (n: number) => n.toFixed(4);

const Stat = ({ label, value, hint }: { label: string; value: string; hint?: string }) => (
  <Card className="panel">
    <CardContent className="pt-6">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-3xl font-bold text-accent">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </CardContent>
  </Card>
);

const AblationTable = ({ rows, keyField }: { rows: AblationRow[]; keyField: "setup" | "framing" }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>{keyField === "setup" ? "Setup" : "Task framing"}</TableHead>
        <TableHead className="w-24">ROC-AUC</TableHead>
        <TableHead className="w-24">PR-AUC</TableHead>
        <TableHead className="w-24">Brier</TableHead>
        <TableHead>Notes</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {rows.map((r) => (
        <TableRow key={r[keyField]}>
          <TableCell className="font-mono text-xs">{r[keyField]}</TableCell>
          <TableCell>{pct(r.roc_auc)}</TableCell>
          <TableCell>{pct(r.pr_auc)}</TableCell>
          <TableCell>{pct(r.brier)}</TableCell>
          <TableCell className="text-xs text-muted-foreground">{r.note ?? r.target}</TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const ModelDashboard = () => {
  const { data, isPending, error } = useQuery({ queryKey: ["metrics"], queryFn: api.metrics });

  if (isPending) return <div className="container px-4 py-12"><Loading label="Loading metrics" /></div>;
  if (error)
    return (
      <div className="container px-4 py-12">
        <Failed error={error as Error} />
      </div>
    );

  const reliability = data.reliability.bin_centers.map((c, i) => ({
    predicted: Number(c.toFixed(3)),
    observed: data.reliability.observed[i],
    ideal: Number(c.toFixed(3)),
  }));

  const importance = data.importance.slice(0, 12).map((d) => ({
    feature: d.feature,
    value: d.mean_abs_shap,
  }));

  return (
    <div className="container px-4 py-12">
      <div className="mx-auto max-w-6xl space-y-8">
        <div>
          <h1 className="text-3xl font-bold">Model performance</h1>
          <p className="mt-2 text-muted-foreground">
            {data.model.name} · trained on {data.model.n_train_rows.toLocaleString()} KOIs
            across {data.model.n_train_stars.toLocaleString()} stars · evaluated on{" "}
            {data.test.n_test.toLocaleString()} rows from stars never seen in training.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="ROC-AUC"
            value={pct(data.test.roc_auc)}
            hint={
              data.test.ci95
                ? `95% CI ${pct(data.test.ci95.roc_auc[0])}–${pct(data.test.ci95.roc_auc[1])}`
                : "held-out stars"
            }
          />
          <Stat
            label="PR-AUC"
            value={pct(data.test.pr_auc)}
            hint={
              data.test.ci95
                ? `95% CI ${pct(data.test.ci95.pr_auc[0])}–${pct(data.test.ci95.pr_auc[1])}`
                : "ranking quality"
            }
          />
          <Stat label="Brier" value={pct(data.test.brier)} hint="lower is better" />
          <Stat
            label="Precision@50"
            value={pct(data.test.precision_at_50)}
            hint="top of the shortlist"
          />
        </div>

        <p className="text-xs text-muted-foreground">
          Confidence intervals come from a bootstrap that resamples <em>host stars</em>, not
          rows — sibling KOIs on one star are not independent draws, so resampling rows would
          report an interval narrower than the data supports.
        </p>

        <Card className="panel">
          <CardHeader>
            <CardTitle>Leakage ablation</CardTitle>
            <CardDescription>
              One variable changed at a time. The first row reproduces the original
              methodology and is the number this project exists to correct.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <AblationTable rows={data.ablation.leakage} keyField="setup" />
          </CardContent>
        </Card>

        <Card className="panel">
          <CardHeader>
            <CardTitle>Task framing</CardTitle>
            <CardDescription>
              All three framings scored on the one decision they share — planet-like versus
              false positive — so they are directly comparable.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <AblationTable rows={data.ablation.framing} keyField="framing" />
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="panel">
            <CardHeader>
              <CardTitle>Calibration</CardTitle>
              <CardDescription>
                Predicted probability against observed frequency; the dashed diagonal is
                perfect calibration.
                {data.calibration && (
                  <>
                    {" "}
                    Isotonic and sigmoid were both fitted and scored on a third,
                    star-disjoint slice — <strong>{data.calibration.chosen}</strong> won
                    (Brier{" "}
                    {Object.entries(data.calibration.brier_by_method)
                      .map(([m, v]) => `${m} ${v.toFixed(4)}`)
                      .join(" vs ")}
                    ).
                  </>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={reliability}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="predicted" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: 8,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="ideal"
                    stroke="hsl(var(--muted-foreground))"
                    strokeDasharray="4 4"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="observed"
                    stroke="hsl(var(--accent))"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="panel">
            <CardHeader>
              <CardTitle>Feature importance</CardTitle>
              <CardDescription>Mean absolute SHAP value across the test set.</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={importance} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={120}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: 8,
                    }}
                  />
                  <Bar dataKey="value" fill="hsl(var(--accent))" radius={3} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        <Card className="panel">
          <CardHeader>
            <CardTitle>Cross-mission generalisation</CardTitle>
            <CardDescription>
              Trained on Kepler, evaluated zero-shot on TESS objects using only features
              both catalogs express.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-3">
            <div>
              <p className="text-xs uppercase text-muted-foreground">Kepler (in domain)</p>
              <p className="text-2xl font-bold">{pct(data.transfer.in_domain.roc_auc)}</p>
              <p className="text-xs text-muted-foreground">
                n = {data.transfer.in_domain.n}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase text-muted-foreground">TESS (zero-shot)</p>
              <p className="text-2xl font-bold">{pct(data.transfer.zero_shot.roc_auc)}</p>
              <p className="text-xs text-muted-foreground">n = {data.transfer.zero_shot.n}</p>
            </div>
            <div>
              <p className="text-xs uppercase text-muted-foreground">ROC-AUC drop</p>
              <p className="figure text-2xl font-bold text-accent">
                {data.transfer.roc_auc_drop.toFixed(4)}
              </p>
              <p className="text-xs text-muted-foreground">domain shift</p>
            </div>
          </CardContent>
        </Card>

        <Card className="panel">
          <CardHeader>
            <CardTitle>Model ladder</CardTitle>
            <CardDescription>
              Every family scored identically under star-grouped CV, so the gain over the
              baseline is attributable to the model rather than to how it was measured. The
              ± column is the fold-to-fold spread: the boosted families sit inside each
              other's, so the ordering among them is not meaningful.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>PR-AUC</TableHead>
                  <TableHead>ROC-AUC</TableHead>
                  <TableHead>Brier</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.ladder.map((row) => (
                  <TableRow key={row.name}>
                    <TableCell className="font-mono text-xs">{row.name}</TableCell>
                    <TableCell>
                      {pct(row.pr_auc)}
                      {row.pr_auc_std != null && (
                        <span className="ml-1 text-xs text-muted-foreground">
                          ± {row.pr_auc_std.toFixed(4)}
                        </span>
                      )}
                    </TableCell>
                    <TableCell>{pct(row.roc_auc)}</TableCell>
                    <TableCell>{pct(row.brier)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ModelDashboard;
