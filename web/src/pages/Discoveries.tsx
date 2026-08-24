import { useQuery } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";

import { api } from "../lib/api";
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

/**
 * The project's actual output: unvetted KOI candidates the model believes are
 * real planets. These rows were never in training — the classifier learns from
 * resolved dispositions only, so every candidate here is genuinely unseen.
 */
const Discoveries = () => {
  const { data, isPending, error } = useQuery({
    queryKey: ["discoveries"],
    queryFn: () => api.discoveries(50),
  });

  return (
    <div className="container px-4 py-12">
      <div className="mx-auto max-w-5xl">
        <div className="mb-2 inline-flex items-center gap-2 rounded-full glass-card px-4 py-2">
          <Sparkles className="h-4 w-4 text-cosmic-cyan" />
          <span className="text-sm">Never seen during training</span>
        </div>
        <h1 className="text-3xl font-bold">Candidate shortlist</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          The Kepler catalog holds 1,979 KOIs that passed automated vetting but were never
          confirmed. The classifier is trained only on resolved dispositions, so these are
          genuinely unseen. Ranked by calibrated probability, with the features that drove
          each score.
        </p>

        {isPending && <Loading label="Loading candidate ranking" />}
        {error && <Failed error={error as Error} />}

        {data && (
          <Card className="glass-card mt-8">
            <CardHeader>
              <CardTitle className="text-lg">Top {data.n} by probability</CardTitle>
              <CardDescription>
                A vetting team works a finite shortlist, so ranking quality matters more
                than overall accuracy. Isotonic calibration saturates at the top, so
                several rows share a probability — the raw score is what orders them.
              </CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead>KOI</TableHead>
                    <TableHead className="w-28">Probability</TableHead>
                    <TableHead className="w-24">Score</TableHead>
                    <TableHead>Top contributing features</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.candidates.map((c) => (
                    <TableRow key={c.kepoi_name}>
                      <TableCell className="text-muted-foreground">{c.rank}</TableCell>
                      <TableCell className="font-medium">
                        {c.kepoi_name}
                        <span className="block text-xs text-muted-foreground">
                          KIC {c.kepid}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="font-semibold text-cosmic-purple">
                          {(c.probability * 100).toFixed(1)}%
                        </span>
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {c.score.toFixed(4)}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {c.top_reasons}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Discoveries;
