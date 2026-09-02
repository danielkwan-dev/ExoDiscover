import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";

/**
 * The overview reads as the front page of a report, not a product page.
 *
 * Deliberately absent: a hero image behind the headline, a pill badge above it,
 * a three-column grid of icon-topped cards, and any gradient. Those are the
 * house style of generated landing pages, and on a measurement tool they
 * undercut the thing being measured.
 */

const Dash = () => <span className="text-muted-foreground">&mdash;</span>;

const pct = (x: number | undefined) =>
  x === undefined ? <Dash /> : `${(x * 100).toFixed(1)}%`;

const num = (x: number | undefined, dp = 3) =>
  x === undefined ? <Dash /> : x.toFixed(dp);

const Index = () => {
  // Fetched, never hardcoded: with the API down these read as a dash rather
  // than showing numbers the model did not produce.
  const { data } = useQuery({ queryKey: ["metrics"], queryFn: api.metrics, retry: false });
  const tess = data?.transfer.zero_shot;
  const kepler = data?.transfer.in_domain;

  return (
    <div className="container max-w-3xl px-4 py-16">
      <header className="border-b border-border pb-10">
        <p className="label">NASA Space Apps 2025 &middot; A World Away</p>
        <h1 className="mt-3 text-4xl leading-tight md:text-5xl">
          Trained on Kepler.
          <br />
          Tested on TESS.
        </h1>
        <p className="mt-5 text-muted-foreground">
          A transit classifier that learns from one telescope&rsquo;s catalogue and is
          then scored on a different telescope&rsquo;s &mdash; objects it has never
          seen, from a mission with a redder bandpass, shorter baselines and larger
          pixels. The figure below is what it does on data it was not trained on.
        </p>
      </header>

      <section className="mt-10">
        <h2 className="text-xl">Zero-shot result</h2>
        <table className="mt-4 w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label py-2 font-normal">Metric</th>
              <th className="label py-2 text-right font-normal">TESS (unseen)</th>
              <th className="label py-2 text-right font-normal">Kepler (reference)</th>
            </tr>
          </thead>
          <tbody className="figure">
            <tr className="border-b border-border/60">
              <td className="py-2 font-sans">Accuracy</td>
              <td className="py-2 text-right font-semibold text-accent">
                {pct(tess?.accuracy)}
              </td>
              <td className="py-2 text-right text-muted-foreground">
                {pct(kepler?.accuracy)}
              </td>
            </tr>
            <tr className="border-b border-border/60">
              <td className="py-2 font-sans text-muted-foreground">
                majority-class baseline
              </td>
              <td className="py-2 text-right text-muted-foreground">
                {pct(tess?.majority_baseline)}
              </td>
              <td className="py-2 text-right text-muted-foreground">
                {pct(kepler?.majority_baseline)}
              </td>
            </tr>
            <tr className="border-b border-border/60">
              <td className="py-2 font-sans">ROC-AUC</td>
              <td className="py-2 text-right">{num(tess?.roc_auc)}</td>
              <td className="py-2 text-right text-muted-foreground">
                {num(kepler?.roc_auc)}
              </td>
            </tr>
            <tr className="border-b border-border/60">
              <td className="py-2 font-sans">Brier</td>
              <td className="py-2 text-right">{num(tess?.brier)}</td>
              <td className="py-2 text-right text-muted-foreground">
                {num(kepler?.brier)}
              </td>
            </tr>
            <tr>
              <td className="py-2 font-sans text-muted-foreground">objects scored</td>
              <td className="py-2 text-right text-muted-foreground">
                {tess ? tess.n.toLocaleString() : <Dash />}
              </td>
              <td className="py-2 text-right text-muted-foreground">
                {kepler ? kepler.n.toLocaleString() : <Dash />}
              </td>
            </tr>
          </tbody>
        </table>
        <p className="mt-4 text-sm text-muted-foreground">
          Read each accuracy against its baseline. Kepler&rsquo;s held-out slice is 63%
          false positives while TESS is close to balanced, so the two are not directly
          comparable &mdash; the base-rate-free comparison is ROC-AUC. The ranking
          largely survives the change of telescope; the calibration does not, and the
          Brier score more than doubles.
        </p>
      </section>

      <section className="mt-12 border-t border-border pt-10">
        <h2 className="text-xl">Why the catalogue needs handling with care</h2>
        <p className="mt-4 text-muted-foreground">
          The Kepler table ships the answer key. <code>koi_score</code> is the
          automated vetting pipeline&rsquo;s own confidence in its verdict, and four{" "}
          <code>koi_fpflag_*</code> columns are its individual false-positive
          decisions. A model given them reaches ROC-AUC 0.9999 &mdash; it has learned
          to read the label rather than the physics.
        </p>

        <dl className="mt-8 space-y-6">
          <div>
            <dt className="font-medium">The verdict columns are quarantined</dt>
            <dd className="mt-1 text-sm text-muted-foreground">
              Dropped on the way into feature construction and asserted absent on the
              way out. A second guard reads values rather than column names, flagging
              any feature whose rank correlation with a vetting column exceeds 0.80 -
              a threshold set by measurement rather than taste.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Whole stars are held out</dt>
            <dd className="mt-1 text-sm text-muted-foreground">
              9,564 signals come from only 8,214 stars, up to seven on one star,
              sharing stellar parameters and photometry. Splitting rows at random puts
              siblings on both sides, so every split here is grouped by host star.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Features encode physics, not just columns</dt>
            <dd className="mt-1 text-sm text-muted-foreground">
              The transit duration implied by Kepler&rsquo;s third law, and the depth
              implied by the planet-to-star radius ratio. Eclipsing binaries and
              blended stars break those relationships in characteristic ways &mdash;
              and unlike instrument character, physics transfers between missions.
            </dd>
          </div>
        </dl>
      </section>

      <nav className="mt-12 flex flex-wrap gap-x-8 gap-y-2 border-t border-border pt-6 text-sm">
        <Link className="underline underline-offset-4 hover:text-accent" to="/predict">
          Classify a signal
        </Link>
        <Link className="underline underline-offset-4 hover:text-accent" to="/discoveries">
          Candidate shortlist
        </Link>
        <Link className="underline underline-offset-4 hover:text-accent" to="/model">
          Ablations and calibration
        </Link>
      </nav>
    </div>
  );
};

export default Index;
