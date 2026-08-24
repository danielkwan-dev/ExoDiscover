import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ShieldCheck, Telescope, Waypoints } from "lucide-react";

import { api } from "../lib/api";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import heroImage from "../assets/hero-exoplanet.jpg";

const Index = () => {
  // Fetched, not hardcoded: if the API is down these read "—" rather than
  // showing numbers the model never produced.
  const { data } = useQuery({ queryKey: ["metrics"], queryFn: api.metrics, retry: false });

  const headline = data
    ? [
        { label: "ROC-AUC on held-out stars", value: data.test.roc_auc.toFixed(3) },
        { label: "Training KOIs", value: data.model.n_train_rows.toLocaleString() },
        { label: "Distinct host stars", value: data.model.n_train_stars.toLocaleString() },
      ]
    : [
        { label: "ROC-AUC on held-out stars", value: "—" },
        { label: "Training KOIs", value: "—" },
        { label: "Distinct host stars", value: "—" },
      ];

  return (
    <div>
      <section className="relative overflow-hidden">
        <img
          src={heroImage}
          alt=""
          aria-hidden
          className="absolute inset-0 h-full w-full object-cover opacity-20"
        />
        <div className="container relative px-4 py-24">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full glass-card px-4 py-2">
              <Telescope className="h-4 w-4 text-cosmic-purple" />
              <span className="text-sm">NASA Space Apps 2025 · A World Away</span>
            </div>
            <h1 className="text-5xl font-bold leading-tight md:text-6xl">
              Hunting exoplanets with{" "}
              <span className="text-cosmic-purple">honest</span> machine learning
            </h1>
            <p className="mt-6 text-lg text-muted-foreground">
              A classifier over NASA's Kepler catalog that separates real planets from
              false positives — built so that the number it reports is the number you'd
              actually get on new data.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Button asChild size="lg">
                <Link to="/predict">
                  Classify a signal <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="secondary">
                <Link to="/discoveries">See the candidate shortlist</Link>
              </Button>
            </div>
          </div>

          <div className="mt-16 grid max-w-3xl gap-4 sm:grid-cols-3">
            {headline.map((s) => (
              <div key={s.label} className="glass-card rounded-xl p-5">
                <p className="text-3xl font-bold text-cosmic-purple">{s.value}</p>
                <p className="mt-1 text-xs text-muted-foreground">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="container px-4 py-20">
        <h2 className="text-3xl font-bold">What makes this different</h2>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Most exoplanet classifiers report accuracy above 97%. Almost all of them are
          measuring something other than what they claim.
        </p>

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          <Card className="glass-card">
            <CardHeader>
              <ShieldCheck className="h-6 w-6 text-cosmic-cyan" />
              <CardTitle className="mt-3 text-lg">The labels are quarantined</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                The Kepler table ships <code className="font-mono text-xs">koi_score</code>{" "}
                and four <code className="font-mono text-xs">koi_fpflag_*</code> columns —
                these <em>are</em> the vetting pipeline's verdict. A model given them
                reproduces the answer instead of learning the physics. A test fails the
                build if one ever reaches the feature matrix.
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <Waypoints className="h-6 w-6 text-cosmic-cyan" />
              <CardTitle className="mt-3 text-lg">Whole stars are held out</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                9,564 candidate signals come from only 8,214 stars — up to seven on one
                star, sharing stellar parameters and photometry. Splitting rows at random
                puts siblings on both sides. Every split here is grouped by host star.
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <Telescope className="h-6 w-6 text-cosmic-cyan" />
              <CardTitle className="mt-3 text-lg">Physics, not just columns</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Features include the transit duration implied by Kepler's third law and the
                depth implied by the planet-to-star radius ratio. Eclipsing binaries and
                blended stars break those relationships in characteristic ways.
              </CardDescription>
            </CardContent>
          </Card>
        </div>

        <div className="mt-10">
          <Button asChild variant="secondary">
            <Link to="/model">
              See the full ablation and calibration
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
};

export default Index;
