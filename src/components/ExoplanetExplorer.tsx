import { useState } from "react";
import { mockExoplanets, ExoplanetData } from "../data/mockData";
import ExoplanetCard from "./ExoplanetCard";
import FilterPanel from "./FilterPanel";
import LightCurveChart from "./LightCurveChart";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";
import { Badge } from "../components/ui/badge";
import { Brain, Sparkles } from "lucide-react";
import React from "react";

const ExoplanetExplorer = () => {
  const [orbitalPeriod, setOrbitalPeriod] = useState<number[]>([0, 50]);
  const [transitDepth, setTransitDepth] = useState<number[]>([0, 0.003]);
  const [radius, setRadius] = useState<number[]>([0, 3]);
  const [selectedPlanet, setSelectedPlanet] = useState<ExoplanetData | null>(null);

  const filteredPlanets = mockExoplanets.filter((planet) => {
    return (
      planet.orbitalPeriod >= orbitalPeriod[0] &&
      planet.orbitalPeriod <= orbitalPeriod[1] &&
      planet.transitDepth >= transitDepth[0] &&
      planet.transitDepth <= transitDepth[1] &&
      planet.radius >= radius[0] &&
      planet.radius <= radius[1]
    );
  });

  return (
    <section id="explorer" className="py-20 relative">
      <div className="container px-4">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4">
              <Brain className="w-4 h-4 text-cosmic-purple" />
              <span className="text-sm font-medium">Dual AI Classification</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Exoplanet <span className="text-cosmic-purple">Explorer</span>
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Filter and explore confirmed exoplanets with real-time predictions from both tabular ML and light curve deep learning models.
            </p>
          </div>

          {/* Main Content */}
          <div className="grid lg:grid-cols-4 gap-6">
            {/* Filters */}
            <div className="lg:col-span-1">
              <FilterPanel
                orbitalPeriod={orbitalPeriod}
                setOrbitalPeriod={setOrbitalPeriod}
                transitDepth={transitDepth}
                setTransitDepth={setTransitDepth}
                radius={radius}
                setRadius={setRadius}
              />
            </div>

            {/* Results */}
            <div className="lg:col-span-3">
              <div className="mb-6 flex items-center justify-between">
                <p className="text-muted-foreground">
                  Found <span className="text-cosmic-purple font-semibold">{filteredPlanets.length}</span> exoplanet{filteredPlanets.length !== 1 ? "s" : ""}
                </p>
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cosmic-cyan" />
                  <span className="text-sm text-muted-foreground">Live predictions</span>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {filteredPlanets.map((planet) => (
                  <ExoplanetCard
                    key={planet.id}
                    data={planet}
                    onViewDetails={() => setSelectedPlanet(planet)}
                  />
                ))}
              </div>

              {filteredPlanets.length === 0 && (
                <div className="text-center py-20 glass-card rounded-lg">
                  <p className="text-muted-foreground text-lg">
                    No exoplanets match your current filters. Try adjusting the ranges above.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Details Dialog */}
      <Dialog open={!!selectedPlanet} onOpenChange={() => setSelectedPlanet(null)}>
        <DialogContent className="glass-card max-w-4xl max-h-[90vh] overflow-y-auto">
          {selectedPlanet && (
            <>
              <DialogHeader>
                <DialogTitle className="text-2xl text-cosmic-purple">
                  {selectedPlanet.name}
                </DialogTitle>
                <DialogDescription>
                  Detailed analysis and predictions for {selectedPlanet.id}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6 mt-4">
                {/* Properties Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: "Orbital Period", value: `${selectedPlanet.orbitalPeriod.toFixed(2)} days` },
                    { label: "Transit Depth", value: selectedPlanet.transitDepth.toFixed(4) },
                    { label: "Planet Radius", value: `${selectedPlanet.radius.toFixed(2)} R⊕` },
                    { label: "Distance", value: `${selectedPlanet.distance} ly` },
                    { label: "Star Temperature", value: `${selectedPlanet.starTemp} K` },
                  ].map((prop, i) => (
                    <div key={i} className="p-3 rounded-lg bg-muted/50">
                      <p className="text-xs text-muted-foreground mb-1">{prop.label}</p>
                      <p className="font-semibold">{prop.value}</p>
                    </div>
                  ))}
                </div>

                {/* AI Predictions */}
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="glass-card p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">ExoML Classifier</h4>
                      <Badge className="bg-cosmic-purple/20 text-cosmic-purple border-cosmic-purple/30">
                        Tabular ML
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Prediction:</span>
                        <span className="font-semibold text-cosmic-cyan">
                          {selectedPlanet.tabularPrediction}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Confidence:</span>
                        <span className="font-semibold">
                          {(selectedPlanet.tabularConfidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="glass-card p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">Light Curve Vision</h4>
                      <Badge className="bg-cosmic-blue/20 text-cosmic-blue border-cosmic-blue/30">
                        Deep Learning
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Prediction:</span>
                        <span className="font-semibold text-cosmic-cyan">
                          {selectedPlanet.lightCurvePrediction}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Confidence:</span>
                        <span className="font-semibold">
                          {(selectedPlanet.lightCurveConfidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Light Curve */}
                <LightCurveChart
                  transitDepth={selectedPlanet.transitDepth}
                  period={selectedPlanet.orbitalPeriod}
                  planetName={selectedPlanet.name}
                />

                {/* Citizen Science */}
                <div className="glass-card p-4">
                  <h4 className="font-semibold mb-3">Community Validation</h4>
                  <div className="flex items-center justify-between">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-400">
                        {selectedPlanet.votes.agree}
                      </p>
                      <p className="text-sm text-muted-foreground">Agree</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-red-400">
                        {selectedPlanet.votes.disagree}
                      </p>
                      <p className="text-sm text-muted-foreground">Disagree</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-cosmic-purple">
                        {(
                          (selectedPlanet.votes.agree /
                            (selectedPlanet.votes.agree + selectedPlanet.votes.disagree)) *
                          100
                        ).toFixed(1)}%
                      </p>
                      <p className="text-sm text-muted-foreground">Agreement</p>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
};

export default ExoplanetExplorer;
