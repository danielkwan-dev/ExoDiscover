import { Card } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { ThumbsUp, ThumbsDown, TrendingUp, Eye } from "lucide-react";
import { ExoplanetData } from "../data/mockData";
import { useState } from "react";
import React from "react";

interface ExoplanetCardProps {
  data: ExoplanetData;
  onViewDetails: () => void;
}

const ExoplanetCard = ({ data, onViewDetails }: ExoplanetCardProps) => {
  const [votes, setVotes] = useState(data.votes);
  const [userVote, setUserVote] = useState<"agree" | "disagree" | null>(null);

  const handleVote = (vote: "agree" | "disagree") => {
    if (userVote === vote) {
      setVotes((prev) => ({
        ...prev,
        [vote]: prev[vote] - 1,
      }));
      setUserVote(null);
    } else {
      setVotes((prev) => ({
        agree: vote === "agree" ? prev.agree + 1 : userVote === "agree" ? prev.agree - 1 : prev.agree,
        disagree: vote === "disagree" ? prev.disagree + 1 : userVote === "disagree" ? prev.disagree - 1 : prev.disagree,
      }));
      setUserVote(vote);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "text-green-400";
    if (confidence >= 0.8) return "text-cosmic-cyan";
    if (confidence >= 0.7) return "text-yellow-400";
    return "text-orange-400";
  };

  const getPredictionBadge = (prediction: string) => {
    const colors = {
      Confirmed: "bg-green-500/20 text-green-400 border-green-500/30",
      Candidate: "bg-cosmic-blue/20 text-cosmic-blue border-cosmic-blue/30",
      "False Positive": "bg-red-500/20 text-red-400 border-red-500/30",
    };
    return colors[prediction as keyof typeof colors];
  };

  return (
    <Card className="glass-card p-6 space-y-4 hover:star-glow transition-all duration-300 group">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-xl font-bold text-cosmic-purple group-hover:text-cosmic-blue transition-colors">
            {data.name}
          </h3>
          <p className="text-sm text-muted-foreground">{data.id}</p>
        </div>
        <Badge className={getPredictionBadge(data.tabularPrediction)}>
          {data.tabularPrediction}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-muted-foreground">Orbital Period</p>
          <p className="font-semibold">{data.orbitalPeriod.toFixed(2)} days</p>
        </div>
        <div>
          <p className="text-muted-foreground">Distance</p>
          <p className="font-semibold">{data.distance} ly</p>
        </div>
        <div>
          <p className="text-muted-foreground">Radius</p>
          <p className="font-semibold">{data.radius.toFixed(2)} R⊕</p>
        </div>
        <div>
          <p className="text-muted-foreground">Star Temp</p>
          <p className="font-semibold">{data.starTemp} K</p>
        </div>
      </div>

      <div className="space-y-2 pt-2 border-t border-border">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">ExoML Classifier</span>
          <span className={`text-sm font-semibold ${getConfidenceColor(data.tabularConfidence)}`}>
            {(data.tabularConfidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Light Curve Vision</span>
          <span className={`text-sm font-semibold ${getConfidenceColor(data.lightCurveConfidence)}`}>
            {(data.lightCurveConfidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2 pt-2">
        <Button
          size="sm"
          variant={userVote === "agree" ? "default" : "outline"}
          className={userVote === "agree" ? "bg-green-500/20 border-green-500/30" : ""}
          onClick={() => handleVote("agree")}
        >
          <ThumbsUp className="w-4 h-4 mr-1" />
          {votes.agree}
        </Button>
        <Button
          size="sm"
          variant={userVote === "disagree" ? "default" : "outline"}
          className={userVote === "disagree" ? "bg-red-500/20 border-red-500/30" : ""}
          onClick={() => handleVote("disagree")}
        >
          <ThumbsDown className="w-4 h-4 mr-1" />
          {votes.disagree}
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="ml-auto"
          onClick={onViewDetails}
        >
          <Eye className="w-4 h-4 mr-1" />
          Details
        </Button>
      </div>
    </Card>
  );
};

export default ExoplanetCard;
