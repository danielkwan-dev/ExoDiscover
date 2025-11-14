import { Slider } from "@/components/ui/slider";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Filter } from "lucide-react";

interface FilterPanelProps {
  orbitalPeriod: number[];
  setOrbitalPeriod: (value: number[]) => void;
  transitDepth: number[];
  setTransitDepth: (value: number[]) => void;
  radius: number[];
  setRadius: (value: number[]) => void;
}

const FilterPanel = ({
  orbitalPeriod,
  setOrbitalPeriod,
  transitDepth,
  setTransitDepth,
  radius,
  setRadius,
}: FilterPanelProps) => {
  return (
    <Card className="glass-card p-6 space-y-6">
      <div className="flex items-center gap-2 mb-4">
        <Filter className="w-5 h-5 text-cosmic-purple" />
        <h3 className="text-lg font-semibold">Filter Exoplanets</h3>
      </div>

      <div className="space-y-6">
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <Label>Orbital Period (days)</Label>
            <span className="text-sm text-muted-foreground">
              {orbitalPeriod[0]} - {orbitalPeriod[1]}
            </span>
          </div>
          <Slider
            value={orbitalPeriod}
            onValueChange={setOrbitalPeriod}
            min={0}
            max={50}
            step={0.1}
            className="w-full"
          />
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <Label>Transit Depth</Label>
            <span className="text-sm text-muted-foreground">
              {transitDepth[0].toFixed(4)} - {transitDepth[1].toFixed(4)}
            </span>
          </div>
          <Slider
            value={transitDepth}
            onValueChange={setTransitDepth}
            min={0}
            max={0.003}
            step={0.0001}
            className="w-full"
          />
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <Label>Planet Radius (Earth radii)</Label>
            <span className="text-sm text-muted-foreground">
              {radius[0].toFixed(2)} - {radius[1].toFixed(2)}
            </span>
          </div>
          <Slider
            value={radius}
            onValueChange={setRadius}
            min={0}
            max={3}
            step={0.1}
            className="w-full"
          />
        </div>
      </div>

      <div className="pt-4 border-t border-border">
        <div className="text-sm text-muted-foreground">
          <p className="mb-2">
            <span className="text-cosmic-cyan font-medium">Orbital Period:</span> Time for one complete orbit
          </p>
          <p className="mb-2">
            <span className="text-cosmic-blue font-medium">Transit Depth:</span> Brightness dip during transit
          </p>
          <p>
            <span className="text-cosmic-purple font-medium">Planet Radius:</span> Size relative to Earth
          </p>
        </div>
      </div>
    </Card>
  );
};

export default FilterPanel;
