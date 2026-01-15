import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { generateLightCurve } from "@/data/mockData";

interface LightCurveChartProps {
  transitDepth: number;
  period: number;
  planetName: string;
}

const LightCurveChart = ({ transitDepth, period, planetName }: LightCurveChartProps) => {
  const data = generateLightCurve(transitDepth, period);

  return (
    <Card className="glass-card p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-cosmic-purple">Light Curve Analysis</h3>
        <p className="text-sm text-muted-foreground">Transit signature for {planetName}</p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
          <XAxis
            dataKey="time"
            label={{ value: "Time (days)", position: "insideBottom", offset: -5 }}
            stroke="hsl(var(--muted-foreground))"
            tick={{ fill: "hsl(var(--muted-foreground))" }}
          />
          <YAxis
            label={{ value: "Normalized Flux", angle: -90, position: "insideLeft" }}
            domain={[0.998, 1.002]}
            stroke="hsl(var(--muted-foreground))"
            tick={{ fill: "hsl(var(--muted-foreground))" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "var(--radius)",
            }}
            labelStyle={{ color: "hsl(var(--foreground))" }}
          />
          <Line
            type="monotone"
            dataKey="flux"
            stroke="hsl(var(--cosmic-purple))"
            strokeWidth={2}
            dot={false}
            isAnimationActive={true}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-4 p-3 rounded-lg bg-muted/50 border border-border">
        <p className="text-sm text-muted-foreground">
          <span className="text-cosmic-cyan font-medium">Transit detected:</span> The characteristic dip in brightness indicates a planetary transit event with depth of{" "}
          <span className="text-foreground font-semibold">{(transitDepth * 100).toFixed(3)}%</span>
        </p>
      </div>
    </Card>
  );
};

export default LightCurveChart;
