import { Telescope, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-exoplanet.jpg";

const Hero = () => {
  const scrollToExplorer = () => {
    document.getElementById("explorer")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 z-0">
        <img
          src={heroImage}
          alt="Exoplanet in deep space"
          className="w-full h-full object-cover opacity-30"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/50 via-background/80 to-background" />
      </div>

      {/* Animated Stars Background */}
      <div className="absolute inset-0 z-0">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cosmic-glow rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

      {/* Content */}
      <div className="container relative z-10 px-4 py-20">
        <div className="max-w-4xl mx-auto text-center space-y-8 animate-float">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card">
            <Sparkles className="w-4 h-4 text-cosmic-purple" />
            <span className="text-sm font-medium">Real-Time AI Classification</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
            <span className="bg-gradient-to-r from-cosmic-purple via-cosmic-blue to-cosmic-cyan bg-clip-text text-transparent">
              ExoDiscover
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Explore distant worlds using cutting-edge AI. Classify exoplanets from NASA's Kepler, K2, and TESS missions with dual machine learning models.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
            <Button
              size="lg"
              onClick={scrollToExplorer}
              className="group relative overflow-hidden bg-primary hover:bg-primary/90 cosmic-glow transition-all duration-300"
            >
              <Telescope className="w-5 h-5 mr-2" />
              Start Exploring
              <div className="absolute inset-0 bg-gradient-to-r from-cosmic-purple/20 to-cosmic-blue/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            </Button>

            <Button
              size="lg"
              variant="outline"
              className="glass-card border-primary/30 hover:border-primary/60 hover:bg-primary/10 transition-all duration-300"
            >
              View Methodology
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 pt-12 max-w-2xl mx-auto">
            {[
              { value: "5,000+", label: "Exoplanets Analyzed" },
              { value: "2", label: "AI Models" },
              { value: "98.3%", label: "Accuracy Rate" },
            ].map((stat, i) => (
              <div key={i} className="space-y-2">
                <div className="text-3xl md:text-4xl font-bold text-cosmic-purple">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent z-10" />
    </section>
  );
};

export default Hero;
