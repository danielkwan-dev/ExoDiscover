import { NavLink } from "react-router-dom";
import { Telescope } from "lucide-react";

const links = [
  { to: "/", label: "Overview", end: true },
  { to: "/predict", label: "Classify" },
  { to: "/discoveries", label: "Discoveries" },
  { to: "/model", label: "Model" },
];

const Nav = () => (
  <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-lg">
    <nav className="container flex h-16 items-center gap-8 px-4">
      <NavLink to="/" className="flex items-center gap-2 font-semibold">
        <Telescope className="h-5 w-5 text-accent" />
        ExoDiscover
      </NavLink>
      <ul className="flex items-center gap-6 text-sm">
        {links.map((link) => (
          <li key={link.to}>
            <NavLink
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                isActive
                  ? "text-foreground font-medium"
                  : "text-muted-foreground transition-colors hover:text-foreground"
              }
            >
              {link.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  </header>
);

export default Nav;
