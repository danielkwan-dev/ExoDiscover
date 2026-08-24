import { Link } from "react-router-dom";

import { Button } from "../components/ui/button";

const NotFound = () => (
  <div className="container flex min-h-[60vh] items-center justify-center px-4">
    <div className="text-center">
      <h1 className="text-6xl font-bold text-cosmic-purple">404</h1>
      <p className="mt-4 text-xl text-muted-foreground">
        Nothing in this part of the sky.
      </p>
      <Button asChild className="mt-8">
        <Link to="/">Back to the overview</Link>
      </Button>
    </div>
  </div>
);

export default NotFound;
