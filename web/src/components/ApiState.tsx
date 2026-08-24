import { AlertCircle, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";

/**
 * Honest loading and failure states.
 *
 * The previous version of this app rendered hardcoded numbers labelled "live
 * predictions". If the API is unavailable now, the UI says so instead.
 */

export const Loading = ({ label = "Loading" }: { label?: string }) => (
  <div className="flex items-center gap-3 py-16 text-muted-foreground">
    <Loader2 className="h-5 w-5 animate-spin" />
    <span>{label}…</span>
  </div>
);

export const Failed = ({ error }: { error: Error }) => (
  <Alert variant="destructive" className="my-8">
    <AlertCircle className="h-4 w-4" />
    <AlertTitle>Could not load data</AlertTitle>
    <AlertDescription className="space-y-2">
      <p>{error.message}</p>
      <p className="text-xs opacity-80">
        Start the API with <code className="font-mono">make serve</code>, then reload.
      </p>
    </AlertDescription>
  </Alert>
);
