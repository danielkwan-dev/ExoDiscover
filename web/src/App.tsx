import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import Space from "./pages/Space";

/**
 * One scene, no routes.
 *
 * The app used to be five pages of prose, forms and tables. It is now a place:
 * you are inside the catalogue, and the model's opinion of any object appears
 * when you click it. The analysis those pages carried lives in README.md,
 * docs/LEAKAGE.md and docs/MODEL_CARD.md, where it can be read properly.
 */

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

const App = () => (
  <QueryClientProvider client={queryClient}>
    <Space />
  </QueryClientProvider>
);

export default App;
