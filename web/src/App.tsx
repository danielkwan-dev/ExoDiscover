import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import Nav from "./components/Nav";
import { Toaster } from "./components/ui/sonner";
import { TooltipProvider } from "./components/ui/tooltip";
import Discoveries from "./pages/Discoveries";
import Index from "./pages/Index";
import ModelDashboard from "./pages/ModelDashboard";
import NotFound from "./pages/NotFound";
import Predict from "./pages/Predict";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <BrowserRouter>
        <Nav />
        <main>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/discoveries" element={<Discoveries />} />
            <Route path="/model" element={<ModelDashboard />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
