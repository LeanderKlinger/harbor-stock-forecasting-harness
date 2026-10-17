import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router";
import { Leaderboard } from "@/pages/Leaderboard";
import { RunDetail } from "@/pages/RunDetail";
import { RunComparison } from "@/pages/RunComparison";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/runs/:modelType/:runDir" element={<RunDetail />} />
          <Route path="/compare" element={<RunComparison />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
