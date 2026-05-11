import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "@/components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import Nodes from "./pages/Nodes";
import AddNode from "./pages/AddNode";
import MapView from "./pages/MapView";
import DataTable from "./pages/DataTable";
import Alerts from "./pages/Alerts";
import AiDoctor from "./pages/AiDoctor";
import SettingsPage from "./pages/Settings";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/nodes" element={<Nodes />} />
            <Route path="/add-node" element={<AddNode />} />
            <Route path="/map" element={<MapView />} />
            <Route path="/data" element={<DataTable />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/ai-doctor" element={<AiDoctor />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
