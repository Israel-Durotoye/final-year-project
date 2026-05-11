import { Outlet } from "react-router-dom";
import { AppSidebar } from "./AppSidebar";
import { AiWidget } from "./AiWidget";

export const AppLayout = () => (
  <div className="min-h-screen flex w-full bg-background">
    <AppSidebar />
    <main className="flex-1 min-w-0 flex flex-col">
      <Outlet />
    </main>
    <AiWidget />
  </div>
);
