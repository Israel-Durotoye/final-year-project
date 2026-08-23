import { Outlet } from "react-router-dom";
import { TopNavbar } from "./TopNavbar";
import { LiveBackground } from "./LiveBackground";
import { FloatingControls } from "./FloatingControls";
import { AiWidget } from "./AiWidget";

export const AppLayout = () => (
  <div className="relative min-h-screen w-full bg-background flex flex-col font-sans">
    <LiveBackground />
    <TopNavbar />
    <main className="flex-1 w-full max-w-[1920px] mx-auto px-4 sm:px-6 md:px-8 py-6 flex flex-col relative z-10 pt-24">
      <Outlet />
    </main>
    <FloatingControls />
    <AiWidget />
  </div>
);
