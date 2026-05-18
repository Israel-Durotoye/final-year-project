import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Sprout } from "lucide-react";

export const ProtectedRoute = () => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background">
        <div className="h-16 w-16 rounded-2xl bg-transparent border-2 border-primary flex items-center justify-center shadow-glow mb-4 animate-pulse">
          <Sprout className="h-8 w-8 text-primary animate-bounce" />
        </div>
        <p className="text-xs font-mono tracking-[0.3em] text-muted-foreground uppercase animate-pulse">Authenticating...</p>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
};
