import { useState, useEffect } from "react";
import { Sprout, Mail, Lock, ArrowRight, Eye, EyeOff } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";

export default function Login() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // If already logged in, redirect to dashboard or intended route
  useEffect(() => {
    if (user) {
      const from = location.state?.from?.pathname || "/";
      navigate(from, { replace: true });
    }
  }, [user, navigate, location]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        toast.success("Account created successfully. You are now logged in!");
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        toast.success("Welcome back!");
      }
    } catch (error: any) {
      toast.error(error.message || "Authentication failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: window.location.origin
        }
      });
      if (error) throw error;
    } catch (error: any) {
      toast.error(error.message || "Google Authentication failed.");
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center p-4 relative overflow-hidden isolate bg-background">
      {/* Abstract Background Elements */}
      <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-primary/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-success/10 blur-[120px] pointer-events-none" />

      <div className="w-full max-w-md animate-float-in z-10">
        <div className="glass-panel p-8 md:p-10 rounded-3xl flex flex-col items-center border border-white/10 dark:border-white/5 relative overflow-hidden">

          {/* Decorative Top Accent */}
          <div className="absolute top-0 left-0 right-0 h-1 gradient-primary opacity-80" />

          {/* Logo */}
          <div className="mb-8 flex flex-col items-center">
            <div className="h-14 w-14 rounded-2xl bg-transparent border-2 border-primary flex items-center justify-center shadow-glow mb-4">
              <Sprout className="h-7 w-7 text-primary" />
            </div>
            <h1 className="text-3xl font-bold font-display tracking-widest text-foreground uppercase">
              Soil<span className="text-primary">Net</span>
            </h1>
            <p className="text-[10px] font-mono uppercase tracking-[0.3em] text-muted-foreground mt-1">
              Login
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="w-full space-y-4">
            <div className="space-y-1">
              <label className="text-[11px] uppercase tracking-widest text-muted-foreground font-semibold px-1">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                </div>
                <Input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="manager@farm.com"
                  className="pl-10 h-12 bg-black/5 dark:bg-black/20 border-white/10 dark:border-white/5 focus:border-primary focus:ring-1 focus:ring-primary font-mono text-sm"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-[11px] uppercase tracking-widest text-muted-foreground font-semibold px-1 flex justify-between">
                <span>Password</span>
                {!isSignUp && <a href="#" className="text-primary hover:underline lowercase tracking-normal font-sans">Forgot Password?</a>}
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <Lock className="h-4 w-4 text-muted-foreground" />
                </div>
                <Input
                  type={showPassword ? "text" : "password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="pl-10 pr-10 h-12 bg-black/5 dark:bg-black/20 border-white/10 dark:border-white/5 focus:border-primary focus:ring-1 focus:ring-primary font-mono tracking-widest text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3 text-muted-foreground hover:text-foreground transition-colors focus:outline-none"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full h-12 mt-4 gradient-primary text-primary-foreground hover:opacity-95 shadow-glow font-bold tracking-widest uppercase text-xs"
            >
              {isLoading ? "Signing in..." : (isSignUp ? "Create Account" : "Sign In")}
              {!isLoading && <ArrowRight className="h-4 w-4 ml-2" />}
            </Button>
          </form>

          {/* Divider */}
          <div className="w-full flex items-center gap-3 my-6">
            <div className="h-px flex-1 bg-border" />
            <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">OR</span>
            <div className="h-px flex-1 bg-border" />
          </div>

          {/* OAuth */}
          <Button
            type="button"
            variant="outline"
            onClick={handleGoogleLogin}
            className="w-full h-12 bg-transparent border-border hover:bg-black/5 dark:hover:bg-white/5 font-medium tracking-wide"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5 mr-3" fill="currentColor">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 15.01 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
            </svg>
            Continue with Google
          </Button>

          {/* Toggle State */}
          <p className="mt-8 text-xs text-muted-foreground">
            {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
            <button
              type="button"
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-primary font-bold hover:underline"
            >
              {isSignUp ? "Sign In" : "Register"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
