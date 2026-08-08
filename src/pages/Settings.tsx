import { useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { BellRing, RotateCcw, Save } from "lucide-react";
import { toast } from "sonner";
import {
  AlertThresholds,
  DEFAULT_THRESHOLDS,
  METRICS,
  MetricKey,
  loadAlertThresholds,
  saveAlertThresholds,
} from "@/lib/alerting";
import { UserProfile, loadUserProfile, saveUserProfile } from "@/lib/profile";

const Settings = () => {
  const [thresholds, setThresholds] = useState<AlertThresholds>(loadAlertThresholds);
  const [notifications, setNotifications] = useState({ laptop: true, email: false, sms: false });
  const [profile, setProfile] = useState<UserProfile>(loadUserProfile);

  const saveProfile = () => {
    if (!profile.name.trim() || !profile.role.trim()) {
      toast.error("Add both a name and role before saving your profile.");
      return;
    }
    saveUserProfile({ ...profile, name: profile.name.trim(), role: profile.role.trim(), email: profile.email.trim() });
    toast.success("Profile updated.");
  };

  const updateThreshold = (metric: MetricKey, bound: "min" | "max", value: string) => {
    const numericValue = value === "" ? Number.NaN : Number(value);
    setThresholds((current) => ({
      ...current,
      [metric]: { ...current[metric], [bound]: numericValue },
    }));
  };

  const saveThresholds = () => {
    const invalidMetric = METRICS.find(({ key }) => {
      const { min, max } = thresholds[key];
      return !Number.isFinite(min) || !Number.isFinite(max) || min >= max;
    });

    if (invalidMetric) {
      toast.error(`${invalidMetric.label} needs a minimum lower than its maximum.`);
      return;
    }

    saveAlertThresholds(thresholds);
    toast.success("Alert thresholds saved. New node readings will be checked against them.");
  };

  const resetThresholds = () => {
    setThresholds(DEFAULT_THRESHOLDS);
    toast.message("Default values restored. Save to apply them.");
  };

  return (
    <>
      <PageHeader title="Settings" subtitle="Configure your account, alerts, and integrations" />
      <div className="max-w-4xl space-y-6 p-6">
        <section className="space-y-4 rounded-xl border border-border bg-card p-6 shadow-card">
          <div><h2 className="text-lg font-semibold">Profile</h2><p className="mt-1 text-sm text-muted-foreground">This name and role appear in the collapsible account control at the bottom-left of the app.</p></div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-2"><Label htmlFor="profile-name">Full name</Label><Input id="profile-name" value={profile.name} onChange={(event) => setProfile((current) => ({ ...current, name: event.target.value }))} /></div>
            <div className="space-y-2"><Label htmlFor="profile-role">Role</Label><Input id="profile-role" value={profile.role} onChange={(event) => setProfile((current) => ({ ...current, role: event.target.value }))} /></div>
            <div className="space-y-2 sm:col-span-2"><Label htmlFor="profile-email">Email</Label><Input id="profile-email" type="email" value={profile.email} onChange={(event) => setProfile((current) => ({ ...current, email: event.target.value }))} /></div>
          </div>
          <div className="flex justify-end"><Button onClick={saveProfile}><Save className="mr-2 h-4 w-4" />Save profile</Button></div>
        </section>

        <section className="space-y-5 rounded-xl border border-border bg-card p-6 shadow-card">
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-start">
            <div>
              <div className="flex items-center gap-2"><BellRing className="h-5 w-5 text-primary" /><h2 className="text-lg font-semibold">Alert thresholds</h2></div>
              <p className="mt-1 text-sm text-muted-foreground">An alert is created whenever the latest reading for a node falls outside these limits.</p>
            </div>
            <Button variant="ghost" size="sm" onClick={resetThresholds}><RotateCcw className="mr-2 h-4 w-4" />Restore defaults</Button>
          </div>

          <div className="overflow-hidden rounded-lg border border-border">
            <div className="grid grid-cols-[1fr_7rem_7rem] gap-3 bg-secondary/60 px-4 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground sm:grid-cols-[1fr_9rem_9rem]">
              <span>Sensor reading</span><span>Minimum</span><span>Maximum</span>
            </div>
            {METRICS.map((metric) => (
              <div key={metric.key} className="grid grid-cols-[1fr_7rem_7rem] items-center gap-3 border-t border-border px-4 py-3 sm:grid-cols-[1fr_9rem_9rem]">
                <div><p className="text-sm font-medium">{metric.label}</p><p className="text-xs text-muted-foreground">{metric.unit}</p></div>
                <Input aria-label={`Minimum ${metric.label}`} type="number" value={thresholds[metric.key].min} onChange={(event) => updateThreshold(metric.key, "min", event.target.value)} />
                <Input aria-label={`Maximum ${metric.label}`} type="number" value={thresholds[metric.key].max} onChange={(event) => updateThreshold(metric.key, "max", event.target.value)} />
              </div>
            ))}
          </div>

          <div className="flex justify-end"><Button onClick={saveThresholds}><Save className="mr-2 h-4 w-4" />Save thresholds</Button></div>
        </section>

        <section className="space-y-4 rounded-xl border border-border bg-card p-6 shadow-card">
          <h2 className="text-lg font-semibold">Notifications</h2>
          {[
            ["Laptop alerts", "Show threshold breaches in the Alerts page", "laptop"],
            ["Email alerts", "Reserved for your Supabase notification workflow", "email"],
            ["SMS alerts", "Reserved for your Supabase notification workflow", "sms"],
          ].map(([title, description, key]) => (
            <div key={key} className="flex items-center justify-between gap-4 py-2">
              <div><p className="text-sm font-medium">{title}</p><p className="text-xs text-muted-foreground">{description}</p></div>
              <Switch checked={notifications[key as keyof typeof notifications]} onCheckedChange={(checked) => setNotifications((current) => ({ ...current, [key]: checked }))} />
            </div>
          ))}
        </section>
      </div>
    </>
  );
};

export default Settings;
