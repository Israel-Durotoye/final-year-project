import { PageHeader } from "@/components/layout/PageHeader";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

const Settings = () => (
  <>
    <PageHeader title="Settings" subtitle="Configure your account, alerts, and integrations" />
    <div className="p-6 max-w-3xl space-y-6">
      <section className="bg-card border border-border rounded-xl shadow-card p-6 space-y-4">
        <h2 className="font-semibold text-lg">Profile</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2"><Label>Full name</Label><Input defaultValue="Israel Durotoye" /></div>
          <div className="space-y-2"><Label>Email</Label><Input defaultValue="israeldurotoye@gmail.com" /></div>
        </div>
      </section>

      <section className="bg-card border border-border rounded-xl shadow-card p-6 space-y-4">
        <h2 className="font-semibold text-lg">Alert Thresholds</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="space-y-2"><Label>Min N (ppm)</Label><Input type="number" defaultValue={25} /></div>
          <div className="space-y-2"><Label>Min Moisture (%)</Label><Input type="number" defaultValue={30} /></div>
          <div className="space-y-2"><Label>Max Temp (°C)</Label><Input type="number" defaultValue={32} /></div>
        </div>
      </section>

      <section className="bg-card border border-border rounded-xl shadow-card p-6 space-y-4">
        <h2 className="font-semibold text-lg">Notifications</h2>
        {[
          ["Email alerts", "Receive critical alerts via email", true],
          ["SMS alerts", "Text messages for critical issues", false],
          ["Daily summary", "Morning report of all node activity", true],
        ].map(([t, d, v]) => (
          <div key={t as string} className="flex items-center justify-between py-2">
            <div>
              <p className="font-medium text-sm">{t}</p>
              <p className="text-xs text-muted-foreground">{d}</p>
            </div>
            <Switch defaultChecked={v as boolean} />
          </div>
        ))}
      </section>

      <Button>Save changes</Button>
    </div>
  </>
);

export default Settings;
