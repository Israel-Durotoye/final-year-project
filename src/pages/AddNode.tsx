import { useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MapPreview } from "@/components/MapPreview";
import { nodes } from "@/lib/mockData";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MapPin, Save } from "lucide-react";
import { toast } from "sonner";

const AddNode = () => {
  const [pin, setPin] = useState({ lat: 36.778, lng: -119.418 });
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success("Node registered successfully", { description: "It will appear after first telemetry transmission." });
  };

  return (
    <>
      <PageHeader title="Add Node" subtitle="Register a new sensor node to your network" />
      <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <form onSubmit={handleSubmit} className="bg-card border border-border rounded-xl shadow-card p-6 space-y-5">
          <div>
            <h2 className="font-semibold text-lg">Node Details</h2>
            <p className="text-sm text-muted-foreground">Provide identification and farm assignment</p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="nodeId">Node ID</Label>
            <Input id="nodeId" placeholder="NODE_07" required />
          </div>
          <div className="space-y-2">
            <Label htmlFor="farm">Farm Name</Label>
            <Select>
              <SelectTrigger><SelectValue placeholder="Select or add farm" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="green-valley">Green Valley Farm</SelectItem>
                <SelectItem value="sunrise">Sunrise Acres</SelectItem>
                <SelectItem value="orchard">Orchard Hills</SelectItem>
                <SelectItem value="new">+ Create new farm</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="loc">Location / Field</Label>
            <Input id="loc" placeholder="e.g. North Field A" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label>Latitude</Label>
              <Input value={pin.lat.toFixed(4)} onChange={(e) => setPin({ ...pin, lat: +e.target.value || 0 })} />
            </div>
            <div className="space-y-2">
              <Label>Longitude</Label>
              <Input value={pin.lng.toFixed(4)} onChange={(e) => setPin({ ...pin, lng: +e.target.value || 0 })} />
            </div>
          </div>
          <Button type="submit" className="w-full gap-2"><Save className="h-4 w-4" />Register Node</Button>
        </form>

        <div className="bg-card border border-border rounded-xl shadow-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="font-semibold text-lg">Drop a Pin</h2>
              <p className="text-sm text-muted-foreground">Click anywhere on the map to set coordinates</p>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground bg-secondary px-2.5 py-1 rounded-full">
              <MapPin className="h-3 w-3" />
              {pin.lat.toFixed(4)}, {pin.lng.toFixed(4)}
            </div>
          </div>
          <div onClick={(e) => {
            const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
            const x = (e.clientX - r.left) / r.width;
            const y = (e.clientY - r.top) / r.height;
            setPin({ lat: +(36.775 + (1 - y) * 0.01).toFixed(4), lng: +(-119.425 + x * 0.015).toFixed(4) });
          }}>
            <MapPreview nodes={[...nodes, { id: "NEW", farm: "", location: "", lat: pin.lat, lng: pin.lng, status: "online", metrics: nodes[0].metrics, lastReading: "" }]} height="h-[420px]" selectedId="NEW" />
          </div>
        </div>
      </div>
    </>
  );
};

export default AddNode;
