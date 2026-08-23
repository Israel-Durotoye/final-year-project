/**
 * Nigerian Climate Season Detection
 *
 * Determines the agricultural season from a timestamp based on Nigeria's
 * bimodal rainfall distribution (Guinea Savanna / Middle Belt zone):
 *
 *   Month       Season
 *   ─────────   ──────────────────────
 *   Nov – Feb   Dry (Harmattan)
 *   Mar         Late Dry
 *   Apr – Jun   Early Rainy
 *   Jul – Sep   Peak Rainy
 *   Oct         Late Rainy
 */

export function getNigerianSeason(timestamp?: string | Date | null): string {
  let date: Date;

  if (!timestamp) {
    date = new Date();
  } else if (timestamp instanceof Date) {
    date = timestamp;
  } else {
    date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      date = new Date();
    }
  }

  const month = date.getMonth() + 1; // JS months are 0-indexed

  if ([11, 12, 1, 2].includes(month)) return "Dry (Harmattan)";
  if (month === 3) return "Late Dry";
  if ([4, 5, 6].includes(month)) return "Early Rainy";
  if ([7, 8, 9].includes(month)) return "Peak Rainy";
  if (month === 10) return "Late Rainy";

  return "Unknown";
}
