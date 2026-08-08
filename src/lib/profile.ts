export type UserProfile = {
  name: string;
  role: string;
  email: string;
};

export const PROFILE_STORAGE_KEY = "soilnet-user-profile";

export const DEFAULT_PROFILE: UserProfile = {
  name: "Israel Durotoye",
  role: "Road Manager",
  email: "israeldurotoye@gmail.com",
};

export function loadUserProfile(): UserProfile {
  if (typeof window === "undefined") return DEFAULT_PROFILE;

  try {
    const stored = JSON.parse(window.localStorage.getItem(PROFILE_STORAGE_KEY) ?? "{}");
    return {
      name: typeof stored.name === "string" && stored.name.trim() ? stored.name.trim() : DEFAULT_PROFILE.name,
      role: typeof stored.role === "string" && stored.role.trim() ? stored.role.trim() : DEFAULT_PROFILE.role,
      email: typeof stored.email === "string" && stored.email.trim() ? stored.email.trim() : DEFAULT_PROFILE.email,
    };
  } catch {
    return DEFAULT_PROFILE;
  }
}

export function saveUserProfile(profile: UserProfile) {
  window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profile));
  window.dispatchEvent(new Event("soilnet:profile-updated"));
}

export function profileInitials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "ID";
}
