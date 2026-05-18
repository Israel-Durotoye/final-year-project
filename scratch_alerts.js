const { createClient } = require("@supabase/supabase-js");

const url = process.env.VITE_SUPABASE_URL;
const key = process.env.VITE_SUPABASE_ANON_KEY;

const supabase = createClient(url, key);

async function main() {
  const { data, error } = await supabase.from('alerts').select('*').limit(5);
  console.log("Alerts:", data);
  console.log("Error:", error);
}

main();
