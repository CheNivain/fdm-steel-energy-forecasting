import { copyFileSync, mkdirSync } from "node:fs";
mkdirSync("vendor", { recursive: true });
for (const [source, target] of [
  ["node_modules/chart.js/dist/chart.umd.js", "chart.js"],
  ["node_modules/chart.js/LICENSE.md", "CHART-LICENSE.md"],
  ["node_modules/lucide/dist/umd/lucide.min.js", "lucide.js"],
  ["node_modules/lucide/LICENSE", "LUCIDE-LICENSE"],
])
  copyFileSync(source, `vendor/${target}`);
