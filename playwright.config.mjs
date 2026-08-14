import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests/browser",
  // The isolated acceptance app deliberately uses one seeded SQLite database.
  // Browser scenarios mutate it, so parallel workers would create artificial
  // write contention rather than exercising a customer-facing workflow.
  workers: 1,
  use: {
    baseURL: process.env.ATCROSTER_E2E_BASE_URL || "http://127.0.0.1:5000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
});
