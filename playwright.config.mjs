import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests/browser",
  use: {
    baseURL: process.env.ATCROSTER_E2E_BASE_URL || "http://127.0.0.1:5000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
});
