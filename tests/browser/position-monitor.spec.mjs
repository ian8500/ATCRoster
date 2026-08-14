import { expect, test } from "@playwright/test";

const username = process.env.ATCROSTER_E2E_KIOSK_USERNAME || "lba.kiosk";
const password = process.env.ATCROSTER_E2E_PASSWORD || "Test-ATCRoster-2026!";
const rosterMonth = process.env.ATCROSTER_E2E_ROSTER_MONTH
  || new Date().toISOString().slice(0, 7);

test("Position Monitor kiosk is operational and remains least-privilege", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill(username);
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/live-positions/kiosk"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);

  await expect(page.locator(".live-position-kiosk")).toBeVisible();
  const startKiosk = page.getByRole("button", { name: "Start kiosk" });
  if (await startKiosk.count()) await startKiosk.click();

  const tower = page.locator("article").filter({ hasText: "Aerodrome Control" });
  const logOn = tower.getByRole("button", { name: "Log on" });
  await expect(logOn).toBeVisible();
  await logOn.click();
  await expect(page.getByLabel("Primary controller")).toBeVisible();
  await expect(page.getByLabel("Primary controller")).toContainText("Alex Taylor");
  await expect(page.getByLabel("Secondary role")).toContainText("OJTI");
  await page.getByRole("button", { name: "Cancel" }).click();

  await page.goto("/administration");
  await expect(page).toHaveURL(/\/live-positions\/kiosk$/);
  await page.goto(`/roster/${rosterMonth}`);
  await expect(page).toHaveURL(/\/live-positions\/kiosk$/);
});
