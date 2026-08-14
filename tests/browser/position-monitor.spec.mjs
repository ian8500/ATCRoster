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
  await page.getByLabel("Primary controller").selectOption({ label: "Alex Taylor" });
  await page.getByRole("button", { name: "Confirm" }).click();
  await expect(tower).toContainText("Alex Taylor");
  await tower.getByRole("button", { name: "Add secondary" }).click();
  await page.getByLabel("Secondary role").selectOption("ojti");
  const secondary = page.locator("#live-position-support-person");
  const secondaryOption = await secondary.locator("option").evaluateAll((options) => {
    const eligible = options.find((option) => option.value);
    return eligible ? { value: eligible.value, label: eligible.textContent } : null;
  });
  expect(secondaryOption).not.toBeNull();
  await secondary.selectOption(secondaryOption.value);
  await page.getByRole("button", { name: "Confirm" }).click();
  await expect(tower).toContainText("OJTI");

  await tower.locator("footer button[data-operation='logoff']").click();
  await page.getByRole("button", { name: /\(OJTI\) only$/ }).click();
  await expect(tower).not.toContainText("OJTI");
  await expect(tower).toContainText("Alex Taylor");

  await tower.getByRole("button", { name: "Hand over" }).click();
  const primary = page.getByLabel("Primary controller");
  const handoverOption = await primary.locator("option").evaluateAll((options) => {
    const eligible = options.find((option) => option.value && option.textContent !== "Alex Taylor");
    return eligible ? { value: eligible.value, label: eligible.textContent } : null;
  });
  expect(handoverOption).not.toBeNull();
  await primary.selectOption(handoverOption.value);
  await page.getByRole("button", { name: "Confirm" }).click();
  await expect(tower).toContainText(handoverOption.label);
  await expect(tower).not.toContainText("Alex Taylor");

  const logOff = tower.locator("footer button[data-operation='logoff']");
  await logOff.click();
  await expect(tower.getByRole("button", { name: "Log on" })).toBeVisible();

  await page.goto("/administration");
  await expect(page).toHaveURL(/\/live-positions\/kiosk$/);
  await page.goto(`/roster/${rosterMonth}`);
  await expect(page).toHaveURL(/\/live-positions\/kiosk$/);
});

test("Position Monitor makes an offline display explicitly stale", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill(username);
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/live-positions/kiosk"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);
  await expect(page.locator("#live-position-grid article").first()).toBeVisible();

  await page.evaluate(() => window.dispatchEvent(new Event("offline")));
  await expect(page.locator("#live-position-warning")).toContainText(/may be stale/i);
  await expect(page.locator("#live-position-board-viewport")).toHaveClass(/is-stale/);
  await expect(page.locator("#live-position-connection")).toContainText("Reconnecting");
});
