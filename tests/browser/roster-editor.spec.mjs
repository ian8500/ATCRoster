import { expect, test } from "@playwright/test";

const username = process.env.ATCROSTER_E2E_USERNAME || "admin";
const password = process.env.ATCROSTER_E2E_PASSWORD || "admin-password";

async function signIn(page) {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill(username);
  await page.getByLabel(/password/i).fill(password);
  await page.getByRole("button", { name: /sign in|login/i }).click();
}

test("roster editor supports async save, validation feedback and concurrency recovery", async ({ page }) => {
  await signIn(page); await page.goto("/roster/2025-04");
  const cell = page.locator(".cell.editable").filter({ has: page.locator("[data-roster-cell-action]") }).first();
  await cell.locator("select").selectOption("M");
  await expect(page.getByText("Saved in this session")).toBeVisible();
  await page.route("**/assign/**", route => route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ ok: false, error: "Unknown shift code" }) }));
  await cell.locator("select").selectOption("D");
  await expect(page.locator("[data-roster-save-status]")).toContainText(/unknown shift/i);
  await page.unrouteAll({ behavior: "ignoreErrors" });
  await page.route("**/assign/**", route => route.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify({ ok: false, error: "This roster cell changed after the page was loaded." }) }));
  await cell.locator("select").selectOption("A");
  await expect(page.locator("[data-roster-save-status]")).toContainText(/changed after the page was loaded/i);
});

test("roster editor supports undo, command palette, readiness filtering and keyboard navigation", async ({ page }) => {
  await signIn(page); await page.goto("/roster/2025-04");
  const cells = page.locator(".cell.editable:has([data-roster-cell-action] select:not(:disabled))");
  await cells.first().click(); await page.keyboard.press("ControlOrMeta+K");
  await expect(page.locator("[data-roster-command-palette]")).toBeVisible();
  await page.locator("[data-roster-command-input]").fill("M"); await page.keyboard.press("Enter");
  await page.getByRole("button", { name: /undo last change/i }).click();
  await page.getByRole("button", { name: /coverage/i }).click();
  await expect(page.locator("[data-roster-readiness-dialog]")).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".cell.is-selected [data-roster-cell-action] select:not(:disabled)")).toBeVisible();
});
