import { expect, test } from "@playwright/test";
import crypto from "node:crypto";

const username = process.env.ATCROSTER_E2E_USERNAME || "lba.admin";
const password = process.env.ATCROSTER_E2E_PASSWORD || "Test-ATCRoster-2026!";
const mfaSecret = process.env.ATCROSTER_E2E_MFA_SECRET || "JBSWY3DPEHPK3PXP";
const rosterMonth = process.env.ATCROSTER_E2E_ROSTER_MONTH
  || new Date().toISOString().slice(0, 7);

function totp(secret) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  let bits = "";
  for (const character of secret.replace(/=+$/, "")) bits += alphabet.indexOf(character).toString(2).padStart(5, "0");
  const key = Buffer.from(bits.match(/.{1,8}/g).map(value => parseInt(value.padEnd(8, "0"), 2)));
  const counter = Buffer.alloc(8); counter.writeBigUInt64BE(BigInt(Math.floor(Date.now() / 30_000)));
  const hash = crypto.createHmac("sha1", key).update(counter).digest();
  const offset = hash[19] & 15;
  return String(((hash.readUInt32BE(offset) & 0x7fffffff) % 1_000_000)).padStart(6, "0");
}

async function signIn(page) {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill(username);
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/login/mfa"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);
  await page.getByLabel(/code/i).fill(totp(mfaSecret));
  await Promise.all([
    page.waitForURL(url => !url.pathname.endsWith("/login/mfa")),
    page.getByRole("button", { name: /verify|continue/i }).click(),
  ]);
}

let authenticatedCookies;

test.describe.configure({ mode: "serial" });

test("roster editor supports async save, validation feedback and concurrency recovery", async ({ page }) => {
  await signIn(page); await page.goto(`/roster/${rosterMonth}`);
  authenticatedCookies = await page.context().cookies();
  const cell = page.locator(".cell.editable").filter({ has: page.locator("[data-roster-shift-select]") }).first();
  await cell.locator("[data-roster-shift-select]").selectOption("M");
  await expect(page.locator("[data-roster-save-status]")).toContainText(/saved/i);
  await cell.locator("[data-roster-shift-select]").selectOption("A");
  await expect(page.locator("[data-roster-save-status]")).toContainText(/saved/i);
  await page.reload();
  await expect(cell.locator("[data-roster-shift-select]")).toBeVisible();
});

test("roster editor reports a stale async save", async ({ page }) => {
  if (authenticatedCookies) await page.context().addCookies(authenticatedCookies);
  else await signIn(page);
  await page.route("**/assign/**", (route) => route.fulfill({
    status: 409,
    contentType: "application/json",
    body: JSON.stringify({
      ok: false,
      error: "This roster cell changed after the page was loaded.",
      reload_required: true,
    }),
  }));
  await page.goto(`/roster/${rosterMonth}`);
  const cell = page.locator(".cell.editable").filter({ has: page.locator("[data-roster-shift-select]") }).first();
  await cell.locator("[data-roster-shift-select]").selectOption("M");
  await expect(page.locator("[data-roster-save-status]")).toContainText(/changed after the page was loaded/i);
});

test("roster editor keeps direct accessible in-cell selection", async ({ page }) => {
  await page.context().addCookies(authenticatedCookies); await page.goto(`/roster/${rosterMonth}`);
  await expect(page.locator("[data-roster-readiness]")).toHaveCount(0);
  await expect(page.locator("[data-roster-command-palette]")).toHaveCount(0);
  await expect(page.locator("[data-roster-inspector]")).toHaveCount(1);
  await expect(page.locator(".cell.editable [data-roster-shift-select]").first()).toBeVisible();
});
