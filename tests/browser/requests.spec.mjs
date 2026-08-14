import { expect, test } from "@playwright/test";
import crypto from "node:crypto";

const password = process.env.ATCROSTER_E2E_PASSWORD || "Test-ATCRoster-2026!";
const mfaSecret = process.env.ATCROSTER_E2E_MFA_SECRET || "JBSWY3DPEHPK3PXP";
const requestDate = (() => {
  const date = new Date();
  date.setUTCMonth(date.getUTCMonth() + 2, 15);
  return date.toISOString().slice(0, 10);
})();
const approvalMonth = (() => {
  const date = new Date();
  date.setUTCMonth(date.getUTCMonth() + 2, 1);
  return date.toISOString().slice(0, 7);
})();

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

test("ATCO can submit and review a future shift request", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill("lba.atco02");
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
  await page.goto("/requests");

  await page.getByLabel("Date").fill(requestDate);
  await page.getByLabel("Requested shift or status").selectOption("M");
  await page.getByLabel(/comment/i).fill("Browser acceptance request");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("status", { name: /request status: pending/i })).toBeVisible();
  await expect(page.getByText("Browser acceptance request")).toBeVisible();
});

test("manager can approve a pending request onto the roster", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill("ema.admin");
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
  await page.goto(`/requests?view=admin&ym=${approvalMonth}`);

  const pending = page.getByRole("row").filter({ hasText: /Pending/ }).first();
  await expect(pending).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await pending.getByRole("button", { name: /approve and add to roster/i }).click();
  await expect(page.getByRole("listitem").filter({
    hasText: "Request approved and added to the roster.",
  })).toBeVisible();
});
