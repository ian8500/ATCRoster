import { expect, test } from "@playwright/test";
import crypto from "node:crypto";

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

test("staff roster view is read-only without misleading editor controls", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill("lba.atco01");
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/login/mfa"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);
  await page.getByLabel(/code/i).fill(totp(mfaSecret));
  await page.getByRole("button", { name: /verify|continue/i }).click();
  await page.goto(`/roster/${rosterMonth}`);

  await expect(page.locator("[data-roster-readiness]")).toBeVisible();
  await expect(page.locator("[data-roster-command-open]")).toHaveCount(0);
  await expect(page.locator("[data-roster-command-palette]")).toHaveCount(0);
  await expect(page.locator("[data-roster-inspector]")).toHaveCount(0);
  await expect(page.locator("[data-roster-undo]")).toHaveCount(0);
  await expect(page.locator(".cell.editable")).toHaveCount(0);
});
