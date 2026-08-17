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

test("roster editor displays a returned fatigue warning immediately", async ({ page }) => {
  if (authenticatedCookies) await page.context().addCookies(authenticatedCookies);
  else await signIn(page);
  await page.goto(`/roster/${rosterMonth}`);
  const cell = page.locator(".cell.editable").filter({ has: page.locator("[data-roster-shift-select]") }).first();
  const day = await cell.getAttribute("data-roster-day");
  const staffId = await cell.getAttribute("data-roster-staff");
  await page.route("**/assign/**", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      ok: true,
      staff_id: Number(staffId),
      day,
      code: "A",
      version: 999,
      is_training: false,
      fatigue_updates: [{ day, reasons: ["Minimum rest period breached"] }],
    }),
  }));
  await cell.locator("[data-roster-shift-select]").selectOption("A");
  await expect(cell.locator(".fatigue-hazard")).toHaveAttribute(
    "aria-label", /Minimum rest period breached/
  );
  await expect(cell.locator("[data-roster-shift-select]")).toHaveAttribute("aria-invalid", "true");
});

test("roster editor keeps direct accessible in-cell selection", async ({ page }) => {
  await page.context().addCookies(authenticatedCookies); await page.goto(`/roster/${rosterMonth}`);
  await expect(page.locator("[data-roster-readiness]")).toHaveCount(0);
  await expect(page.locator("[data-roster-command-palette]")).toHaveCount(0);
  await expect(page.locator("[data-roster-inspector]")).toHaveCount(1);
  await expect(page.locator(".cell.editable [data-roster-shift-select]").first()).toBeVisible();
});

test("roster keeps its controller row header opaque above scrolled cells", async ({ page }) => {
  if (authenticatedCookies) await page.context().addCookies(authenticatedCookies);
  else await signIn(page);
  const telemetry = page.waitForRequest("**/roster/telemetry");
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(`/roster/${rosterMonth}`);
  await expect(page.locator("table.roster caption")).toHaveText(/ATC roster for/);
  const nameCell = page.locator("table.roster tbody th.col-name").first();
  await expect(nameCell).toHaveAttribute("scope", "row");
  await expect(nameCell).toHaveCSS("position", "sticky");
  await page.evaluate(() => {
    window.scrollTo({ left: document.documentElement.scrollWidth, top: window.scrollY });
  });
  await expect.poll(() => page.evaluate(() => window.scrollX)).toBeGreaterThan(0);
  const layering = await nameCell.evaluate((cell) => {
    const table = cell.closest("table");
    const header = table?.querySelector("thead th.col-name");
    const rect = cell.getBoundingClientRect();
    const headerRect = header?.getBoundingClientRect();
    const target = document.elementFromPoint(
      rect.left + Math.min(12, Math.max(1, rect.width / 2)),
      rect.top + Math.min(12, Math.max(1, rect.height / 2)),
    );
    const style = getComputedStyle(cell);
    const headerStyle = header ? getComputedStyle(header) : null;
    const ordinaryCell = cell.parentElement?.querySelector(".cell");
    const ordinaryZIndex = ordinaryCell ? Number(getComputedStyle(ordinaryCell).zIndex) || 0 : 0;
    return {
      opaque: style.backgroundColor !== "transparent" && style.backgroundColor !== "rgba(0, 0, 0, 0)" && Number(style.opacity) === 1,
      remainsOnTop: Boolean(target && cell.contains(target)),
      bodyZIndex: Number(style.zIndex) || 0,
      headerZIndex: Number(headerStyle?.zIndex) || 0,
      ordinaryZIndex,
      alignsWithHeader: Boolean(headerRect && Math.abs(rect.left - headerRect.left) <= 1),
    };
  });
  expect(layering.opaque).toBe(true);
  expect(layering.remainsOnTop).toBe(true);
  expect(layering.bodyZIndex).toBeGreaterThan(layering.ordinaryZIndex);
  expect(layering.headerZIndex).toBeGreaterThan(layering.bodyZIndex);
  expect(layering.alignsWithHeader).toBe(true);
  await nameCell.hover();
  const hoverBackground = await nameCell.evaluate((cell) => {
    const style = getComputedStyle(cell);
    return style.backgroundColor !== "transparent" && style.backgroundColor !== "rgba(0, 0, 0, 0)" && Number(style.opacity) === 1;
  });
  expect(hoverBackground).toBe(true);
  await expect(page.locator("[data-roster-inspector]")).toBeHidden();
  await page.locator(".cell.editable [data-roster-shift-select]").first().focus();
  await expect(page.locator("[data-roster-inspector]")).toBeVisible();
  expect((await telemetry).method()).toBe("POST");
});
