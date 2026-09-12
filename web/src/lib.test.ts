import assert from "node:assert/strict";
import test from "node:test";

import { formatDuration, formatPercent, isEvaluationPending, progressPercent } from "./lib.ts";

test("formats evaluation values for people", () => {
  assert.equal(formatPercent(0.92345), "92.3%");
  assert.equal(formatDuration(61.2), "1m 01s");
});

test("bounds evaluation progress", () => {
  assert.equal(progressPercent(25, 100), 25);
  assert.equal(progressPercent(2, 0), 0);
  assert.equal(progressPercent(200, 100), 100);
});

test("keeps the console pending while an evaluation is starting or running", () => {
  assert.equal(isEvaluationPending("starting"), true);
  assert.equal(isEvaluationPending("running"), true);
  assert.equal(isEvaluationPending("complete"), false);
});
