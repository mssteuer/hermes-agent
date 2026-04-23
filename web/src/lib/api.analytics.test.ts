/// <reference types="node" />

import test from "node:test";
import assert from "node:assert/strict";

import { normalizeAnalyticsResponse, type AnalyticsResponse } from "./api.ts";

test("normalizeAnalyticsResponse fills in missing skill analytics for older backends", () => {
  const raw = {
    daily: [],
    by_model: [],
    totals: {
      total_input: 0,
      total_output: 0,
      total_cache_read: 0,
      total_reasoning: 0,
      total_estimated_cost: 0,
      total_actual_cost: 0,
      total_sessions: 0,
      total_api_calls: 0,
    },
  } as AnalyticsResponse;

  const normalized = normalizeAnalyticsResponse(raw);

  assert.deepEqual(normalized.skills, {
    summary: {
      total_skill_loads: 0,
      total_skill_edits: 0,
      total_skill_actions: 0,
      distinct_skills_used: 0,
    },
    top_skills: [],
  });
});

test("normalizeAnalyticsResponse preserves populated skill analytics", () => {
  const raw: AnalyticsResponse = {
    daily: [],
    by_model: [],
    totals: {
      total_input: 0,
      total_output: 0,
      total_cache_read: 0,
      total_reasoning: 0,
      total_estimated_cost: 0,
      total_actual_cost: 0,
      total_sessions: 0,
      total_api_calls: 0,
    },
    skills: {
      summary: {
        total_skill_loads: 2,
        total_skill_edits: 1,
        total_skill_actions: 3,
        distinct_skills_used: 2,
      },
      top_skills: [
        {
          skill: "systematic-debugging",
          view_count: 2,
          manage_count: 1,
          total_count: 3,
          percentage: 100,
          last_used_at: 1713900000,
        },
      ],
    },
  };

  const normalized = normalizeAnalyticsResponse(raw);

  assert.deepEqual(normalized.skills, raw.skills);
});
