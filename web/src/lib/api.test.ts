import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./api";

afterEach(() => vi.unstubAllGlobals());

function mockFetch(response: Partial<Response> & { json?: () => Promise<unknown> }) {
  const fn = vi.fn().mockResolvedValue({ ok: true, status: 200, ...response });
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("api client", () => {
  it("returns the parsed body on success", async () => {
    mockFetch({ json: async () => ({ status: "ok", model_loaded: true }) });
    await expect(api.health()).resolves.toEqual({ status: "ok", model_loaded: true });
  });

  it("surfaces the server's detail message on a 4xx", async () => {
    mockFetch({
      ok: false,
      status: 422,
      statusText: "Unprocessable",
      json: async () => ({ detail: "CSV missing columns: koi_period" }),
    });
    await expect(api.metrics()).rejects.toThrow("CSV missing columns: koi_period");
  });

  it("reports an unreachable API rather than throwing a raw network error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed to fetch")));
    await expect(api.health()).rejects.toBeInstanceOf(ApiError);
    await expect(api.health()).rejects.toThrow(/Cannot reach the API/);
  });

  it("posts predictions as JSON", async () => {
    const fn = mockFetch({ json: async () => ({ label: "planet", probability: 0.9 }) });
    await api.predict({
      koi_period: 1,
      koi_depth: 100,
      koi_duration: 2,
      koi_prad: 1,
      koi_srad: 1,
      koi_slogg: 4.4,
      koi_steff: 5500,
      koi_impact: 0,
      n_kois_on_star: 1,
    });
    const [, init] = fn.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body).koi_period).toBe(1);
  });
});
