import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Failed, Loading } from "./ApiState";

/**
 * These guard a claim the project makes about itself.
 *
 * The hackathon version rendered hardcoded numbers under the label "live
 * predictions", so a dead API looked exactly like a working one. Every screen
 * now routes through these two components instead, which means the honesty of
 * the UI rests on them saying something true when the API is unreachable.
 */

afterEach(cleanup);

describe("Loading", () => {
  it("names what is being waited on", () => {
    render(<Loading label="Fetching metrics" />);
    expect(screen.getByText(/Fetching metrics/)).toBeInTheDocument();
  });
});

describe("Failed", () => {
  it("shows the underlying error rather than a generic message", () => {
    render(<Failed error={new Error("Cannot reach the API at http://localhost:8000")} />);
    expect(
      screen.getByText("Cannot reach the API at http://localhost:8000"),
    ).toBeInTheDocument();
  });

  it("tells the reader how to start the API", () => {
    render(<Failed error={new Error("boom")} />);
    expect(screen.getByText("make serve")).toBeInTheDocument();
  });

  it("never renders a number that could pass for a prediction", () => {
    const { container } = render(<Failed error={new Error("boom")} />);
    expect(container.textContent).not.toMatch(/0\.\d{2,}/);
  });
});
