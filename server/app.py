# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI entrypoint for the support-ops environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from ..models import SupportOpsAction, SupportOpsObservation
    from .support_ops_env_environment import SupportOpsEnvironment
except ModuleNotFoundError:
    from models import SupportOpsAction, SupportOpsObservation
    from server.support_ops_env_environment import SupportOpsEnvironment


app = create_app(
    SupportOpsEnvironment,
    SupportOpsAction,
    SupportOpsObservation,
    env_name="support_ops_env",
    max_concurrent_envs=4,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> str:
    """Show a small human-friendly landing page for Spaces and browsers."""

    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Support Ops OpenEnv</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 0;
            background: #f5f7fb;
            color: #1f2937;
          }
          main {
            max-width: 760px;
            margin: 48px auto;
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
          }
          h1 {
            margin-top: 0;
            font-size: 2rem;
          }
          p {
            line-height: 1.6;
          }
          ul {
            line-height: 1.8;
          }
          code {
            background: #eef2ff;
            padding: 2px 6px;
            border-radius: 6px;
          }
          a {
            color: #1d4ed8;
            text-decoration: none;
          }
          a:hover {
            text-decoration: underline;
          }
        </style>
      </head>
      <body>
        <main>
          <h1>Support Ops OpenEnv</h1>
          <p>
            This Space hosts a real-world OpenEnv environment for SaaS support operations.
            Agents can triage tickets, add notes, reply to customers, merge duplicates, and
            complete graded workflows through the standard <code>reset</code>, <code>step</code>,
            and <code>state</code> API.
          </p>
          <p>Useful endpoints:</p>
          <ul>
            <li><a href="/docs">/docs</a> - interactive API docs</li>
            <li><a href="/metadata">/metadata</a> - environment metadata</li>
            <li><a href="/schema">/schema</a> - action, observation, and state schemas</li>
            <li><a href="/health">/health</a> - health check</li>
          </ul>
        </main>
      </body>
    </html>
    """


@app.get("/web", include_in_schema=False)
def web_redirect() -> RedirectResponse:
    """Keep browser-oriented links working by redirecting to the landing page."""

    return RedirectResponse(url="/", status_code=307)


def main():
    """Run the environment server on port 8000."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
