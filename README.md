# boss-react

`boss-react` is a self-contained LangChain ReAct agent and BOSS browser tool
layer. Its production backend uses nodriver from this project's own uv
environment and maintains its own login profile without Playwright's
`Runtime.enable` fingerprint. The Playwright middleware remains available for
local test pages.

## Run the agent

The executable agent is built with LangChain `create_agent`. It loads the
OpenAI-compatible `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` from this
project's `.env`. Use `.env.example` as the template.

Edit `config/agent.toml` and optionally set `[agent].task`, then run:

```powershell
uv run boss-react
```

When `task` is empty, the terminal keeps line breaks from a bracketed multi-line
paste and submits the complete text when Enter is pressed afterward. Consoles
without enhanced paste support fall back to empty-line submission. You can also
override the task for one run:

```powershell
uv run boss-react --task "从首页推荐中选择并沟通一个匹配岗位"
```

Before asking for a task, the CLI loads the configured checkpoint and prints
its readable conversation history. Resume contents are represented by a short
placeholder. Tool calls, screenshots, and tool results are omitted entirely.
When the checkpoint has no messages, no history section is printed.
During a run, press Esc to request a LangGraph pause at the next step boundary
and return to the CLI prompt. The current model or browser tool call finishes
first. Esc also exits `/run forever`; completed steps remain in the checkpoint.

LangGraph state is persisted by `AsyncSqliteSaver`. The database path and stable
`thread_id` are configured in `[checkpoint]`. On the first run, the state starts
with the redacted resume text from `config/resume.txt` and the task. Later runs
restore that thread and append only the new task, including after a process
restart. Change `thread_id` when you intentionally want a fresh conversation;
the old thread remains in the same database. The general browser instructions
live in `config/system_prompt.txt`.

The agent includes a custom `AgentNodeCompactionMiddleware`. Its threshold is in
the `[context]` section of `config/agent.toml`. At the start of an invocation,
an oversized history receives one trailing compaction `HumanMessage` and runs
through the normal agent model node so the unchanged prefix can use provider
prompt caching. Tools are removed from that model request and any returned tool
call is rejected. The checkpoint is then replaced with only the resume, current
task, and compacted history before normal agent execution continues.

## Nodriver tools

- `browser_state`
- `browser_observe`
- `browser_screenshot`
- `browser_click_text`
- `browser_hover_text`
- `browser_input_text`
- `browser_scroll`
- `browser_scroll_to_text`
- `browser_press`
- `browser_back`
- `browser_reset`
- `browser_wait`
- `browser_eval_js`
- `browser_upload_text`
- `boss_send_greeting`
- `boss_send_image` (only when `[agent].resume_image_path` is configured; sends that fixed resume image)

Normal actions locate elements from visible text and accessibility metadata,
with optional `role`, surrounding `scope_text`, matching mode, and occurrence.
They do not accept model-provided screen coordinates. `browser_eval_js` accepts
arbitrary page JavaScript and returns the value from an explicit `return`.

No tool or button is hidden based on page state. `browser_state` still reports
the detected page kind for context, while invalid actions fail with a tool error
the agent can inspect and recover from.

Tabs are intentionally invisible to the agent. A click that opens a new page
automatically makes it active and closes older pages. `browser_back` uses the
active page's history and returns a tool error when there is no previous page.
`browser_reset` opens a new homepage and closes previous tabs while preserving
the profile and login cookies.

## LangChain setup

```python
from langchain.agents import create_agent

from boss_react import NodriverBrowserConfig, NodriverBrowserMiddleware

browser = NodriverBrowserMiddleware(
    NodriverBrowserConfig(
        profile_dir=r"G:\WorkSpace\boss_react\browser-profile\nodriver",
        start_url="https://www.zhipin.com/",
    )
)

agent = create_agent(model, middleware=[browser])
result = await agent.ainvoke({"messages": [{"role": "user", "content": "查看当前页面"}]})

# Keep the middleware alive across agent invocations, then close it explicitly.
await browser.aclose()
```

The middleware launches `scripts/nodriver_tool_driver.py` with this project's
`.venv` Python. The JSON-lines subprocess owns the browser object; LangChain
only receives structured tool results and screenshot image content. Override
`python_executable`, `profile_dir`, or `driver_script` in
`NodriverBrowserConfig` when needed.

Before every agent invocation, the browser middleware checks the current login
state. If the page shows a login wall, login header control, login-required
content, or a known login URL, it opens the BOSS login page before the model can
act. The user completes any required scan in that browser profile.

## Playwright test backend

`PlaywrightBrowserMiddleware` is still exported for ordinary pages and unit
tests. It is not the recommended BOSS backend because attaching Playwright over
CDP caused the live site to enter its anti-debug refresh loop.
