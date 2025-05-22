Keep pre-created configs and pre-setup profiles for playwright here.
Pre-setup browser profiles may be copied to the docker by adding appropriate commands to `dockerfile_lines`
As this folder will copy to the docker, mcp-server-config for playwright may use `/deps/langgraph-mcp/playwright/config.json` as the config path as follows:

```
{
    "playwright": {
        "command": "npx",
        "args": [
          "@playwright/mcp@latest",
          "--no-sandbox",
          "--config",
          "/deps/langgraph-mcp/playwright/config.json"
        ],
        "description": "Use for browser based interaction or automation tasks"
    }
}
```