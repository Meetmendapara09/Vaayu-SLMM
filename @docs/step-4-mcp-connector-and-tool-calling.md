# Step 4: Native Model Context Protocol (MCP) Connector & Tool Calling

## 1. Overview

The **Model Context Protocol (MCP)** is the open standard that connects AI models to tools, resources, and execution contexts. 

Vaayu SLMM natively bridges to MCP servers using standard JSON-RPC 2.0 messages across `stdio` pipes or HTTP/SSE transports.

---

## 2. Dynamic Interface Comprehension

When Vaayu starts or connects to local software:
1. **Capability Discovery (`tools/list`)**:
   Vaayu queries the local MCP server:
   ```json
   {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
   ```
2. **Schema Ingestion**:
   The server returns JSON schemas for each tool (e.g. `read_file`, `execute_query`, `system_exec`).
   Vaayu injects these definitions into its system context.
3. **Structured Tool Emission**:
   When prompted to interact with the environment, Vaayu generates:
   ```
   <|tool_call_start|>{"name": "tool_name", "arguments": {...}}<|tool_call_end|>
   ```
4. **Execution & Feedback Loop**:
   The `MCPServerConnection` dispatches `tools/call`, receives the execution result, wraps it in `<|tool_result_start|>` ... `<|tool_result_end|>`, and passes it back to Vaayu to conclude the task.

---

## 3. Architecture of `vaayu.mcp_client`

- [`vaayu/mcp_client.py`](../vaayu/mcp_client.py):
  - `MCPServerConnection`: Background thread listener with thread-safe request/response queues.
  - Automatic `initialize` handshake negotiating protocolVersion `2024-11-05`.
  - Tool registration and dynamic parameter validation.