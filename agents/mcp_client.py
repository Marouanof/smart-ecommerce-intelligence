from agents.mcp_server import MCPServer, MCPError


class MCPClient:
    def __init__(self):
        self._server: MCPServer | None = None

    def connect(self, role: str = "viewer"):
        self._server = MCPServer(role=role)

    def disconnect(self):
        self._server = None

    @property
    def connected(self) -> bool:
        return self._server is not None

    @property
    def current_role(self) -> str:
        if self._server:
            return self._server.role
        return ""

    def list_tools(self) -> list[dict]:
        if not self._server:
            return []
        return self._server.list_tools()

    def call_tool(self, tool_name: str, params: dict | None = None, user: str = "dashboard") -> dict:
        if not self._server:
            return {"error": "MCP non connecte", "tool": tool_name}
        try:
            return self._server.call_tool(tool_name, params or {}, user=user)
        except MCPError as e:
            return {"error": e.message, "code": e.code, "tool": tool_name}

    def set_role(self, role: str):
        if self._server:
            self._server.set_role(role)

    def get_logs(self, n: int = 50) -> list[dict]:
        if not self._server:
            return []
        return self._server.get_logs(n)

    def get_roles(self) -> dict:
        if not self._server:
            return {}
        return self._server.perm_manager.get_all_roles()

    def clear_logs(self, user: str = "dashboard") -> bool:
        if not self._server:
            return False
        return self._server.clear_logs(user=user)
