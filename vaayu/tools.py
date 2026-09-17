"""
Built-in standard tools and schema inference engine for Vaayu SLMM.
Provides:
- infer_schema_from_callable: Automatic JSON schema generation from Python functions & type hints
- FilesystemTool: Sandboxed local file reading, writing, searching, and listing
- ShellTool: Sandboxed command-line process execution
- HttpTool: Network HTTP GET/POST requester
- SqliteTool: Embedded SQLite database query engine
"""

import os
import sys
import json
import inspect
import sqlite3
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, get_type_hints

def python_type_to_json_type(py_type: Any) -> str:
    """Maps Python types to JSON Schema data types."""
    if py_type in (int,):
        return "integer"
    elif py_type in (float,):
        return "number"
    elif py_type in (bool,):
        return "boolean"
    elif py_type in (list, List):
        return "array"
    elif py_type in (dict, Dict):
        return "object"
    return "string"

def infer_schema_from_callable(func: Callable, name_override: Optional[str] = None, desc_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Introspects a Python callable to automatically extract its tool name,
    docstring description, and JSON Schema for parameters.
    """
    func_name = name_override or func.__name__
    doc = desc_override or inspect.getdoc(func) or f"Executes {func_name}."
    sig = inspect.signature(func)
    
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
            
        py_type = type_hints.get(param_name, param.annotation if param.annotation != inspect.Parameter.empty else str)
        json_type = python_type_to_json_type(py_type)
        
        prop_def: Dict[str, Any] = {"type": json_type}
        if param.default != inspect.Parameter.empty and param.default is not None:
            prop_def["default"] = param.default
        else:
            required.append(param_name)
            
        properties[param_name] = prop_def

    return {
        "name": func_name,
        "description": doc.strip().split("\n\n")[0].replace("\n", " "),
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

class FilesystemTool:
    """Safe, sandboxed filesystem access tool."""
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()

    def _resolve_safe(self, rel_path: str) -> Path:
        target = (self.root_dir / rel_path).resolve()
        if not str(target).startswith(str(self.root_dir)):
            raise PermissionError(f"Access denied: path '{rel_path}' is outside sandbox root '{self.root_dir}'.")
        return target

    def read_file(self, path: str, max_bytes: int = 65536) -> str:
        """Reads the content of a file within the workspace root."""
        target = self._resolve_safe(path)
        if not target.is_file():
            raise FileNotFoundError(f"File '{path}' does not exist.")
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)

    def write_file(self, path: str, content: str) -> str:
        """Writes or creates a text file inside the workspace root."""
        target = self._resolve_safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."

    def list_dir(self, path: str = ".") -> List[str]:
        """Lists files and directories located at the given relative path."""
        target = self._resolve_safe(path)
        if not target.is_dir():
            raise NotADirectoryError(f"Path '{path}' is not a directory.")
        return sorted([p.name + ("/" if p.is_dir() else "") for p in target.iterdir()])

    def search_files(self, pattern: str, path: str = ".") -> List[str]:
        """Searches for files matching a glob pattern starting from path (recursive)."""
        target = self._resolve_safe(path)
        if not pattern.startswith("**"):
            matches = target.rglob(pattern)
        else:
            matches = target.glob(pattern)
        return sorted([str(p.relative_to(self.root_dir)).replace("\\", "/") for p in matches if p.is_file()])

class ShellTool:
    """Sandboxed command execution tool with timeout and safety controls."""
    def __init__(self, allowed_commands: Optional[List[str]] = None, timeout: float = 30.0, cwd: str = "."):
        self.allowed_commands = allowed_commands
        self.timeout = timeout
        self.cwd = cwd

    def execute_command(self, command: str) -> Dict[str, Any]:
        """Executes a shell command and captures stdout, stderr, and exit code."""
        cmd_root = command.strip().split()[0] if command.strip() else ""
        if self.allowed_commands and cmd_root not in self.allowed_commands:
            return {"status": "error", "message": f"Command '{cmd_root}' is not in allowed command list."}

        try:
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd
            )
            return {
                "status": "success" if res.returncode == 0 else "error",
                "exit_code": res.returncode,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Command timed out after {self.timeout}s."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class HttpTool:
    """Lightweight HTTP requester for REST endpoints."""
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def http_fetch(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Fetches content from an external HTTP/HTTPS URL."""
        req_headers = {"User-Agent": "Vaayu-SLMM/1.1.0"}
        if headers:
            req_headers.update(headers)
            
        req = urllib.request.Request(url, headers=req_headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")
                return {"status": "success", "status_code": status, "body": body[:65536]}
        except urllib.error.HTTPError as e:
            return {"status": "error", "status_code": e.code, "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class SqliteTool:
    """Local SQLite database manager."""
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path) if db_path == ":memory:" else None

    def _get_connection(self):
        if self._conn is not None:
            return self._conn, False
        return sqlite3.connect(self.db_path), True

    def execute_sql(self, query: str) -> Dict[str, Any]:
        """Executes a SQL query against the SQLite database."""
        conn, close_after = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                if close_after:
                    conn.close()
                return {"status": "success", "row_count": len(results), "rows": results[:100]}
            else:
                conn.commit()
                rowcount = cursor.rowcount
                if close_after:
                    conn.close()
                return {"status": "success", "rows_affected": rowcount}
        except Exception as e:
            if close_after:
                conn.close()
            return {"status": "error", "message": str(e)}

    def close(self):
        """Closes the underlying database connection if persistent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
