"""
    MCP Server for Tölvera

    Current implementations:
        Tools: 
            - get_sketch_code -> passes the full contents of a script to the LLM
            - update_sketch_code -> updates and save the contents of a script to the LLM
            
        Resources:
            - tolvera://docs/guide (https://afhverjuekki.github.io/tolvera/guide/)
            - tolvera://source/pixels (https://github.com/afhverjuekki/tolvera/blob/main/src/tolvera/pixels.py)
        
    Known Issues:
        - Printing out needs to be specified, otherwise can cause issues when running MCP server
        
    TODO:
        - [FEATURE] Add a 'list_sketches' @tool to show available .py files in SKETCH_DIRECTORY
        - [FEATURE] Add a 'create_sketch' tool
        - [CONTEXT] Maybe truncate large resource content?  Need to worry about max tokens hitting a threshold with too many examples
        - [SETUP] Have easier way for user to configure TOLVERA_SRC_DIRECTORY, SKETCH_DIRECTORY, TOLVERA_DOCS_DIRECTORY
        
        
    If using in Claude, update the config.json to look like this:
        {
            "mcpServers": {
                "tolvera": {
                "command": YOUR_PATH_TO_PYTHON_EXE,
                "args": ["{YOUR_PATH_TO_THIS_DIRECTORY}/mcp_server.py"],
                "workingDirectory": YOUR_PATH_TO_THIS_DIRECTORY
                }
            }
        }
"""

import os
import sys
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP, Context

### Make sure to use full-paths for all these globals ###
TOLVERA_SRC_DIRECTORY = ""
TOLVERA_DOCS_DIRECTORY = ""
SKETCH_DIRECTORY = ""

mcp = FastMCP("tolvera") 

### Helper functions to make file loading and saving as robust as possible ###

def get_path(base_dir: str, filename: str, ensure_exists: bool = False) -> str | None:
    """Validates filename and tries to return the absolute path."""
    
    try:
        abs_base_dir = os.path.abspath(base_dir)
        if ".." in filename or filename.startswith(("/", "\\")) or not filename:
            print(f"Error: Invalid filename pattern: {filename}", file=sys.stderr) 
            return None
        abs_file_path = os.path.abspath(os.path.join(abs_base_dir, filename))
        if not abs_file_path.startswith(abs_base_dir):
            print(f"Error: Security violation - attempted path traversal: {filename}", file=sys.stderr)
            return None
        if ensure_exists and not os.path.exists(abs_file_path):
            print(f"Error: File does not exist: {abs_file_path}", file=sys.stderr)
            return None
        return abs_file_path
    except Exception as e:
        print(f"Error during path validation for '{filename}' in '{base_dir}': {e}", file=sys.stderr)
        return None
    
def _read_file(base_dir: str, filename: str, operation_desc: str) -> tuple[str | None, str | None]:
    """Safely reads a file within a base directory."""
    
    safe_path = get_path(base_dir, filename, ensure_exists=True)
    if not safe_path:
        return None, f"Invalid, unsafe, or non-existent path for {operation_desc}: {filename}"
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Helper: Read {operation_desc} from {safe_path}", file=sys.stderr)
        return content, None
    except Exception as e:
        error_msg = f"Error reading {operation_desc} file {filename}: {e}"
        print(error_msg, file=sys.stderr)
        return None, error_msg

def _write_file(base_dir: str, filename: str, content: str, operation_desc: str) -> str | None:
    """Safely writes content to a file within a base directory."""
    
    safe_path = get_path(base_dir, filename, ensure_exists=False)
    if not safe_path: 
        return f"Invalid or unsafe path for {operation_desc}: {filename}"
    try:
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, 'w', encoding='utf-8') as f: 
            f.write(content)
        print(f"Helper: Wrote {operation_desc} to {safe_path}", file=sys.stderr)
        return None 
    except Exception as e:
        error_msg = f"Error writing {operation_desc} file {filename}: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg

### MCP Tool Definitions ###

@mcp.tool() 
def get_sketch_code(sketch_filename: str, ctx: Context = None) -> Dict[str, Any]:
    """Reads the current Python code content of a specified Tolvera sketch file."""
    
    if not sketch_filename.endswith(".py"):
        sketch_filename += ".py"
    content, error = _read_file(SKETCH_DIRECTORY, sketch_filename, "sketch code")
    if error:
        return {"status": "error", "message": error}
    else:
        return {"status": "ok", "filename": sketch_filename, "code": content}

@mcp.tool() 
def update_sketch_code(sketch_filename: str, new_code_content: str, ctx: Context = None) -> str:
    """Overwrites the specified Tolvera sketch file with new Python code content."""
    
    if not sketch_filename.endswith(".py"):
        sketch_filename += ".py"
    error = _write_file(SKETCH_DIRECTORY, sketch_filename, new_code_content, "sketch code")
    if error:
        return f"Error updating sketch: {error}"
    else:
        return f"Sketch '{sketch_filename}' updated successfully. Tolvera should reload via file watcher."


### MCP Resource Definitions ###

@mcp.resource("tolvera://docs/guide") 
async def get_full_guide() -> str: 
    """Provides the full content of guide.md."""
    content, error = _read_file(TOLVERA_DOCS_DIRECTORY, "guide.md", "guide doc")
    if error:
        return f"Error retrieving documentation: {error}" 
    else:
        return content

@mcp.resource("tolvera://docs/gsoc") 
async def get_full_guide() -> str: 
    """Provides the full content of gsoc.md."""
    content, error = _read_file(TOLVERA_DOCS_DIRECTORY, "gsco.md", "gsoc doc")
    if error:
        return f"Error retrieving documentation: {error}" 
    else:
        return content
    
@mcp.resource("tolvera://docs/examples") 
async def get_full_guide() -> str: 
    """Provides the full content of examples.md."""
    content, error = _read_file(TOLVERA_DOCS_DIRECTORY, "examples.md", "examples doc")
    if error:
        return f"Error retrieving documentation: {error}" 
    else:
        return content
    
@mcp.resource("tolvera://docs/experiments") 
async def get_full_guide() -> str: 
    """Provides the full content of examples.md."""
    content, error = _read_file(TOLVERA_DOCS_DIRECTORY, "experiments.md", "experiments doc")
    if error:
        return f"Error retrieving documentation: {error}" 
    else:
        return content

@mcp.resource("tolvera://source/pixels") 
async def get_pixels_source() -> str: 
    """Provides the source code for the Pixels (tv.px) module."""
    content, error = _read_file(TOLVERA_SRC_DIRECTORY, "pixels.py", "pixels source") 
    if error:
        return f"Error retrieving source code: {error}"
    else:
        return content

# Main run through
if __name__ == "__main__":
    print("---------------------------------------------", file=sys.stderr)
    print("Starting Tolvera Editor + Guide MCP Server...", file=sys.stderr)
    print("---------------------------------------------", file=sys.stderr)
    try:
        abs_sketch_dir = os.path.abspath(SKETCH_DIRECTORY)
        abs_docs_dir = os.path.abspath(TOLVERA_DOCS_DIRECTORY)
        abs_src_dir = os.path.abspath(TOLVERA_SRC_DIRECTORY)
        
        # Print paths for debugging
        print(f"Using sketch directory: {abs_sketch_dir}", file=sys.stderr)
        print(f"Using docs directory: {abs_docs_dir}", file=sys.stderr)
        print(f"Using source directory: {abs_src_dir}", file=sys.stderr)
        
        os.makedirs(abs_sketch_dir, exist_ok=True)
        os.makedirs(abs_docs_dir, exist_ok=True) 
        
        mcp.run() 
        
    except PermissionError as e:
        print(f"Failed to start MCP server due to permissions error: {e}", file=sys.stderr)
        print(f"Please check write permissions for the user running Claude Desktop in: {os.path.abspath('.')}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to start MCP server: {e}", file=sys.stderr)