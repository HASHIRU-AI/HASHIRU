
import subprocess
import sys
import os
import tempfile

__all__ = ['PythonSandboxTool']

class PythonSandboxTool():
    dependencies = []

    inputSchema = {
        "name": "PythonSandboxTool",
        "description": "Executes Python code in a sandbox environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute."
                }
            },
            "required": ["code"]
        }
    }

    def run(self, **kwargs):
        code = kwargs.get("code")
        if not code:
            return {"status": "error", "message": "Missing required parameter: 'code'", "output": None}

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary file inside the directory
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False, dir=tmpdir) as temp_file:
                temp_file.write(code)
                temp_file_name = temp_file.name

            # Construct the command to execute the Python code
            command = [sys.executable, temp_file_name]

            try:
                # Execute the command in a subprocess
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=10,  # Timeout after 10 seconds
                    check=False  # Do not raise an exception on non-zero exit code
                )

                # Get the output and error messages
                stdout = process.stdout
                stderr = process.stderr

                # Check the return code
                return_code = process.returncode

                # Prepare the result
                result = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": return_code
                }

                # Return the result
                return {"status": "success", "message": "Python code executed successfully.", "output": result}

            except subprocess.TimeoutExpired:
                return {"status": "error", "message": "Python code execution timed out.", "output": None}
            except Exception as e:
                return {"status": "error", "message": f"Python code execution failed: {str(e)}", "output": None}


