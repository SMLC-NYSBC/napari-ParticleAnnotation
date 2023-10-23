from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import subprocess

app = FastAPI()


@app.post("/execute_command/")
async def execute_command(command: str):
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        # If the command fails, return the error output
        output = e.stderr
        return PlainTextResponse(content=output, status_code=500)
    except Exception as e:
        # Handle other exceptions
        raise HTTPException(status_code=500, detail=str(e))

    return PlainTextResponse(content=output)
