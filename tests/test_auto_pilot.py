import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from artefacts_toolkit.config import get_artefacts_params


@pytest.fixture(scope="function")
def isaac_sim_process():
    """Fixture to start and stop Isaac Sim as an external process."""

    python_executable = sys.executable 

    try:
        # TODO: Shouldn't this return an empty dict instead of raising?
        params = get_artefacts_params()
        difficulty = params.get("difficulty", 0.5)
    except Exception as _e:
        # Test is running without Artefacts
        difficulty = 0.5
        

    command = [
        str(python_executable),
        "-m",
        "simulation",
        "--scene",
        "generated_pyramid",
        "--use-auto-pilot",
        "--difficulty",
        str(difficulty),
    ]

    print(f"Starting simulation with command: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=0,  # Unbuffered - changed from 1
        preexec_fn=os.setsid,  # Create new process group for clean termination
        env={
            **os.environ, 
            "PYTHONUNBUFFERED": "1",  # Force unbuffered output
        },
    )

    # Give Isaac Sim time to start up and begin publishing data
    startup_time = 10.0  # Adjust based on your system
    print(f"Waiting {startup_time} seconds for Isaac Sim to start...")
    time.sleep(startup_time)

    # Check if process is still running
    if process.poll() is not None:
        # The output will be in the console, so we just need to fail the test
        pytest.fail(
            "Isaac Sim process failed to start. Check the console output above for details."
        )

    yield process

    # Cleanup: terminate the process
    print("Terminating Isaac Sim process...")
    try:
        # Try graceful termination first
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        # Force kill if graceful termination fails
        print("Force killing Isaac Sim process...")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        # Process already terminated
        pass

    print("Isaac Sim process terminated")


def test_integration_isaac_sim_standalone(isaac_sim_process):
    """Test Isaac Sim integration with external process."""
    process = isaac_sim_process

    # Wait for the "All waypoints reached!" message with timeout
    timeout = 50.0  # seconds
    start_time = time.time()
    message_found = False
    target_message = "All waypoints reached!"
    
    print(f"Waiting up to {timeout} seconds for '{target_message}' message...")
    
    while time.time() - start_time < timeout:
        # print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
        # Check if process is still running
        if process.poll() is not None:
            pytest.fail(
                f"Isaac Sim process terminated unexpectedly with return code {process.returncode}"
            )
        
        # Read all available output lines (non-blocking)
        if process.stdout:
            import select
            
            # Check if there's data available to read
            while True:
                # Use select to check if stdout is readable (non-blocking)
                readable, _, _ = select.select([process.stdout], [], [], 0)
                
                if not readable:
                    break
                
                line = process.stdout.readline()
                if not line:
                    break
                    
                line = line.strip()
                if line:
                    print(f"[Isaac Sim] {line}")  # Echo the output
                    
                    if target_message in line:
                        message_found = True
                        break
            
            if message_found:
                break
        
    
    if not message_found:
        pytest.fail(f"Timeout: '{target_message}' message not received within {timeout} seconds")
    
    print(f"Success: '{target_message}' message received in {time.time() - start_time:.2f} seconds")
