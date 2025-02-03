import subprocess

# Define the Docker command as a list of arguments
command = [
    'docker', 'run', '--gpus', 'all', '--rm', '-it',
    '-v', '/dev/shm:/dev/shm', 'forex-trainer'
]

# Execute the command using Popen
with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
    # Iterate over the lines of stdout
    for line in process.stdout:
        print(line, end='')  # Print each line as it is received

    # Wait for the process to complete and get the return code
    return_code = process.wait()

    # Check for any errors in stderr
    if return_code != 0:
        error_output = process.stderr.read()
        print(f"Error: {error_output}")