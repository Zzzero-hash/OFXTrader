import subprocess

# Define the Docker command as a list of arguments
build = [
    'docker', 'build', '-t', 'forex-trainer', '.'
]
push = [
    'docker', 'push', 'cpenrod90/forex-trainer:latest'
]
command = [
    'docker', 'run', '--gpus', 'all', '--rm', '-it',
    '-v', '/dev/shm:/dev/shm', 'forex-trainer'
]

# Create a list of commands to execute
commands = [build, push, command]

# Execute the command using Popen
for cmd in commands:
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        # Iterate over the lines of stdout
        for line in process.stdout:
            print(line, end='')  # Print each line as it is received

        # Wait for the process to complete and get the return code
        return_code = process.wait()

        # Check for any errors in stderr
        if return_code != 0:
            error_output = process.stderr.read()
            print(f"Error: {error_output}")