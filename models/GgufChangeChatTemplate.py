import subprocess
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
# input_file = "gemma-4-26B-A4B-it.gguf"
subprocess.run([
    "gguf-new-metadata",
    "--chat-template-file", "chat_template.jinja",
    input_file,
    output_file
])